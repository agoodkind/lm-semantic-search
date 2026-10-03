package daemon

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/store"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

const collectionTestManagerCloseTimeout = 15 * time.Second

// offlineCollectionDaemon is a real daemon over the offline local vector store,
// served on a unix socket and driven through its gRPC client. A restart keeps
// the state root. The registry, Merkle checkpoints, and local rows persist
// across daemon processes the way they do for the installed daemon.
type offlineCollectionDaemon struct {
	t          *testing.T
	config     config.Config
	manager    *Manager
	connection *grpc.ClientConn
	client     pb.SemanticSearchDaemonServiceClient
	stopServer func()
}

func newOfflineCollectionDaemon(t *testing.T) *offlineCollectionDaemon {
	t.Helper()
	return newOfflineCollectionDaemonWithEmbedder(t, newTestEmbeddingServer(t).URL)
}

// newOfflineCollectionDaemonWithEmbedder starts an offline collection daemon
// that embeds through the OpenAI-compatible endpoint at embeddingURL.
func newOfflineCollectionDaemonWithEmbedder(t *testing.T, embeddingURL string) *offlineCollectionDaemon {
	t.Helper()
	stateRoot := t.TempDir()
	socketDirectory, err := os.MkdirTemp("", "lms-coll-")
	if err != nil {
		t.Fatalf("create socket directory: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(socketDirectory) })
	cfg := config.Config{
		StateRoot:                 stateRoot,
		SocketPath:                filepath.Join(socketDirectory, "d.sock"),
		RegistryPath:              filepath.Join(stateRoot, "registry.json"),
		JobsPath:                  filepath.Join(stateRoot, "jobs.jsonl"),
		EventsPath:                filepath.Join(stateRoot, "events.jsonl"),
		LogsDir:                   filepath.Join(stateRoot, "logs"),
		LogPath:                   filepath.Join(stateRoot, "logs", "lm-semantic-search-daemon.log"),
		MerkleDir:                 filepath.Join(stateRoot, "merkle"),
		LocksDir:                  filepath.Join(stateRoot, "locks"),
		SocketsDir:                filepath.Join(stateRoot, "sockets"),
		ChunksDir:                 filepath.Join(stateRoot, "chunks"),
		GraphDir:                  filepath.Join(stateRoot, "graph"),
		ContextRoot:               filepath.Join(stateRoot, "context"),
		IndexBackend:              config.IndexBackendLocal,
		EmbeddingProvider:         config.EmbeddingProviderOpenAI,
		EmbeddingModel:            "text-embedding-3-small",
		EmbeddingBatchSize:        8,
		EmbeddingBatchTokenBudget: 1000,
		EmbeddingDimension:        3,
		OpenAIAPIKey:              "test-key",
		OpenAIBaseURL:             embeddingURL,
		MaxConcurrentIndexJobs:    1,
	}
	for _, directory := range []string{cfg.StateRoot, cfg.LogsDir, cfg.MerkleDir, cfg.LocksDir, cfg.SocketsDir, cfg.ChunksDir, cfg.GraphDir, cfg.ContextRoot} {
		if err := store.EnsureDir(directory); err != nil {
			t.Fatalf("EnsureDir(%s) returned error: %v", directory, err)
		}
	}
	if err := store.WriteRegistry(cfg.RegistryPath, model.RegistryFile{}); err != nil {
		t.Fatalf("WriteRegistry returned error: %v", err)
	}
	daemon := &offlineCollectionDaemon{t: t, config: cfg}
	daemon.start()
	t.Cleanup(daemon.stop)
	return daemon
}

func (daemon *offlineCollectionDaemon) start() {
	daemon.t.Helper()
	manager, err := NewManager(context.Background(), daemon.config)
	if err != nil {
		daemon.t.Fatalf("NewManager returned error: %v", err)
	}
	if manager.semantic.BackendName() != config.IndexBackendLocal {
		daemon.t.Fatalf("semantic backend = %q, want the local vector store", manager.semantic.BackendName())
	}
	daemon.manager = manager
	daemon.stopServer = startGRPCServerForTest(daemon.t, manager, daemon.config.SocketPath)
	connection, client, err := grpcutil.DialDaemon(context.Background(), daemon.config.SocketPath)
	if err != nil {
		daemon.t.Fatalf("DialDaemon returned error: %v", err)
	}
	daemon.connection = connection
	daemon.client = client
}

func (daemon *offlineCollectionDaemon) stop() {
	if daemon.manager == nil {
		return
	}
	_ = daemon.connection.Close()
	daemon.stopServer()
	ctx, cancel := context.WithTimeout(context.Background(), collectionTestManagerCloseTimeout)
	defer cancel()
	if err := daemon.manager.Close(ctx); err != nil {
		daemon.t.Errorf("manager.Close returned error: %v", err)
	}
	daemon.manager = nil
}

// restart stops the daemon, runs between while no daemon owns the state root,
// and starts a new daemon over the same state root.
func (daemon *offlineCollectionDaemon) restart(between func()) {
	daemon.t.Helper()
	daemon.stop()
	if between != nil {
		between()
	}
	daemon.start()
}

func (daemon *offlineCollectionDaemon) registerCollection(collectionID string, itemIDColumn string, scalars []*pb.ScalarColumnDeclaration) (*pb.RegisterCollectionResponse, error) {
	return daemon.client.RegisterCollection(grpcutil.WithCorrelation(context.Background()), &pb.RegisterCollectionRequest{
		CollectionId: collectionID,
		ItemIdColumn: itemIDColumn,
		Scalars:      scalars,
		Client:       &pb.ClientInfo{Name: "collection-test"},
	})
}

func (daemon *offlineCollectionDaemon) registerConversationCollection(collectionID string) (*pb.RegisterConversationCollectionResponse, error) {
	return daemon.client.RegisterConversationCollection(grpcutil.WithCorrelation(context.Background()), &pb.RegisterConversationCollectionRequest{
		CollectionId: collectionID,
		Client:       &pb.ClientInfo{Name: "collection-test"},
	})
}

// ingestConversation streams one two-message conversation through the
// conversation upsert RPC and waits for the ingest job to complete. The job
// writes local rows and the Merkle checkpoint.
func (daemon *offlineCollectionDaemon) ingestConversation(collectionID string, conversationID string) {
	daemon.t.Helper()
	stream, err := daemon.client.UpsertConversationDocumentsStream(grpcutil.WithCorrelation(context.Background()))
	if err != nil {
		daemon.t.Fatalf("open UpsertConversationDocumentsStream returned error: %v", err)
	}
	chunks := []*pb.UpsertConversationDocumentsChunk{
		{Chunk: &pb.UpsertConversationDocumentsChunk_Header{Header: &pb.UpsertConversationDocumentsHeader{
			CollectionId: collectionID,
			Client:       &pb.ClientInfo{Name: "collection-test"},
		}}},
		{Chunk: &pb.UpsertConversationDocumentsChunk_Documents{Documents: &pb.UpsertConversationDocumentsDocuments{
			Documents: []*pb.ConversationDocument{
				{ConversationId: conversationID, MessageIndex: 0, Role: "user", TimestampUnix: 1712345678, Text: "how do collections register their schema"},
				{ConversationId: conversationID, MessageIndex: 1, Role: "assistant", TimestampUnix: 1712345679, Text: "registration saves the declared scalar columns"},
			},
		}}},
		{Chunk: &pb.UpsertConversationDocumentsChunk_Manifest{Manifest: &pb.UpsertConversationDocumentsManifest{
			Manifest: []*pb.ConversationFingerprint{{ConversationId: conversationID, Fingerprint: "fingerprint-" + conversationID}},
		}}},
	}
	for _, chunk := range chunks {
		if err := stream.Send(chunk); err != nil {
			daemon.t.Fatalf("send upsert chunk returned error: %v", err)
		}
	}
	response, err := stream.CloseAndRecv()
	if err != nil {
		daemon.t.Fatalf("CloseAndRecv returned error: %v", err)
	}
	job := waitForRPCJobTerminal(daemon.t, daemon.client, response.GetJobId())
	if job.GetState() != string(model.JobStateCompleted) {
		daemon.t.Fatalf("ingest job state = %q, want completed: %+v", job.GetState(), job.GetError())
	}
}

func (daemon *offlineCollectionDaemon) checkpointPath(codebaseID string) string {
	return filepath.Join(daemon.config.MerkleDir, codebaseID+".json")
}

// documentCollectionRecords lists the tracked records with the chat:/// path
// of collectionID as their canonical path.
func (daemon *offlineCollectionDaemon) documentCollectionRecords(collectionID string) []*pb.Codebase {
	daemon.t.Helper()
	response, err := daemon.client.ListIndexes(grpcutil.WithCorrelation(context.Background()), &pb.ListIndexesRequest{})
	if err != nil {
		daemon.t.Fatalf("ListIndexes returned error: %v", err)
	}
	records := make([]*pb.Codebase, 0)
	for _, codebase := range response.GetIndexes() {
		if codebase.GetCanonicalPath() == conversationCanonicalPath(collectionID) {
			records = append(records, codebase)
		}
	}
	return records
}

func readFileBytes(t *testing.T, path string) []byte {
	t.Helper()
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return content
}

func requireFileAbsent(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("stat %s returned %v, want not exist", path, err)
	}
}

// removeRegistryDeclarations rewrites the registry file in the format a daemon
// wrote before declarations were saved: every record without the declaration
// key. It fails when no record had one, so a test cannot simulate the old
// format against a registry that never saved a declaration.
func removeRegistryDeclarations(t *testing.T, registryPath string) {
	t.Helper()
	var registry map[string]json.RawMessage
	if err := json.Unmarshal(readFileBytes(t, registryPath), &registry); err != nil {
		t.Fatalf("decode registry: %v", err)
	}
	var codebases []map[string]json.RawMessage
	if err := json.Unmarshal(registry["codebases"], &codebases); err != nil {
		t.Fatalf("decode registry codebases: %v", err)
	}
	removed := 0
	for _, codebase := range codebases {
		if _, found := codebase["declaration"]; found {
			delete(codebase, "declaration")
			removed++
		}
	}
	if removed == 0 {
		t.Fatal("registry has no saved declaration to remove")
	}
	encodedCodebases, err := json.Marshal(codebases)
	if err != nil {
		t.Fatalf("encode registry codebases: %v", err)
	}
	registry["codebases"] = encodedCodebases
	encodedRegistry, err := json.Marshal(registry)
	if err != nil {
		t.Fatalf("encode registry: %v", err)
	}
	if err := os.WriteFile(registryPath, encodedRegistry, 0o600); err != nil {
		t.Fatalf("write registry: %v", err)
	}
}

// savedRegistryDeclaration reads the declaration saved on collectionID's
// record from the registry file.
func savedRegistryDeclaration(t *testing.T, registryPath string, collectionID string) *model.CollectionDeclaration {
	t.Helper()
	var registry model.RegistryFile
	if err := json.Unmarshal(readFileBytes(t, registryPath), &registry); err != nil {
		t.Fatalf("decode registry: %v", err)
	}
	for _, codebase := range registry.Codebases {
		if codebase.CanonicalPath == conversationCanonicalPath(collectionID) {
			return codebase.Declaration
		}
	}
	t.Fatalf("registry has no record for collection %s", collectionID)
	return nil
}

// requireColumnError decodes err's gRPC status and requires code, the
// ErrorInfo reason, and the ErrorInfo column metadata.
func requireColumnError(t *testing.T, err error, wantCode codes.Code, wantReason string, wantColumn string) {
	t.Helper()
	if err == nil {
		t.Fatalf("registration succeeded, want %s with reason %s on column %q", wantCode, wantReason, wantColumn)
	}
	grpcStatus, ok := status.FromError(err)
	if !ok {
		t.Fatalf("error %v has no gRPC status", err)
	}
	if grpcStatus.Code() != wantCode {
		t.Fatalf("status code = %s, want %s: %v", grpcStatus.Code(), wantCode, err)
	}
	var errorInfo *errdetails.ErrorInfo
	for _, detail := range grpcStatus.Details() {
		if info, isErrorInfo := detail.(*errdetails.ErrorInfo); isErrorInfo {
			errorInfo = info
		}
	}
	if errorInfo == nil {
		t.Fatalf("status %v has no ErrorInfo detail", err)
	}
	if errorInfo.GetReason() != wantReason {
		t.Fatalf("ErrorInfo reason = %q, want %q", errorInfo.GetReason(), wantReason)
	}
	if got := errorInfo.GetMetadata()[adapterr.ErrorInfoColumnKey]; got != wantColumn {
		t.Fatalf("ErrorInfo column = %q, want %q (metadata %v)", got, wantColumn, errorInfo.GetMetadata())
	}
}

func requireScalars(t *testing.T, label string, got []*pb.ScalarColumnDeclaration, want []*pb.ScalarColumnDeclaration) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d scalars, want %d: %v", label, len(got), len(want), got)
	}
	for index := range want {
		if !proto.Equal(got[index], want[index]) {
			t.Fatalf("%s: scalar %d = %v, want %v", label, index, got[index], want[index])
		}
	}
}

func documentScalars() []*pb.ScalarColumnDeclaration {
	return []*pb.ScalarColumnDeclaration{
		{Column: "docId", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING, Nullable: false, MaxLength: 128},
		{Column: "title", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING, Nullable: true, MaxLength: 512},
		{Column: "pinned", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_BOOL, Nullable: true, MaxLength: 0},
		{Column: "rank", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_INT64, Nullable: false, MaxLength: 0},
	}
}

func conversationScalarsPB() []*pb.ScalarColumnDeclaration {
	return scalarColumnsToPB(semantic.ConversationDeclaration().Scalars)
}

// TestRegisterCollectionPersistsDeclarationAcrossRestart registers a generic
// collection, restarts the daemon, and registers it again. The record, its
// saved declaration, and the absent checkpoint all survive unchanged.
func TestRegisterCollectionPersistsDeclarationAcrossRestart(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)

	first, err := daemon.registerCollection("docs-persist", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}
	if first.GetCodebaseId() == "" || first.GetCollectionName() == "" {
		t.Fatalf("RegisterCollection returned codebase %q collection %q, want both set", first.GetCodebaseId(), first.GetCollectionName())
	}
	if first.GetItemIdColumn() != "docId" {
		t.Fatalf("item id column = %q, want docId", first.GetItemIdColumn())
	}
	requireScalars(t, "first registration", first.GetScalars(), documentScalars())
	checkpointPath := daemon.checkpointPath(first.GetCodebaseId())
	requireFileAbsent(t, checkpointPath)

	daemon.restart(nil)

	second, err := daemon.registerCollection("docs-persist", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection after restart returned error: %v", err)
	}
	if second.GetCodebaseId() != first.GetCodebaseId() {
		t.Fatalf("codebase id after restart = %q, want %q", second.GetCodebaseId(), first.GetCodebaseId())
	}
	if second.GetCollectionName() != first.GetCollectionName() {
		t.Fatalf("collection name after restart = %q, want %q", second.GetCollectionName(), first.GetCollectionName())
	}
	requireScalars(t, "registration after restart", second.GetScalars(), documentScalars())
	saved := savedRegistryDeclaration(t, daemon.config.RegistryPath, "docs-persist")
	if saved == nil || saved.ItemIDColumn != "docId" {
		t.Fatalf("registry declaration = %+v, want item id column docId", saved)
	}
	requireScalars(t, "registry declaration", scalarColumnsToPB(saved.Scalars), documentScalars())
	requireFileAbsent(t, checkpointPath)
	if records := daemon.documentCollectionRecords("docs-persist"); len(records) != 1 {
		t.Fatalf("records for docs-persist = %d, want 1", len(records))
	}
}

// TestRegisterCollectionConflictReturnsSchemaMismatch registers a generic
// collection and then conflicting declarations. Each conflict fails with the
// collection_schema_mismatch reason and the conflicting column, and the saved
// declaration still registers afterward.
func TestRegisterCollectionConflictReturnsSchemaMismatch(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)

	if _, err := daemon.registerCollection("docs-conflict", "docId", documentScalars()); err != nil {
		t.Fatalf("RegisterCollection returned error: %v", err)
	}

	longerTitle := documentScalars()
	longerTitle[1].MaxLength = 1024
	_, err := daemon.registerCollection("docs-conflict", "docId", longerTitle)
	requireColumnError(t, err, codes.FailedPrecondition, adapterr.CodeCollectionSchemaMismatch, "title")

	nonNullableTitle := documentScalars()
	nonNullableTitle[1].Nullable = false
	_, err = daemon.registerCollection("docs-conflict", "docId", nonNullableTitle)
	requireColumnError(t, err, codes.FailedPrecondition, adapterr.CodeCollectionSchemaMismatch, "title")

	omittedRank := documentScalars()[:3]
	_, err = daemon.registerCollection("docs-conflict", "docId", omittedRank)
	requireColumnError(t, err, codes.FailedPrecondition, adapterr.CodeCollectionSchemaMismatch, "rank")

	addedColumn := append(documentScalars(), &pb.ScalarColumnDeclaration{Column: "author", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING, Nullable: true, MaxLength: 64})
	_, err = daemon.registerCollection("docs-conflict", "docId", addedColumn)
	requireColumnError(t, err, codes.FailedPrecondition, adapterr.CodeCollectionSchemaMismatch, "author")

	_, err = daemon.registerCollection("docs-conflict", "title", documentScalars())
	requireColumnError(t, err, codes.FailedPrecondition, adapterr.CodeCollectionSchemaMismatch, "title")

	again, err := daemon.registerCollection("docs-conflict", "docId", documentScalars())
	if err != nil {
		t.Fatalf("RegisterCollection with the saved declaration returned error: %v", err)
	}
	requireScalars(t, "saved declaration", again.GetScalars(), documentScalars())
}

// TestConversationRegistrationRPCsResolveOneRecord registers a conversation
// collection through the old RPC, ingests a conversation, and registers the
// same collection through the generic RPC with the conversation declaration,
// before and after a restart. Every registration returns the same record, and
// the ingest checkpoint stays byte-identical.
func TestConversationRegistrationRPCsResolveOneRecord(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)

	old, err := daemon.registerConversationCollection("conv-shared")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	daemon.ingestConversation("conv-shared", "claude:conv-shared-1")
	checkpointPath := daemon.checkpointPath(old.GetCodebaseId())
	checkpoint := readFileBytes(t, checkpointPath)

	generic, err := daemon.registerCollection("conv-shared", "conversationId", conversationScalarsPB())
	if err != nil {
		t.Fatalf("RegisterCollection with the conversation declaration returned error: %v", err)
	}
	if generic.GetCodebaseId() != old.GetCodebaseId() || generic.GetCollectionName() != old.GetCollectionName() {
		t.Fatalf("generic registration = %s/%s, want %s/%s", generic.GetCodebaseId(), generic.GetCollectionName(), old.GetCodebaseId(), old.GetCollectionName())
	}
	requireScalars(t, "conversation declaration", generic.GetScalars(), conversationScalarsPB())

	conflicting := conversationScalarsPB()
	conflicting[4].MaxLength = 2048
	_, err = daemon.registerCollection("conv-shared", "conversationId", conflicting)
	requireColumnError(t, err, codes.FailedPrecondition, adapterr.CodeCollectionSchemaMismatch, conflicting[4].GetColumn())

	daemon.restart(nil)

	oldAgain, err := daemon.registerConversationCollection("conv-shared")
	if err != nil {
		t.Fatalf("RegisterConversationCollection after restart returned error: %v", err)
	}
	genericAgain, err := daemon.registerCollection("conv-shared", "conversationId", conversationScalarsPB())
	if err != nil {
		t.Fatalf("RegisterCollection after restart returned error: %v", err)
	}
	for _, codebaseID := range []string{oldAgain.GetCodebaseId(), genericAgain.GetCodebaseId()} {
		if codebaseID != old.GetCodebaseId() {
			t.Fatalf("codebase id after restart = %q, want %q", codebaseID, old.GetCodebaseId())
		}
	}
	if records := daemon.documentCollectionRecords("conv-shared"); len(records) != 1 {
		t.Fatalf("records for conv-shared = %d, want 1", len(records))
	}
	if !bytes.Equal(readFileBytes(t, checkpointPath), checkpoint) {
		t.Fatal("Merkle checkpoint changed across registrations")
	}
}

// TestRegisterCollectionReorderedConversationDeclaration registers the
// conversation declaration with its columns in reverse order. Registration
// compares declarations without regard to column order, and the conversation
// declaration check does too. The collection keeps conversation behavior: a
// generic row stores the conversation fields and no generic scalars, and after
// a restart the conversation manifest RPC accepts the collection.
func TestRegisterCollectionReorderedConversationDeclaration(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)

	reordered := conversationScalarsPB()
	slices.Reverse(reordered)
	registered, err := daemon.registerCollection("conv-reordered", "conversationId", reordered)
	if err != nil {
		t.Fatalf("RegisterCollection with the reordered conversation declaration returned error: %v", err)
	}
	conversationID := "claude:reordered-1"
	row := &pb.CollectionRow{
		RowKey:  "conv/" + conversationID + "/0",
		ItemId:  conversationID,
		Text:    "a message stored under a reordered declaration",
		Scalars: []*pb.CollectionScalarValue{stringScalar("role", "user"), int64Scalar("messageIndex", 0)},
	}
	daemon.upsertItems(
		collectionHeader("conv-reordered", pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_UNSPECIFIED, false, false),
		[]*pb.CollectionRow{row},
		map[string]string{conversationID: "fp-reordered"},
	)
	stored := rowByPath(t, daemon.localRows(registered.GetCollectionName()), row.GetRowKey())
	if stored.ConversationID != conversationID || stored.Role != "user" || stored.Scalars != nil {
		t.Fatalf("stored row has conversationId %q, role %q, and scalars %v, want the conversation fields and no generic scalars", stored.ConversationID, stored.Role, stored.Scalars)
	}

	daemon.restart(nil)
	response, err := daemon.client.SyncConversationManifest(grpcutil.WithCorrelation(context.Background()), &pb.SyncConversationManifestRequest{
		CollectionId: "conv-reordered",
		Manifest:     []*pb.ConversationFingerprint{{ConversationId: conversationID, Fingerprint: "fp-reordered"}},
	})
	if err != nil {
		t.Fatalf("SyncConversationManifest after a reordered registration returned error: %v", err)
	}
	if needed := response.GetNeededConversationIds(); len(needed) != 0 {
		t.Fatalf("needed after ingest = %v, want none", needed)
	}
}

// TestRegisterCollectionAdoptsLegacyConversationRecord starts from a registry
// written before declarations were saved. A generic declaration conflicts with
// the legacy conversation declaration. The conversation declaration registers
// against the saved record and its local rows, saves the declaration, and
// leaves the checkpoint unchanged.
func TestRegisterCollectionAdoptsLegacyConversationRecord(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)

	old, err := daemon.registerConversationCollection("conv-legacy")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	daemon.ingestConversation("conv-legacy", "claude:conv-legacy-1")
	checkpointPath := daemon.checkpointPath(old.GetCodebaseId())
	checkpoint := readFileBytes(t, checkpointPath)
	daemon.restart(func() { removeRegistryDeclarations(t, daemon.config.RegistryPath) })

	_, err = daemon.registerCollection("conv-legacy", "docId", documentScalars())
	requireColumnError(t, err, codes.FailedPrecondition, adapterr.CodeCollectionSchemaMismatch, "docId")
	if saved := savedRegistryDeclaration(t, daemon.config.RegistryPath, "conv-legacy"); saved != nil {
		t.Fatalf("conflicting registration saved declaration %+v, want none", saved)
	}

	adopted, err := daemon.registerCollection("conv-legacy", "conversationId", conversationScalarsPB())
	if err != nil {
		t.Fatalf("RegisterCollection with the conversation declaration returned error: %v", err)
	}
	if adopted.GetCodebaseId() != old.GetCodebaseId() {
		t.Fatalf("adopted codebase id = %q, want %q", adopted.GetCodebaseId(), old.GetCodebaseId())
	}
	saved := savedRegistryDeclaration(t, daemon.config.RegistryPath, "conv-legacy")
	if saved == nil || saved.ItemIDColumn != "conversationId" {
		t.Fatalf("registry declaration = %+v, want the conversation declaration", saved)
	}
	requireScalars(t, "adopted declaration", scalarColumnsToPB(saved.Scalars), conversationScalarsPB())
	if !bytes.Equal(readFileBytes(t, checkpointPath), checkpoint) {
		t.Fatal("Merkle checkpoint changed during legacy adoption")
	}
}

// TestRegisterCollectionComparesLocalRows registers a generic declaration for
// a collection id with existing local rows and no registry record. The local
// rows store the conversation scalar fields. The registration fails with a
// schema mismatch, creates no record, and keeps the rows.
func TestRegisterCollectionComparesLocalRows(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)

	old, err := daemon.registerConversationCollection("conv-orphan")
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	daemon.ingestConversation("conv-orphan", "claude:conv-orphan-1")
	localRowsPath := filepath.Join(daemon.config.StateRoot, "localvec", old.GetCollectionName(), "metadata.jsonl")
	rowsBefore := readFileBytes(t, localRowsPath)
	daemon.restart(func() {
		if err := store.WriteRegistry(daemon.config.RegistryPath, model.RegistryFile{}); err != nil {
			t.Fatalf("WriteRegistry returned error: %v", err)
		}
	})

	scalars := append(documentScalars(), conversationScalarsPB()...)
	_, err = daemon.registerCollection("conv-orphan", "docId", scalars)
	requireColumnError(t, err, codes.FailedPrecondition, adapterr.CodeCollectionSchemaMismatch, "docId")
	if records := daemon.documentCollectionRecords("conv-orphan"); len(records) != 0 {
		t.Fatalf("records for conv-orphan = %d, want 0 after a rejected registration", len(records))
	}
	if !bytes.Equal(readFileBytes(t, localRowsPath), rowsBefore) {
		t.Fatal("local rows changed after a rejected registration")
	}
}

// TestRegisterCollectionRejectsInvalidDeclarations sends invalid declarations
// through the gRPC boundary. Each fails with InvalidArgument and the rejected
// column, and none creates a record.
func TestRegisterCollectionRejectsInvalidDeclarations(t *testing.T) {
	t.Parallel()
	daemon := newOfflineCollectionDaemon(t)

	duplicate := append(documentScalars(), &pb.ScalarColumnDeclaration{Column: "title", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING, Nullable: true, MaxLength: 512})
	reserved := append(documentScalars(), &pb.ScalarColumnDeclaration{Column: "relativePath", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING, Nullable: true, MaxLength: 64})
	unsupported := append(documentScalars(), &pb.ScalarColumnDeclaration{Column: "score", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_UNSPECIFIED, Nullable: true, MaxLength: 0})
	missingLength := append(documentScalars(), &pb.ScalarColumnDeclaration{Column: "summary", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING, Nullable: true, MaxLength: 0})
	boolWithLength := append(documentScalars(), &pb.ScalarColumnDeclaration{Column: "hidden", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_BOOL, Nullable: true, MaxLength: 8})
	badIdentifier := append(documentScalars(), &pb.ScalarColumnDeclaration{Column: "9lives", Type: pb.ScalarColumnType_SCALAR_COLUMN_TYPE_INT64, Nullable: true, MaxLength: 0})
	cases := []struct {
		name         string
		itemIDColumn string
		scalars      []*pb.ScalarColumnDeclaration
		wantColumn   string
	}{
		{name: "duplicate column", itemIDColumn: "docId", scalars: duplicate, wantColumn: "title"},
		{name: "reserved column", itemIDColumn: "docId", scalars: reserved, wantColumn: "relativePath"},
		{name: "unsupported type", itemIDColumn: "docId", scalars: unsupported, wantColumn: "score"},
		{name: "string without max length", itemIDColumn: "docId", scalars: missingLength, wantColumn: "summary"},
		{name: "bool with max length", itemIDColumn: "docId", scalars: boolWithLength, wantColumn: "hidden"},
		{name: "invalid identifier", itemIDColumn: "docId", scalars: badIdentifier, wantColumn: "9lives"},
		{name: "undeclared item id column", itemIDColumn: "externalId", scalars: documentScalars(), wantColumn: "externalId"},
		{name: "non-string item id column", itemIDColumn: "rank", scalars: documentScalars(), wantColumn: "rank"},
	}
	for index, testCase := range cases {
		collectionID := fmt.Sprintf("docs-invalid-%d", index)
		_, err := daemon.registerCollection(collectionID, testCase.itemIDColumn, testCase.scalars)
		if err == nil {
			t.Fatalf("%s: registration succeeded, want InvalidArgument", testCase.name)
		}
		requireColumnError(t, err, codes.InvalidArgument, "invalid_argument", testCase.wantColumn)
		if records := daemon.documentCollectionRecords(collectionID); len(records) != 0 {
			t.Fatalf("%s: created %d records, want 0", testCase.name, len(records))
		}
	}

	_, err := daemon.registerCollection("docs-missing-item-id", "", documentScalars())
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("missing item_id_column returned %v, want InvalidArgument", err)
	}
	if records := daemon.documentCollectionRecords("docs-missing-item-id"); len(records) != 0 {
		t.Fatalf("missing item_id_column created %d records, want 0", len(records))
	}
}
