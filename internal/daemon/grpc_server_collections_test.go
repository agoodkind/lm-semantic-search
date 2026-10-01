package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/model"
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
		if codebase.GetCanonicalPath() == documentCanonicalPath(collectionID) {
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
		if codebase.CanonicalPath == documentCanonicalPath(collectionID) {
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
