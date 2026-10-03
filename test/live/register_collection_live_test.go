//go:build live

package live

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	extraStoredColumn          = "extraColumn"
	extraStoredColumnMaxLength = 64
)

// TestRegisterCollectionValidatesMilvusSchemaAcrossRestart ingests into a real
// Milvus collection, restarts the daemon, and registers through both RPCs. Both
// return the same record and leave the checkpoint unchanged. A column added to
// the stored collection outside the declaration then fails registration with
// collection_schema_mismatch, and the stored collection keeps every field.
func TestRegisterCollectionValidatesMilvusSchemaAcrossRestart(t *testing.T) {
	h := newHarness(t)

	ingested := h.upsert(seedConversations(), pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, false, false)
	requireCompleted(t, ingested, "ingest")
	checkpointPath := filepath.Join(h.config.MerkleDir, h.codebaseID+".json")
	checkpoint := readLiveFile(t, checkpointPath)

	h.restart(nil)

	oldResponse, err := h.client.RegisterConversationCollection(correlatedContext(), &pb.RegisterConversationCollectionRequest{
		CollectionId: h.collectionID,
		Client:       &pb.ClientInfo{Name: "live-harness"},
	})
	if err != nil {
		t.Fatalf("RegisterConversationCollection after restart returned error: %v", err)
	}
	genericResponse, err := h.registerConversationDeclaration()
	if err != nil {
		t.Fatalf("RegisterCollection after restart returned error: %v", err)
	}
	for _, codebaseID := range []string{oldResponse.GetCodebaseId(), genericResponse.GetCodebaseId()} {
		if codebaseID != h.codebaseID {
			t.Fatalf("codebase id after restart = %q, want %q", codebaseID, h.codebaseID)
		}
	}
	if genericResponse.GetCollectionName() != h.collectionName {
		t.Fatalf("collection name = %q, want %q", genericResponse.GetCollectionName(), h.collectionName)
	}
	if !bytes.Equal(readLiveFile(t, checkpointPath), checkpoint) {
		t.Fatal("Merkle checkpoint changed across registrations")
	}

	h.addStoredColumn(extraStoredColumn)
	fieldsBefore := h.storedFieldNames()

	_, err = h.registerConversationDeclaration()
	requireLiveColumnError(t, err, extraStoredColumn)
	_, err = h.client.RegisterConversationCollection(correlatedContext(), &pb.RegisterConversationCollectionRequest{
		CollectionId: h.collectionID,
		Client:       &pb.ClientInfo{Name: "live-harness"},
	})
	requireLiveColumnError(t, err, extraStoredColumn)

	fieldsAfter := h.storedFieldNames()
	if len(fieldsAfter) != len(fieldsBefore) {
		t.Fatalf("stored fields after rejected registration = %v, want %v", fieldsAfter, fieldsBefore)
	}
	for index := range fieldsBefore {
		if fieldsAfter[index] != fieldsBefore[index] {
			t.Fatalf("stored fields after rejected registration = %v, want %v", fieldsAfter, fieldsBefore)
		}
	}
	if !bytes.Equal(readLiveFile(t, checkpointPath), checkpoint) {
		t.Fatal("Merkle checkpoint changed after a rejected registration")
	}
}

// TestRegisterCollectionMigratesLegacyMilvusCollection starts from a registry
// record without a saved declaration and a stored collection created without
// the conversation scalar columns. Registration with the conversation
// declaration runs the conversation scalar migration, compares the migrated
// schema, and saves the declaration on the same record.
func TestRegisterCollectionMigratesLegacyMilvusCollection(t *testing.T) {
	h := newHarness(t)

	h.createLegacyCollection()
	h.restart(func() { removeLiveRegistryDeclarations(t, h.config.RegistryPath) })

	response, err := h.registerConversationDeclaration()
	if err != nil {
		t.Fatalf("RegisterCollection for the legacy record returned error: %v", err)
	}
	if response.GetCodebaseId() != h.codebaseID {
		t.Fatalf("codebase id = %q, want %q", response.GetCodebaseId(), h.codebaseID)
	}

	storedFields := h.storedFields()
	for _, column := range semantic.ConversationDeclaration().Scalars {
		field, found := storedFields[column.Name]
		if !found {
			t.Fatalf("stored collection lacks migrated column %s", column.Name)
		}
		if !field.Nullable {
			t.Fatalf("migrated column %s is not nullable", column.Name)
		}
	}

	var registry model.RegistryFile
	if err := json.Unmarshal(readLiveFile(t, h.config.RegistryPath), &registry); err != nil {
		t.Fatalf("decode registry: %v", err)
	}
	declarationSaved := false
	for _, codebase := range registry.Codebases {
		if codebase.ID == h.codebaseID && codebase.Declaration != nil && codebase.Declaration.ItemIDColumn == "conversationId" {
			declarationSaved = true
		}
	}
	if !declarationSaved {
		t.Fatal("registry has no conversation declaration on the legacy record after registration")
	}
	if _, err := os.Stat(filepath.Join(h.config.MerkleDir, h.codebaseID+".json")); err == nil {
		t.Fatal("registration wrote a Merkle checkpoint")
	}
}

// TestRegisterConversationCollectionMigratesUnregisteredLegacyCollection starts
// from a stored collection created without the conversation scalar columns and
// an empty registry. The old conversation RPC still registers it, runs the
// conversation scalar migration, and saves the conversation declaration.
func TestRegisterConversationCollectionMigratesUnregisteredLegacyCollection(t *testing.T) {
	h := newHarness(t)

	h.createLegacyCollection()
	h.restart(func() {
		if err := os.WriteFile(h.config.RegistryPath, []byte(`{"codebases":[]}`), 0o600); err != nil {
			t.Fatalf("write empty registry: %v", err)
		}
	})

	response, err := h.client.RegisterConversationCollection(correlatedContext(), &pb.RegisterConversationCollectionRequest{
		CollectionId: h.collectionID,
		Client:       &pb.ClientInfo{Name: "live-harness"},
	})
	if err != nil {
		t.Fatalf("RegisterConversationCollection for the unregistered legacy collection returned error: %v", err)
	}
	if response.GetCollectionName() != h.collectionName {
		t.Fatalf("collection name = %q, want %q", response.GetCollectionName(), h.collectionName)
	}
	storedFields := h.storedFields()
	for _, column := range semantic.ConversationDeclaration().Scalars {
		if _, found := storedFields[column.Name]; !found {
			t.Fatalf("stored collection lacks migrated column %s", column.Name)
		}
	}
}

func (h *harness) registerConversationDeclaration() (*pb.RegisterCollectionResponse, error) {
	declaration := semantic.ConversationDeclaration()
	scalars := make([]*pb.ScalarColumnDeclaration, 0, len(declaration.Scalars))
	for _, column := range declaration.Scalars {
		scalars = append(scalars, &pb.ScalarColumnDeclaration{
			Column:    column.Name,
			Type:      liveScalarType(column.Type),
			Nullable:  column.Nullable,
			MaxLength: column.MaxLength,
		})
	}
	return h.client.RegisterCollection(correlatedContext(), &pb.RegisterCollectionRequest{
		CollectionId: h.collectionID,
		ItemIdColumn: declaration.ItemIDColumn,
		Scalars:      scalars,
		Client:       &pb.ClientInfo{Name: "live-harness"},
	})
}

func liveScalarType(scalarType model.ScalarType) pb.ScalarColumnType {
	switch scalarType {
	case model.ScalarTypeString:
		return pb.ScalarColumnType_SCALAR_COLUMN_TYPE_STRING
	case model.ScalarTypeBool:
		return pb.ScalarColumnType_SCALAR_COLUMN_TYPE_BOOL
	case model.ScalarTypeInt64:
		return pb.ScalarColumnType_SCALAR_COLUMN_TYPE_INT64
	default:
		return pb.ScalarColumnType_SCALAR_COLUMN_TYPE_UNSPECIFIED
	}
}

// createLegacyCollection creates the harness collection with the base chunk
// schema, its dense vector index, and no conversation scalar columns, the
// schema a conversation collection had before those columns existed.
func (h *harness) createLegacyCollection() {
	h.t.Helper()
	schema := entity.NewSchema().
		WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeVarChar).WithMaxLength(512).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName("content").WithDataType(entity.FieldTypeVarChar).WithMaxLength(65535)).
		WithField(entity.NewField().WithName(relativePathField).WithDataType(entity.FieldTypeVarChar).WithMaxLength(1024)).
		WithField(entity.NewField().WithName("startLine").WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName("endLine").WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName("fileExtension").WithDataType(entity.FieldTypeVarChar).WithMaxLength(32)).
		WithField(entity.NewField().WithName("metadata").WithDataType(entity.FieldTypeVarChar).WithMaxLength(65535)).
		WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(fakeEmbeddingDimension))
	vectorIndex := milvusclient.NewCreateIndexOption(h.collectionName, "vector", index.NewAutoIndex(entity.COSINE))
	createOption := milvusclient.NewCreateCollectionOption(h.collectionName, schema).WithIndexOptions(vectorIndex)
	if err := h.milvus.CreateCollection(correlatedContext(), createOption); err != nil {
		h.t.Fatalf("create legacy collection %s: %v", h.collectionName, err)
	}
}

func (h *harness) addStoredColumn(name string) {
	h.t.Helper()
	field := entity.NewField().
		WithName(name).
		WithDataType(entity.FieldTypeVarChar).
		WithMaxLength(extraStoredColumnMaxLength).
		WithNullable(true)
	if err := h.milvus.AddCollectionField(correlatedContext(), milvusclient.NewAddCollectionFieldOption(h.collectionName, field)); err != nil {
		h.t.Fatalf("add column %s to %s: %v", name, h.collectionName, err)
	}
}

func (h *harness) storedFields() map[string]*entity.Field {
	h.t.Helper()
	collection, err := h.milvus.DescribeCollection(correlatedContext(), milvusclient.NewDescribeCollectionOption(h.collectionName))
	if err != nil {
		h.t.Fatalf("describe %s: %v", h.collectionName, err)
	}
	fields := make(map[string]*entity.Field, len(collection.Schema.Fields))
	for _, field := range collection.Schema.Fields {
		fields[field.Name] = field
	}
	return fields
}

func (h *harness) storedFieldNames() []string {
	h.t.Helper()
	collection, err := h.milvus.DescribeCollection(correlatedContext(), milvusclient.NewDescribeCollectionOption(h.collectionName))
	if err != nil {
		h.t.Fatalf("describe %s: %v", h.collectionName, err)
	}
	names := make([]string, 0, len(collection.Schema.Fields))
	for _, field := range collection.Schema.Fields {
		names = append(names, field.Name)
	}
	return names
}

func readLiveFile(t *testing.T, path string) []byte {
	t.Helper()
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return content
}

// removeLiveRegistryDeclarations rewrites the registry file in the format a
// daemon wrote before declarations were saved: every record without the
// declaration key. It fails when no record had one.
func removeLiveRegistryDeclarations(t *testing.T, registryPath string) {
	t.Helper()
	var registry map[string]json.RawMessage
	if err := json.Unmarshal(readLiveFile(t, registryPath), &registry); err != nil {
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

func requireLiveColumnError(t *testing.T, err error, wantColumn string) {
	t.Helper()
	if err == nil {
		t.Fatalf("registration succeeded, want collection_schema_mismatch on column %q", wantColumn)
	}
	grpcStatus, ok := status.FromError(err)
	if !ok {
		t.Fatalf("error %v has no gRPC status", err)
	}
	if grpcStatus.Code() != codes.FailedPrecondition {
		t.Fatalf("status code = %s, want FailedPrecondition: %v", grpcStatus.Code(), err)
	}
	for _, detail := range grpcStatus.Details() {
		info, isErrorInfo := detail.(*errdetails.ErrorInfo)
		if !isErrorInfo {
			continue
		}
		if info.GetReason() != adapterr.CodeCollectionSchemaMismatch {
			t.Fatalf("ErrorInfo reason = %q, want %q", info.GetReason(), adapterr.CodeCollectionSchemaMismatch)
		}
		if got := info.GetMetadata()[adapterr.ErrorInfoColumnKey]; got != wantColumn {
			t.Fatalf("ErrorInfo column = %q, want %q", got, wantColumn)
		}
		return
	}
	t.Fatalf("status %v has no ErrorInfo detail", err)
}
