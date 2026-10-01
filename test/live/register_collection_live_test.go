//go:build live

package live

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	extraStoredColumn          = "extraColumn"
	extraStoredColumnMaxLength = 64
)

func TestRegisterCollectionValidatesMilvusSchemaAcrossRestart(t *testing.T) {
	h := newHarness(t)

	ingested := h.upsert(seedItems(), pb.CollectionReconcileMode_COLLECTION_RECONCILE_MODE_RETAIN, false, false)
	requireCompleted(t, ingested, "ingest")
	checkpointPath := filepath.Join(h.config.MerkleDir, h.codebaseID+".json")
	checkpoint := readLiveFile(t, checkpointPath)

	h.restart(nil)

	genericResponse, err := h.registerLiveDeclaration()
	if err != nil {
		t.Fatalf("RegisterCollection after restart returned error: %v", err)
	}
	for _, codebaseID := range []string{genericResponse.GetCodebaseId()} {
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

	_, err = h.registerLiveDeclaration()
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

func (h *harness) registerLiveDeclaration() (*pb.RegisterCollectionResponse, error) {
	declaration := liveCollectionDeclaration()
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
