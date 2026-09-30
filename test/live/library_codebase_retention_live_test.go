//go:build live

package live

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"maps"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/tshash"
)

func TestLibraryCodebaseClearRetainsVectorsAndHistoricalCollection(t *testing.T) {
	harness := newLibraryHarness(t)
	root := t.TempDir()
	writeCodebaseFile(t, root, "retained.go", goFile(goFunction("Retained", "retainedmarker")))
	legacyDaemon := newCodebaseLiveDaemonWithHarness(t, harness, config.CodebaseStoreSemantic)
	legacyDaemon.index(t, root)
	canonicalRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatalf("resolve codebase root: %v", err)
	}
	collectionPrefix := "code_chunks_"
	if legacyDaemon.config.HybridMode {
		collectionPrefix = "hybrid_code_chunks_"
	}
	legacyCollection := collectionPrefix + tshash.PathPrefix(canonicalRoot)
	legacyDescription, err := harness.milvus.DescribeCollection(harness.context(), milvusclient.NewDescribeCollectionOption(legacyCollection))
	if err != nil {
		t.Fatalf("describe historical collection: %v", err)
	}
	legacyBefore := readHistoricalCodeRows(t, harness, legacyCollection)
	if len(legacyBefore) == 0 {
		t.Fatal("the historical code collection has no rows")
	}
	libraryDaemon := newCodebaseLiveDaemonWithHarness(t, harness, config.CodebaseStoreLibrary)
	libraryDaemon.index(t, root)
	vectorsBefore := harness.backendVectorIDs("lms_library_codebase")
	if len(vectorsBefore) == 0 {
		t.Fatal("the codebase vector pool has no vectors")
	}
	response, err := libraryDaemon.client.ClearIndex(context.Background(), &pb.ClearIndexRequest{Path: root})
	if err != nil || !response.GetCleared() {
		t.Fatalf("ClearIndex cleared=%t, error=%v", response.GetCleared(), err)
	}
	if owners := libraryDaemon.readCodebaseOwners(t); len(owners) != 0 {
		t.Fatalf("ClearIndex retained %d occurrence owners", len(owners))
	}
	vectorsAfter := harness.backendVectorIDs("lms_library_codebase")
	if !slices.Equal(vectorsBefore, vectorsAfter) {
		t.Fatalf("ClearIndex changed canonical vector IDs: before=%v after=%v", vectorsBefore, vectorsAfter)
	}
	legacyAfter := readHistoricalCodeRows(t, harness, legacyCollection)
	if !maps.Equal(legacyBefore, legacyAfter) {
		t.Fatal("library indexing or ClearIndex changed historical code rows or vectors")
	}
	afterDescription, err := harness.milvus.DescribeCollection(harness.context(), milvusclient.NewDescribeCollectionOption(legacyCollection))
	if err != nil {
		t.Fatalf("describe retained historical collection: %v", err)
	}
	if legacyDescription.ID != afterDescription.ID || !reflect.DeepEqual(legacyDescription.Schema, afterDescription.Schema) || !maps.Equal(legacyDescription.Properties, afterDescription.Properties) {
		t.Fatal("library indexing or ClearIndex replaced the historical collection or changed its schema or properties")
	}
	search, err := legacyDaemon.client.SearchCode(context.Background(), &pb.SearchCodeRequest{
		Path: root, Query: "retainedmarker", Limit: codebaseSearchLimit,
	})
	if err != nil || len(search.GetResults()) != 1 {
		t.Fatalf("historical search returned %d results, error=%v", len(search.GetResults()), err)
	}
	t.Logf("ClearIndex removed every occurrence and retained %d pool vectors and %d historical rows in %s", len(vectorsAfter), len(legacyAfter), legacyCollection)
}

type historicalCodeRow struct {
	Content string
	Vector  []float32
}

func readHistoricalCodeRows(t *testing.T, harness *libraryHarness, collection string) map[string]string {
	t.Helper()
	result, err := harness.milvus.Query(harness.context(), milvusclient.NewQueryOption(collection).
		WithFilter(`id != ""`).WithOutputFields("id", "content", "vector").WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		t.Fatalf("read historical collection %s: %v", collection, err)
	}
	vectors, valid := result.GetColumn("vector").(*column.ColumnFloatVector)
	if !valid {
		t.Fatal("historical collection did not return float vectors")
	}
	rows := make(map[string]string, result.ResultCount)
	for index := range result.ResultCount {
		id, err := result.GetColumn("id").GetAsString(index)
		if err != nil {
			t.Fatalf("read historical row ID: %v", err)
		}
		content, err := result.GetColumn("content").GetAsString(index)
		if err != nil {
			t.Fatalf("read historical row content: %v", err)
		}
		encoded, err := json.Marshal(historicalCodeRow{Content: content, Vector: vectors.Data()[index]})
		if err != nil {
			t.Fatalf("encode historical row: %v", err)
		}
		rows[id] = fmt.Sprintf("%x", sha256.Sum256(encoded))
	}
	return rows
}
