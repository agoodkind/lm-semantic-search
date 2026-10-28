//go:build milvusintegration

package milvusintegration

import (
	"context"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/column"
	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/index"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
)

const (
	recoveryRows          = 200
	recoveryEmptyModels   = 20
	recoveryLostRows      = 30
	recoveryCollection    = "shared_recovery"
	recoverySourceCopy    = "shared_recovery_source"
	recoveryCopyTarget    = "shared_recovery_copy"
	recoveryPartialTarget = "shared_recovery_partial"
)

func requireCollectionExists(t *testing.T, client *milvusclient.Client, name string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	exists, err := client.HasCollection(ctx, milvusclient.NewHasCollectionOption(name))
	if err != nil {
		t.Fatalf("check %s: %v", name, err)
	}
	if !exists {
		t.Fatalf("collection %s does not exist", name)
	}
}

func dropCollection(t *testing.T, client *milvusclient.Client, name string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	if err := client.DropCollection(ctx, milvusclient.NewDropCollectionOption(name)); err != nil {
		t.Fatalf("drop %s: %v", name, err)
	}
}

// requireMmapEverywhere checks that every field except the primary key and the
// BM25 sparse vector, and every index, has mmap enabled.
func requireMmapEverywhere(t *testing.T, client *milvusclient.Client, name string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	described, err := client.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(name))
	if err != nil {
		t.Fatalf("describe %s: %v", name, err)
	}
	for _, field := range described.Schema.Fields {
		if field.PrimaryKey || field.Name == milvusstore.SparseVectorField {
			continue
		}
		if field.TypeParams["mmap.enabled"] != "true" {
			t.Fatalf("field %s of %s has mmap.enabled %q", field.Name, name, field.TypeParams["mmap.enabled"])
		}
	}
	indexNames, err := client.ListIndexes(ctx, milvusclient.NewListIndexOption(name))
	if err != nil {
		t.Fatalf("list indexes of %s: %v", name, err)
	}
	for _, indexName := range indexNames {
		description, describeErr := client.DescribeIndex(ctx, milvusclient.NewDescribeIndexOption(name, indexName))
		if describeErr != nil {
			t.Fatalf("describe index %s of %s: %v", indexName, name, describeErr)
		}
		if description.Params()["mmap.enabled"] != "true" {
			t.Fatalf("index %s of %s has mmap.enabled %q", indexName, name, description.Params()["mmap.enabled"])
		}
	}
}

func requireLoaded(t *testing.T, client *milvusclient.Client, name string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	state, err := client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(name))
	if err != nil {
		t.Fatalf("load state of %s: %v", name, err)
	}
	if state.State != entity.LoadStateLoaded {
		t.Fatalf("collection %s has load state %v", name, state.State)
	}
}

func countMatching(t *testing.T, client *milvusclient.Client, collectionName string, expression string) int64 {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	result, err := client.Query(ctx, milvusclient.NewQueryOption(collectionName).
		WithFilter(expression).
		WithOutputFields("count(*)").
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		t.Fatalf("count %s in %s: %v", expression, collectionName, err)
	}
	count, err := result.GetColumn("count(*)").GetAsInt64(0)
	if err != nil {
		t.Fatalf("read count of %s: %v", collectionName, err)
	}
	return count
}

// TestSharedStoreRecoversPartialStateAndKeepsModelNullState checks recovery
// and copy contracts of the shared vector store on a throwaway Milvus.
// EnsureCollection creates the missing collection when only one of the pair
// exists, and creates both with mmap on every field and index. EnsureMmap
// enables mmap on a loaded collection and leaves it loaded.
// RemoveUnreferencedVectors deletes the vectors that lost occurrence writes
// leave behind. A copy keeps a non-null empty embedding model, VerifyCopy
// reports a changed model null state, and CopyFrom rejects a declaration that
// omits a source column.
func TestSharedStoreRecoversPartialStateAndKeepsModelNullState(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, "2g")
	client := dialMilvus(t, stack)
	ctx := context.Background()
	options := milvusstore.Options{Hybrid: true, EmbeddingModel: "generated-model", DenseIndex: index.NewFlatIndex(entity.COSINE)}
	baseline := milvusstore.New(client, options)
	shared, err := milvusstore.NewShared(client, options)
	if err != nil {
		t.Fatalf("new shared store: %v", err)
	}
	ensure := func(name string) {
		t.Helper()
		if err := shared.EnsureCollection(ctx, collection.EnsureRequest{Collection: name, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
			t.Fatalf("ensure %s: %v", name, err)
		}
	}

	partialVectors := milvusstore.VectorCollectionName(recoveryPartialTarget)
	ensure(recoveryPartialTarget)
	dropCollection(t, client, recoveryPartialTarget)
	ensure(recoveryPartialTarget)
	requireCollectionExists(t, client, recoveryPartialTarget)
	dropCollection(t, client, partialVectors)
	ensure(recoveryPartialTarget)
	requireCollectionExists(t, client, partialVectors)
	requireMmapEverywhere(t, client, recoveryPartialTarget)
	requireMmapEverywhere(t, client, partialVectors)

	recoveryVectors := milvusstore.VectorCollectionName(recoveryCollection)
	ensure(recoveryCollection)
	alterCtx, cancelAlter := context.WithTimeout(ctx, milvusCallTimeout)
	if err := client.AlterCollectionFieldProperty(alterCtx, milvusclient.NewAlterCollectionFieldPropertiesOption(recoveryCollection, milvusstore.ContentField).WithProperty("mmap.enabled", "false")); err != nil {
		t.Fatalf("disable mmap on content: %v", err)
	}
	cancelAlter()
	loadSharedCollections(t, client, recoveryCollection, recoveryVectors)
	changed, err := shared.EnsureMmap(ctx, recoveryCollection)
	if err != nil {
		t.Fatalf("ensure mmap: %v", err)
	}
	if !slices.Equal(changed, []string{recoveryCollection}) {
		t.Fatalf("ensure mmap changed %v, want [%s]", changed, recoveryCollection)
	}
	requireMmapEverywhere(t, client, recoveryCollection)
	requireLoaded(t, client, recoveryCollection)
	requireLoaded(t, client, recoveryVectors)

	corpus := buildSharedCorpus()
	rows := corpus.rows[:recoveryRows]
	if err := shared.Upsert(ctx, recoveryCollection, sharedDeclaration, rows); err != nil {
		t.Fatalf("shared upsert: %v", err)
	}
	time.Sleep(sharedVisibilityWait)
	vectorsBefore := countRows(t, client, recoveryVectors)
	lostIDs := make([]string, 0, recoveryLostRows)
	for _, row := range rows[:recoveryLostRows] {
		lostIDs = append(lostIDs, row.ID)
	}
	deleteCtx, cancelDelete := context.WithTimeout(ctx, milvusCallTimeout)
	if _, err := client.Delete(deleteCtx, milvusclient.NewDeleteOption(recoveryCollection).WithStringIDs(milvusstore.IDField, slices.Clone(lostIDs))); err != nil {
		t.Fatalf("delete occurrences without their vectors: %v", err)
	}
	cancelDelete()
	time.Sleep(sharedVisibilityWait)
	remaining := rows[recoveryLostRows:]
	wantVectors := int64(len(distinctVectorKeys(remaining)))
	if vectorsBefore <= wantVectors {
		t.Fatalf("the deleted occurrences left %d vectors, want more than %d", vectorsBefore, wantVectors)
	}
	removed, err := shared.RemoveUnreferencedVectors(ctx, recoveryCollection)
	if err != nil {
		t.Fatalf("remove unreferenced vectors: %v", err)
	}
	time.Sleep(sharedVisibilityWait)
	if removed != vectorsBefore-wantVectors {
		t.Fatalf("removed %d vectors, want %d", removed, vectorsBefore-wantVectors)
	}
	if got := countRows(t, client, recoveryVectors); got != wantVectors {
		t.Fatalf("vector collection has %d rows, want %d", got, wantVectors)
	}
	requireVectorReferencesConsistent(t, client, "after unreferenced vector removal", recoveryCollection)

	if err := baseline.EnsureCollection(ctx, collection.EnsureRequest{Collection: recoverySourceCopy, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure copy source: %v", err)
	}
	loadSharedCollections(t, client, recoverySourceCopy)
	if err := baseline.Upsert(ctx, recoverySourceCopy, sharedDeclaration, rows); err != nil {
		t.Fatalf("copy source upsert: %v", err)
	}
	emptyModelIDs := make([]string, 0, recoveryEmptyModels)
	for _, row := range rows[:recoveryEmptyModels] {
		emptyModelIDs = append(emptyModelIDs, row.ID)
	}
	emptyModels, err := column.NewNullableColumnVarChar(milvusstore.EmbeddingModelField, make([]string, recoveryEmptyModels), slices.Repeat([]bool{true}, recoveryEmptyModels), column.WithSparseNullableMode[string](true))
	if err != nil {
		t.Fatalf("build empty model column: %v", err)
	}
	partialUpdateRows(t, client, recoverySourceCopy, emptyModelIDs, emptyModels)
	time.Sleep(sharedVisibilityWait)
	if got := countMatching(t, client, recoverySourceCopy, milvusstore.EmbeddingModelField+` == ""`); got != recoveryEmptyModels {
		t.Fatalf("source stores %d non-null empty models, want %d", got, recoveryEmptyModels)
	}

	narrow := collection.Declaration{ItemIDColumn: sharedDeclaration.ItemIDColumn, Scalars: sharedDeclaration.Scalars[:len(sharedDeclaration.Scalars)-1]}
	if _, err := shared.CopyFrom(ctx, recoverySourceCopy, recoveryCopyTarget, narrow, nil); err == nil || !strings.Contains(err.Error(), "does not declare") {
		t.Fatalf("copy with a declaration that omits a source column returned %v, want a rejection", err)
	}

	ivfOptions := options
	ivfOptions.DenseIndex = index.NewIvfFlatIndex(entity.COSINE, 4)
	ivfShared, err := milvusstore.NewShared(client, ivfOptions)
	if err != nil {
		t.Fatalf("new IVF shared store: %v", err)
	}
	if err := ivfShared.EnsureCollection(ctx, collection.EnsureRequest{Collection: recoveryCopyTarget, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure IVF copy target: %v", err)
	}
	copyVectors := milvusstore.VectorCollectionName(recoveryCopyTarget)
	loadSharedCollections(t, client, recoveryCopyTarget, copyVectors)
	replaced, err := shared.MatchDenseIndex(ctx, recoverySourceCopy, recoveryCopyTarget)
	if err != nil || !replaced {
		t.Fatalf("match dense index returned %t, %v, want a replacement", replaced, err)
	}
	describeCtx, cancelDescribe := context.WithTimeout(ctx, milvusCallTimeout)
	description, err := client.DescribeIndex(describeCtx, milvusclient.NewDescribeIndexOption(copyVectors, milvusstore.DenseVectorField))
	cancelDescribe()
	if err != nil {
		t.Fatalf("describe replaced index: %v", err)
	}
	if description.Params()["index_type"] != "FLAT" {
		t.Fatalf("replaced index type %q, want FLAT", description.Params()["index_type"])
	}
	requireMmapEverywhere(t, client, copyVectors)
	requireLoaded(t, client, copyVectors)
	replaced, err = shared.MatchDenseIndex(ctx, recoverySourceCopy, recoveryCopyTarget)
	if err != nil || replaced {
		t.Fatalf("second match dense index returned %t, %v, want no replacement", replaced, err)
	}
	dropCtx, cancelDrop := context.WithTimeout(ctx, milvusCallTimeout)
	if err := client.ReleaseCollection(dropCtx, milvusclient.NewReleaseCollectionOption(copyVectors)); err != nil {
		t.Fatalf("release %s: %v", copyVectors, err)
	}
	if err := client.DropIndex(dropCtx, milvusclient.NewDropIndexOption(copyVectors, milvusstore.DenseVectorField)); err != nil {
		t.Fatalf("drop dense index of %s: %v", copyVectors, err)
	}
	cancelDrop()
	replaced, err = shared.MatchDenseIndex(ctx, recoverySourceCopy, recoveryCopyTarget)
	if err != nil || !replaced {
		t.Fatalf("match dense index without a target index returned %t, %v, want a new index", replaced, err)
	}
	requireMmapEverywhere(t, client, copyVectors)
	loadSharedCollections(t, client, copyVectors)

	resumeAfter := rows[recoveryRows/2].ID
	resumed, err := shared.CopyFromAfter(ctx, recoverySourceCopy, recoveryCopyTarget, sharedDeclaration, resumeAfter, nil)
	if err != nil {
		t.Fatalf("copy after %s: %v", resumeAfter, err)
	}
	wantResumed := int64(0)
	for _, row := range rows {
		if row.ID > resumeAfter {
			wantResumed++
		}
	}
	if resumed != wantResumed {
		t.Fatalf("copy after %s wrote %d rows, want %d", resumeAfter, resumed, wantResumed)
	}
	time.Sleep(sharedVisibilityWait)
	maxID, err := shared.MaxOccurrenceID(ctx, recoveryCopyTarget)
	if err != nil {
		t.Fatalf("max occurrence id: %v", err)
	}
	if maxID != rows[len(rows)-1].ID {
		t.Fatalf("max occurrence id %q, want %q", maxID, rows[len(rows)-1].ID)
	}
	if _, err := shared.CopyFromAfter(ctx, recoverySourceCopy, recoveryCopyTarget, sharedDeclaration, "", nil); err != nil {
		t.Fatalf("copy: %v", err)
	}
	time.Sleep(sharedVisibilityWait)
	if got := countMatching(t, client, recoveryCopyTarget, milvusstore.EmbeddingModelField+` == ""`); got != recoveryEmptyModels {
		t.Fatalf("copy stores %d non-null empty models, want %d", got, recoveryEmptyModels)
	}
	report, err := shared.VerifyCopy(ctx, recoverySourceCopy, recoveryCopyTarget, sharedDeclaration)
	if err != nil {
		t.Fatalf("verify copy: %v", err)
	}
	requireCopyReport(t, "copy with empty models", report, int64(len(rows)), int64(len(distinctVectorKeys(rows))))

	nullModel, err := column.NewNullableColumnVarChar(milvusstore.EmbeddingModelField, []string{""}, []bool{false}, column.WithSparseNullableMode[string](true))
	if err != nil {
		t.Fatalf("build null model column: %v", err)
	}
	partialUpdateRows(t, client, recoveryCopyTarget, emptyModelIDs[:1], nullModel)
	time.Sleep(sharedVisibilityWait)
	drift, err := shared.VerifyCopy(ctx, recoverySourceCopy, recoveryCopyTarget, sharedDeclaration)
	if err != nil {
		t.Fatalf("verify drift: %v", err)
	}
	if drift.MismatchedCount != 1 || !slices.Equal(drift.Mismatched, emptyModelIDs[:1]) {
		t.Fatalf("drift report %+v, want the single mismatched row %s", drift, emptyModelIDs[0])
	}
}
