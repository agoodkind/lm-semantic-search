//go:build milvusintegration

package milvusintegration

import (
	"context"
	"slices"
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
	copyFreshBaseline = "shared_fresh_baseline"
	copyFreshShared   = "shared_fresh_layout"
	copyNullHashRows  = 200
	copyOtherModel    = "older-model"
	copyOtherModelRow = 100
)

// partialUpdateRows writes the given columns of the rows with ids through a
// Milvus partial update and keeps every other stored column.
func partialUpdateRows(t *testing.T, client *milvusclient.Client, collectionName string, ids []string, columns ...column.Column) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	option := milvusclient.NewColumnBasedInsertOption(collectionName).
		WithVarcharColumn(milvusstore.IDField, slices.Clone(ids)).
		WithColumns(columns...).
		WithPartialUpdate(true)
	if _, err := client.Upsert(ctx, option); err != nil {
		t.Fatalf("partial update of %s: %v", collectionName, err)
	}
}

func requireCopyReport(t *testing.T, stage string, report milvusstore.CopyReport, wantRows int64, wantVectors int64) {
	t.Helper()
	if !report.Equal() || report.SourceRows != wantRows || report.TargetVectors != wantVectors {
		t.Fatalf("%s: copy report %+v, want %d equal rows and %d vectors", stage, report, wantRows, wantVectors)
	}
	t.Logf("%s: verified %d rows and %d vectors", stage, report.SourceRows, report.TargetVectors)
}

// TestSharedStoreCopiesSingleCollectionExactly writes generated rows through
// the single-collection store, gives some rows a null content hash and another
// embedding model, copies the collection into a shared vector pair, and checks
// that verification finds every row equal. It then deletes and rewrites source
// rows, checks that verification reports exactly those rows, repairs them, and
// checks that the copy searches like the source.
func TestSharedStoreCopiesSingleCollectionExactly(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStack(t, "4g")
	client := dialMilvus(t, stack)
	ctx := context.Background()
	options := milvusstore.Options{Hybrid: true, EmbeddingModel: "generated-model", DenseIndex: index.NewFlatIndex(entity.COSINE)}
	baseline := milvusstore.New(client, options)
	shared, err := milvusstore.NewShared(client, options)
	if err != nil {
		t.Fatalf("new shared store: %v", err)
	}
	corpus := buildSharedCorpus()
	if err := baseline.EnsureCollection(ctx, collection.EnsureRequest{Collection: sharedBaselineCollection, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure source: %v", err)
	}
	if err := shared.EnsureCollection(ctx, collection.EnsureRequest{Collection: sharedLayoutCollection, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure target: %v", err)
	}
	vectorCollection := milvusstore.VectorCollectionName(sharedLayoutCollection)
	loadSharedCollections(t, client, sharedBaselineCollection, sharedLayoutCollection, vectorCollection)
	for start := 0; start < len(corpus.rows); start += 200 {
		if err := baseline.Upsert(ctx, sharedBaselineCollection, sharedDeclaration, corpus.rows[start:min(start+200, len(corpus.rows))]); err != nil {
			t.Fatalf("source upsert: %v", err)
		}
	}
	nullHashIDs := make([]string, 0, copyNullHashRows)
	for _, row := range corpus.rows[:copyNullHashRows] {
		nullHashIDs = append(nullHashIDs, row.ID)
	}
	nullHashes, err := column.NewNullableColumnVarChar(milvusstore.ContentHashField, make([]string, copyNullHashRows), make([]bool, copyNullHashRows), column.WithSparseNullableMode[string](true))
	if err != nil {
		t.Fatalf("build null content hash column: %v", err)
	}
	partialUpdateRows(t, client, sharedBaselineCollection, nullHashIDs, nullHashes)
	otherModelIDs := make([]string, 0, copyOtherModelRow)
	otherModels := make([]string, 0, copyOtherModelRow)
	for _, row := range corpus.rows[copyNullHashRows : copyNullHashRows+copyOtherModelRow] {
		otherModelIDs = append(otherModelIDs, row.ID)
		otherModels = append(otherModels, copyOtherModel)
	}
	modelColumn, err := milvusstore.NewRowEmbeddingModelColumn(sharedBaselineCollection, otherModels)
	if err != nil {
		t.Fatalf("build embedding model column: %v", err)
	}
	partialUpdateRows(t, client, sharedBaselineCollection, otherModelIDs, modelColumn)
	time.Sleep(sharedVisibilityWait)

	copied, err := shared.CopyFrom(ctx, sharedBaselineCollection, sharedLayoutCollection, sharedDeclaration, nil)
	if err != nil {
		t.Fatalf("copy: %v", err)
	}
	if copied != int64(len(corpus.rows)) {
		t.Fatalf("copy wrote %d rows, want %d", copied, len(corpus.rows))
	}
	time.Sleep(sharedVisibilityWait)
	report, err := shared.VerifyCopy(ctx, sharedBaselineCollection, sharedLayoutCollection, sharedDeclaration)
	if err != nil {
		t.Fatalf("verify copy: %v", err)
	}
	requireCopyReport(t, "initial copy", report, int64(len(corpus.rows)), int64(len(distinctVectorKeys(corpus.rows))))
	requireVectorReferencesConsistent(t, client, "initial copy", sharedLayoutCollection)

	deletedIDs := []string{corpus.rows[400].ID, corpus.rows[401].ID, corpus.rows[402].ID, corpus.rows[403].ID, corpus.rows[404].ID}
	quoted := ""
	for index, id := range deletedIDs {
		if index > 0 {
			quoted += ", "
		}
		quoted += `"` + id + `"`
	}
	if _, err := client.Delete(ctx, milvusclient.NewDeleteOption(sharedBaselineCollection).WithExpr(milvusstore.IDField+" in ["+quoted+"]")); err != nil {
		t.Fatalf("delete source rows: %v", err)
	}
	rewritten := make([]collection.Row, 0, 3)
	for _, row := range corpus.rows[500:503] {
		changed := row
		changed.Vector = slices.Clone(corpus.vectors[0])
		rewritten = append(rewritten, changed)
	}
	if err := baseline.Upsert(ctx, sharedBaselineCollection, sharedDeclaration, rewritten); err != nil {
		t.Fatalf("rewrite source rows: %v", err)
	}
	time.Sleep(sharedVisibilityWait)
	drift, err := shared.VerifyCopy(ctx, sharedBaselineCollection, sharedLayoutCollection, sharedDeclaration)
	if err != nil {
		t.Fatalf("verify drift: %v", err)
	}
	if drift.ExtraCount != int64(len(deletedIDs)) || drift.MismatchedCount != int64(len(rewritten)) || drift.MissingCount != 0 {
		t.Fatalf("drift report %+v, want %d extra and %d mismatched rows", drift, len(deletedIDs), len(rewritten))
	}
	if err := shared.CopyIDs(ctx, sharedBaselineCollection, sharedLayoutCollection, sharedDeclaration, drift.Mismatched); err != nil {
		t.Fatalf("repair mismatched rows: %v", err)
	}
	if _, err := shared.RemoveOccurrences(ctx, sharedLayoutCollection, drift.Extra); err != nil {
		t.Fatalf("remove extra rows: %v", err)
	}
	time.Sleep(sharedVisibilityWait)
	current := make([]collection.Row, 0, len(corpus.rows))
	for _, row := range corpus.rows {
		if slices.Contains(deletedIDs, row.ID) {
			continue
		}
		current = append(current, row)
	}
	for index := range current {
		for _, changed := range rewritten {
			if current[index].ID == changed.ID {
				current[index] = changed
			}
		}
	}
	repaired, err := shared.VerifyCopy(ctx, sharedBaselineCollection, sharedLayoutCollection, sharedDeclaration)
	if err != nil {
		t.Fatalf("verify repair: %v", err)
	}
	requireCopyReport(t, "after repair", repaired, int64(len(current)), int64(len(distinctVectorKeys(current))))
	requireVectorReferencesConsistent(t, client, "after repair", sharedLayoutCollection)

	// Milvus BM25 scores depend on every row version a collection received.
	// The source received partial updates that the copy did not. The search
	// comparison therefore writes the current rows once into a fresh
	// single-collection store and copies them once into a fresh shared pair.
	fresh := searchTargets{baseline: copyFreshBaseline, shared: copyFreshShared}
	if err := baseline.EnsureCollection(ctx, collection.EnsureRequest{Collection: fresh.baseline, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure fresh baseline: %v", err)
	}
	if err := shared.EnsureCollection(ctx, collection.EnsureRequest{Collection: fresh.shared, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure fresh shared: %v", err)
	}
	loadSharedCollections(t, client, fresh.baseline, fresh.shared, milvusstore.VectorCollectionName(fresh.shared))
	for start := 0; start < len(current); start += 200 {
		if err := baseline.Upsert(ctx, fresh.baseline, sharedDeclaration, current[start:min(start+200, len(current))]); err != nil {
			t.Fatalf("fresh baseline upsert: %v", err)
		}
	}
	if _, err := shared.CopyFrom(ctx, sharedBaselineCollection, fresh.shared, sharedDeclaration, nil); err != nil {
		t.Fatalf("fresh copy: %v", err)
	}
	time.Sleep(sharedVisibilityWait)
	freshReport, err := shared.VerifyCopy(ctx, sharedBaselineCollection, fresh.shared, sharedDeclaration)
	if err != nil {
		t.Fatalf("verify fresh copy: %v", err)
	}
	requireCopyReport(t, "fresh copy", freshReport, int64(len(current)), int64(len(distinctVectorKeys(current))))
	compareSearches(t, client, "fresh copy", fresh, baseline, shared, corpus, vectorKeysByID(current))
}
