//go:build milvusintegration

package milvusintegration

import (
	"context"
	"maps"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/index"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
)

const (
	reconcileSource = "reconcile_source"
	reconcileTarget = "reconcile_target"
	// reportSampleLimit is the maximum number of IDs a copy report lists per class.
	reportSampleLimit = 100
	// reconcileLateRows counts rows added to the source after the first copy.
	// Their IDs sort below every row in that copy.
	reconcileLateRows      = 300
	reconcileChangedStart  = 400
	reconcileChangedRows   = 150
	reconcileVectorChanged = 50
	reconcileDeletedStart  = 600
	reconcileDeletedRows   = 120
	reconcileVectorSeed    = 91
	// reconcileTimestampShift is the number of seconds added to each changed
	// row's timestamp, as a timestamp repair rewrites source rows.
	reconcileTimestampShift = 86_400
)

// quotedIDs returns a Milvus expression that matches rows with the given IDs.
func quotedIDs(rows []collection.Row) string {
	quoted := make([]string, 0, len(rows))
	for _, row := range rows {
		quoted = append(quoted, `"`+row.ID+`"`)
	}
	return milvusstore.IDField + " in [" + strings.Join(quoted, ", ") + "]"
}

// TestSharedStoreReconcileRepairsEveryDifference copies 900 source rows into a
// shared pair. It then adds 300 source rows with IDs below the largest copied
// ID, changes metadata and timestamps on 150 rows, changes vectors on 50 of
// those rows, and deletes 120 rows. A resumed copy writes zero rows. The test
// checks that VerifyCopy counts every difference and lists at most 100 IDs per
// class. One ReconcileCopy repairs every difference and reports progress for
// each pass. VerifyCopy then finds every row equal, and a search filtered to a
// changed row's new timestamp returns that row first. A second ReconcileCopy
// repairs zero rows.
func TestSharedStoreReconcileRepairsEveryDifference(t *testing.T) {
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
	if err := baseline.EnsureCollection(ctx, collection.EnsureRequest{Collection: reconcileSource, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure source: %v", err)
	}
	if err := shared.EnsureCollection(ctx, collection.EnsureRequest{Collection: reconcileTarget, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure target: %v", err)
	}
	loadSharedCollections(t, client, reconcileSource, reconcileTarget, milvusstore.VectorCollectionName(reconcileTarget))
	upsertSource := func(rows []collection.Row) {
		t.Helper()
		for start := 0; start < len(rows); start += 200 {
			if err := baseline.Upsert(ctx, reconcileSource, sharedDeclaration, rows[start:min(start+200, len(rows))]); err != nil {
				t.Fatalf("source upsert: %v", err)
			}
		}
	}

	late := corpus.rows[:reconcileLateRows]
	upsertSource(corpus.rows[reconcileLateRows:])
	time.Sleep(sharedVisibilityWait)
	if _, err := shared.CopyFrom(ctx, reconcileSource, reconcileTarget, sharedDeclaration, nil); err != nil {
		t.Fatalf("first copy: %v", err)
	}
	upsertSource(late)
	time.Sleep(sharedVisibilityWait)
	afterID, err := shared.MaxOccurrenceID(ctx, reconcileTarget)
	if err != nil {
		t.Fatalf("largest copied ID: %v", err)
	}
	resumed, err := shared.CopyFromAfter(ctx, reconcileSource, reconcileTarget, sharedDeclaration, afterID, nil)
	if err != nil {
		t.Fatalf("resumed copy: %v", err)
	}
	if resumed.Occurrences != 0 {
		t.Fatalf("copy resumed after %s wrote %d rows, want 0 because the late rows sort below it", afterID, resumed.Occurrences)
	}

	changed := make([]collection.Row, 0, reconcileChangedRows)
	generator := rand.New(rand.NewPCG(reconcileVectorSeed, reconcileVectorSeed))
	for offset, row := range corpus.rows[reconcileChangedStart : reconcileChangedStart+reconcileChangedRows] {
		edited := row
		edited.Metadata = `{"kind":"edited"}`
		edited.Scalars = maps.Clone(row.Scalars)
		edited.Scalars["timestampUnix"] = collection.Int64Scalar(row.Scalars["timestampUnix"].Int64 + reconcileTimestampShift)
		if offset < reconcileVectorChanged {
			edited.Vector = normalizedVector(generator)
		}
		changed = append(changed, edited)
	}
	upsertSource(changed)
	deleted := corpus.rows[reconcileDeletedStart : reconcileDeletedStart+reconcileDeletedRows]
	if _, err := client.Delete(ctx, milvusclient.NewDeleteOption(reconcileSource).WithExpr(quotedIDs(deleted))); err != nil {
		t.Fatalf("delete source rows: %v", err)
	}
	time.Sleep(sharedVisibilityWait)

	drift, err := shared.VerifyCopy(ctx, reconcileSource, reconcileTarget, sharedDeclaration, nil)
	if err != nil {
		t.Fatalf("verify drift: %v", err)
	}
	if drift.MissingCount != reconcileLateRows || drift.MismatchedCount != reconcileChangedRows || drift.ExtraCount != reconcileDeletedRows {
		t.Fatalf("drift report counts %d missing, %d mismatched, %d extra; want %d, %d, %d",
			drift.MissingCount, drift.MismatchedCount, drift.ExtraCount, reconcileLateRows, reconcileChangedRows, reconcileDeletedRows)
	}
	if len(drift.Missing) != reportSampleLimit || len(drift.Mismatched) != reportSampleLimit || len(drift.Extra) != reportSampleLimit {
		t.Fatalf("drift report lists %d missing, %d mismatched, %d extra IDs; want %d of each",
			len(drift.Missing), len(drift.Mismatched), len(drift.Extra), reportSampleLimit)
	}

	current := make([]collection.Row, 0, len(corpus.rows))
	for _, row := range corpus.rows {
		if slices.ContainsFunc(deleted, func(removed collection.Row) bool { return removed.ID == row.ID }) {
			continue
		}
		if position := slices.IndexFunc(changed, func(edited collection.Row) bool { return edited.ID == row.ID }); position >= 0 {
			current = append(current, changed[position])
			continue
		}
		current = append(current, row)
	}
	sourceRows := int64(len(current))
	distinct := int64(len(distinctVectorKeys(current)))
	recorder := newProgressRecorder()
	reconciled, err := shared.ReconcileCopy(ctx, reconcileSource, reconcileTarget, sharedDeclaration, recorder.record)
	if err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	if reconciled.Missing != reconcileLateRows || reconciled.Mismatched != reconcileChangedRows || reconciled.Extra != reconcileDeletedRows {
		t.Fatalf("reconcile repaired %d missing, %d mismatched, %d extra rows; want %d, %d, %d",
			reconciled.Missing, reconciled.Mismatched, reconciled.Extra, reconcileLateRows, reconcileChangedRows, reconcileDeletedRows)
	}
	requireProgress(t, "reconcile", recorder, map[milvusstore.CopyPhase]int64{
		milvusstore.ReconcilePhaseRows:      sourceRows,
		milvusstore.ReconcilePhaseExtraRows: sourceRows + reconcileDeletedRows,
		milvusstore.ReconcilePhaseVectors:   distinct + reconciled.UnreferencedVectors,
	})
	time.Sleep(sharedVisibilityWait)

	report, err := shared.VerifyCopy(ctx, reconcileSource, reconcileTarget, sharedDeclaration, nil)
	if err != nil {
		t.Fatalf("verify after reconcile: %v", err)
	}
	requireCopyReport(t, "after reconcile", report, sourceRows, distinct)
	requireVectorReferencesConsistent(t, client, "after reconcile", reconcileTarget)
	moved := changed[0]
	movedTime := moved.Scalars["timestampUnix"].Int64
	// The range filter excludes its upper bound.
	movedEnd := movedTime + 1
	timeFilter := collection.ColumnRange("timestampUnix", &movedTime, &movedEnd)
	hits, err := shared.Search(ctx, collection.SearchRequest{
		Collection:  reconcileTarget,
		Query:       moved.Content,
		Vector:      moved.Vector,
		Limit:       3,
		Filter:      &timeFilter,
		Declaration: sharedDeclaration,
	})
	if err != nil {
		t.Fatalf("search at the new timestamp: %v", err)
	}
	if len(hits) == 0 || hits[0].RelativePath != moved.RelativePath {
		t.Fatalf("search filtered to timestamp %d returned %+v, want %s first", movedTime, hits, moved.RelativePath)
	}
	again, err := shared.ReconcileCopy(ctx, reconcileSource, reconcileTarget, sharedDeclaration, nil)
	if err != nil {
		t.Fatalf("second reconcile: %v", err)
	}
	if again.Repaired() != 0 {
		t.Fatalf("second reconcile repaired %+v, want 0 rows", again)
	}
	t.Logf("reconcile repaired %d missing, %d mismatched, %d extra rows and %d unreferenced vectors; the copy then verified equal",
		reconciled.Missing, reconciled.Mismatched, reconciled.Extra, reconciled.UnreferencedVectors)
}
