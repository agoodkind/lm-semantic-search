//go:build milvusintegration

package milvusintegration

import (
	"context"
	"fmt"
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

// progressRecorder stores the last report of each phase and the first report
// that breaks a rule within a phase: Done decreases, Total changes, or Done
// exceeds Total.
type progressRecorder struct {
	last      map[milvusstore.CopyPhase]milvusstore.CopyProgress
	violation string
}

func newProgressRecorder() *progressRecorder {
	return &progressRecorder{last: map[milvusstore.CopyPhase]milvusstore.CopyProgress{}, violation: ""}
}

func (recorder *progressRecorder) record(progress milvusstore.CopyProgress) {
	previous, seen := recorder.last[progress.Phase]
	if recorder.violation == "" {
		switch {
		case progress.Done > progress.Total:
			recorder.violation = fmt.Sprintf("%s reported Done %d exceeding Total %d", progress.Phase, progress.Done, progress.Total)
		case seen && progress.Done < previous.Done:
			recorder.violation = fmt.Sprintf("%s Done decreased from %d to %d", progress.Phase, previous.Done, progress.Done)
		case seen && progress.Total != previous.Total:
			recorder.violation = fmt.Sprintf("%s Total changed from %d to %d", progress.Phase, previous.Total, progress.Total)
		}
	}
	recorder.last[progress.Phase] = progress
}

// requireProgress fails if a report breaks a rule above, the reported phases differ
// from want, or a phase's last report has Done or Total unequal to its expected
// total. A progress display computes its percentage from Total and the last Done.
// The expected total comes from want.
func requireProgress(t *testing.T, stage string, recorder *progressRecorder, want map[milvusstore.CopyPhase]int64) {
	t.Helper()
	if recorder.violation != "" {
		t.Fatalf("%s: %s", stage, recorder.violation)
	}
	if len(recorder.last) != len(want) {
		t.Fatalf("%s: %d phases reported progress, want %d", stage, len(recorder.last), len(want))
	}
	for phase, total := range want {
		last, seen := recorder.last[phase]
		if !seen {
			t.Fatalf("%s: phase %s did not report progress", stage, phase)
		}
		if last.Total != total || last.Done != total {
			t.Fatalf("%s: phase %s last reported Done %d and Total %d; expected Done %d and Total %d", stage, phase, last.Done, last.Total, total, total)
		}
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

	distinctVectors := int64(len(distinctVectorKeys(corpus.rows)))
	rowCount := int64(len(corpus.rows))
	copyProgress := newProgressRecorder()
	copied, err := shared.CopyFrom(ctx, sharedBaselineCollection, sharedLayoutCollection, sharedDeclaration, copyProgress.record)
	if err != nil {
		t.Fatalf("copy: %v", err)
	}
	if copied.Occurrences != int64(len(corpus.rows)) || copied.VectorsWritten != distinctVectors {
		t.Fatalf("copy wrote %d occurrences and %d vectors, want %d and %d", copied.Occurrences, copied.VectorsWritten, len(corpus.rows), distinctVectors)
	}
	time.Sleep(sharedVisibilityWait)
	requireProgress(t, "initial copy", copyProgress,
		map[milvusstore.CopyPhase]int64{milvusstore.CopyPhaseOccurrences: rowCount, milvusstore.CopyPhaseVectors: rowCount})
	verifyProgress := newProgressRecorder()
	report, err := shared.VerifyCopy(ctx, sharedBaselineCollection, sharedLayoutCollection, sharedDeclaration, verifyProgress.record)
	if err != nil {
		t.Fatalf("verify copy: %v", err)
	}
	requireCopyReport(t, "initial copy", report, rowCount, distinctVectors)
	requireProgress(t, "initial verify", verifyProgress,
		map[milvusstore.CopyPhase]int64{
			milvusstore.VerifyPhaseRows:       rowCount,
			milvusstore.VerifyPhaseExtraRows:  rowCount,
			milvusstore.VerifyPhaseReferences: rowCount,
			milvusstore.VerifyPhaseVectors:    distinctVectors,
		})
	requireVectorReferencesConsistent(t, client, "initial copy", sharedLayoutCollection)
	repeated, err := shared.CopyFrom(ctx, sharedBaselineCollection, sharedLayoutCollection, sharedDeclaration, nil)
	if err != nil {
		t.Fatalf("repeated copy: %v", err)
	}
	if repeated.VectorsWritten != 0 {
		t.Fatalf("repeated copy wrote %d vectors, want 0", repeated.VectorsWritten)
	}
	t.Logf("copy wrote %d occurrences and %d vectors once; a repeated copy wrote 0 vectors", copied.Occurrences, copied.VectorsWritten)

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
	drift, err := shared.VerifyCopy(ctx, sharedBaselineCollection, sharedLayoutCollection, sharedDeclaration, nil)
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
	repaired, err := shared.VerifyCopy(ctx, sharedBaselineCollection, sharedLayoutCollection, sharedDeclaration, nil)
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
	freshReport, err := shared.VerifyCopy(ctx, sharedBaselineCollection, fresh.shared, sharedDeclaration, nil)
	if err != nil {
		t.Fatalf("verify fresh copy: %v", err)
	}
	requireCopyReport(t, "fresh copy", freshReport, int64(len(current)), int64(len(distinctVectorKeys(current))))
	compareSearches(t, client, "fresh copy", fresh, baseline, shared, corpus, vectorKeysByID(current))
}
