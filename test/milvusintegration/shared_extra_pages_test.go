//go:build milvusintegration

package milvusintegration

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"slices"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/index"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
)

const (
	extraPagesSource = "extra_pages_source"
	extraPagesTarget = "extra_pages_target"
	// extraPagesRows fills three pages of 16,384 target IDs.
	extraPagesRows        = 33_000
	extraPagesUpsertBatch = 1_000
	extraPagesIDPage      = 16_384
	extraPagesStride      = 997
)

// extraPagesDeleted returns sorted indexes for the first and last source rows,
// the rows before and at each 16,384 ID page boundary, and every 997th row.
func extraPagesDeleted() []int {
	picked := map[int]struct{}{0: {}, extraPagesRows - 1: {}}
	for boundary := extraPagesIDPage; boundary < extraPagesRows; boundary += extraPagesIDPage {
		picked[boundary-1] = struct{}{}
		picked[boundary] = struct{}{}
	}
	for index := extraPagesStride; index < extraPagesRows; index += extraPagesStride {
		picked[index] = struct{}{}
	}
	indexes := slices.Collect(maps.Keys(picked))
	slices.Sort(indexes)
	return indexes
}

// occurrenceIDs reads every occurrence ID from a shared target with Strong
// consistency and returns the IDs sorted.
func occurrenceIDs(t *testing.T, client *milvusclient.Client, target string) []string {
	t.Helper()
	ctx := context.Background()
	iterator, err := client.QueryIterator(ctx, milvusclient.NewQueryIteratorOption(target).
		WithBatchSize(extraPagesIDPage).
		WithFilter(milvusstore.IDField+` != ""`).
		WithOutputFields(milvusstore.IDField).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		t.Fatalf("open ID iterator on %s: %v", target, err)
	}
	ids := make([]string, 0, extraPagesRows)
	for {
		page, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			t.Fatalf("read IDs of %s: %v", target, nextErr)
		}
		idColumn := page.GetColumn(milvusstore.IDField)
		for row := range page.ResultCount {
			raw, getErr := idColumn.Get(row)
			if getErr != nil {
				t.Fatalf("read ID %d of %s: %v", row, target, getErr)
			}
			id, ok := raw.(string)
			if !ok {
				t.Fatalf("ID %v of %s is %T", raw, target, raw)
			}
			ids = append(ids, id)
		}
	}
	slices.Sort(ids)
	return ids
}

// TestSharedStoreExtraRowPassesSpanIDPages copies 33,000 source rows to a shared
// target, spanning three pages of 16,384 IDs, then deletes extraPagesDeleted rows
// from the source. VerifyCopy reports 33,000 target rows and every deleted row as
// extra, listing only deleted IDs. One ReconcileCopy reads 33,000 target rows and
// removes exactly the deleted rows. The target then stores exactly the remaining
// source IDs, and VerifyCopy finds every row and vector equal.
func TestSharedStoreExtraRowPassesSpanIDPages(t *testing.T) {
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
	if err := baseline.EnsureCollection(ctx, collection.EnsureRequest{Collection: extraPagesSource, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure source: %v", err)
	}
	if err := shared.EnsureCollection(ctx, collection.EnsureRequest{Collection: extraPagesTarget, Declaration: sharedDeclaration, Dimension: sharedDimension}); err != nil {
		t.Fatalf("ensure target: %v", err)
	}
	loadSharedCollections(t, client, extraPagesSource, extraPagesTarget, milvusstore.VectorCollectionName(extraPagesTarget))

	corpus := buildSharedCorpus()
	rows := make([]collection.Row, 0, extraPagesRows)
	for index := range extraPagesRows {
		base := corpus.rows[index%len(corpus.rows)]
		row := base
		row.ID = fmt.Sprintf("page-%05d", index)
		row.RelativePath = fmt.Sprintf("pages/%05d", index)
		row.Vector = slices.Clone(base.Vector)
		row.Scalars = maps.Clone(base.Scalars)
		rows = append(rows, row)
	}
	for start := 0; start < len(rows); start += extraPagesUpsertBatch {
		if err := baseline.Upsert(ctx, extraPagesSource, sharedDeclaration, rows[start:min(start+extraPagesUpsertBatch, len(rows))]); err != nil {
			t.Fatalf("source upsert at %d: %v", start, err)
		}
	}
	time.Sleep(sharedVisibilityWait)
	if _, err := shared.CopyFrom(ctx, extraPagesSource, extraPagesTarget, sharedDeclaration, nil); err != nil {
		t.Fatalf("copy: %v", err)
	}

	deletedIndexes := extraPagesDeleted()
	deleted := make([]collection.Row, 0, len(deletedIndexes))
	deletedIDs := map[string]struct{}{}
	for _, index := range deletedIndexes {
		deleted = append(deleted, rows[index])
		deletedIDs[rows[index].ID] = struct{}{}
	}
	if _, err := client.Delete(ctx, milvusclient.NewDeleteOption(extraPagesSource).WithExpr(quotedIDs(deleted))); err != nil {
		t.Fatalf("delete source rows: %v", err)
	}
	time.Sleep(sharedVisibilityWait)
	remaining := make([]collection.Row, 0, len(rows)-len(deleted))
	remainingIDs := make([]string, 0, len(rows)-len(deleted))
	for _, row := range rows {
		if _, removed := deletedIDs[row.ID]; removed {
			continue
		}
		remaining = append(remaining, row)
		remainingIDs = append(remainingIDs, row.ID)
	}
	slices.Sort(remainingIDs)

	drift, err := shared.VerifyCopy(ctx, extraPagesSource, extraPagesTarget, sharedDeclaration, nil)
	if err != nil {
		t.Fatalf("verify drift: %v", err)
	}
	if drift.TargetRows != extraPagesRows || drift.ExtraCount != int64(len(deleted)) {
		t.Fatalf("drift report counts %d target rows and %d extra, want %d and %d", drift.TargetRows, drift.ExtraCount, extraPagesRows, len(deleted))
	}
	for _, id := range drift.Extra {
		if _, removed := deletedIDs[id]; !removed {
			t.Fatalf("drift report lists extra ID %s, which the source still stores", id)
		}
	}

	reconciled, err := shared.ReconcileCopy(ctx, extraPagesSource, extraPagesTarget, sharedDeclaration, nil)
	if err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	if reconciled.TargetRows != extraPagesRows || reconciled.Extra != int64(len(deleted)) || reconciled.Missing != 0 || reconciled.Mismatched != 0 {
		t.Fatalf("reconcile read %d target rows and repaired %d extra, %d missing, %d mismatched; want %d, %d, 0, 0",
			reconciled.TargetRows, reconciled.Extra, reconciled.Missing, reconciled.Mismatched, extraPagesRows, len(deleted))
	}
	time.Sleep(sharedVisibilityWait)

	if got := occurrenceIDs(t, client, extraPagesTarget); !slices.Equal(got, remainingIDs) {
		t.Fatalf("target stores %d occurrence IDs after reconcile, want exactly the %d remaining source IDs", len(got), len(remainingIDs))
	}
	report, err := shared.VerifyCopy(ctx, extraPagesSource, extraPagesTarget, sharedDeclaration, nil)
	if err != nil {
		t.Fatalf("verify after reconcile: %v", err)
	}
	requireCopyReport(t, "after reconcile", report, int64(len(remaining)), int64(len(distinctVectorKeys(remaining))))
	requireVectorReferencesConsistent(t, client, "after reconcile", extraPagesTarget)
	t.Logf("extra-row passes read %d target rows over three ID pages and removed exactly %d extra rows", extraPagesRows, len(deleted))
}
