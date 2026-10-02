//go:build live

package live

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/observation"
)

func TestLibraryPagingDuringLargeScalarPublication(t *testing.T) {
	harness := newLibraryHarness(t)
	store := newSearchLiveStore(t, harness.context(), harness, "publication_paging")
	collector := &operationEvents{}
	store.library = store.open(t, func(config *library.Config) { config.Observer = collector })
	const namespace = "publication"
	const rowCount = 41458
	const scalarCount = 15
	const input = "Large owner publication preserves responsive ordered conversation paging."
	spec := library.NamespaceSpec{ID: namespace, Policy: library.ReplaceAllowed}
	for index := range scalarCount {
		spec.Scalars = append(spec.Scalars, library.ScalarColumn{Name: fmt.Sprintf("scalar_%02d", index), Type: library.Int64})
	}
	if err := store.library.RegisterNamespace(store.ctx, spec); err != nil {
		t.Fatal(err)
	}
	rows := make([]library.Occurrence, rowCount)
	for index := range rows {
		key := fmt.Sprintf("row-%05d", index)
		values := make(map[string]library.ScalarValue, scalarCount)
		for column := range scalarCount {
			values[fmt.Sprintf("scalar_%02d", column)] = library.ScalarValue{Type: library.Int64, Int64: int64(index + column)}
		}
		rows[index] = library.Occurrence{RowKey: key, SortKey: key, SourceText: input + " " + key,
			SearchText: input + " " + key, EmbeddingInput: input, Scalars: values}
	}
	if _, err := store.library.Apply(store.ctx, library.Batch{Namespace: namespace, OwnerID: "seed", GenerationOrder: 1,
		IdempotencyToken: "seed", Mode: library.Replace, Rows: rows[:4]}); err != nil {
		t.Fatal(err)
	}
	peer := store.open(t, func(config *library.Config) { config.QueryTimeout = 0 })
	request := library.SearchRequest{Namespace: namespace, Query: input, PageSize: 1}
	first, err := peer.Search(store.ctx, request)
	if err != nil || !first.HasMore {
		t.Fatalf("seed search: %+v, %v", first, err)
	}
	request.Cursor = first.NextCursor
	key := library.GenerationKey{Namespace: namespace, OwnerID: "large", GenerationOrder: 1, IdempotencyToken: "large"}
	for start := 0; start < len(rows); start += searchLiveStageRows {
		if err := store.library.Stage(store.ctx, library.StageBatch{Key: key, Mode: library.Replace,
			Rows: rows[start:min(start+searchLiveStageRows, len(rows))]}); err != nil {
			t.Fatal(err)
		}
	}
	seal, err := library.SealRows(rows)
	if err != nil {
		t.Fatal(err)
	}
	type commitResult struct {
		receipt library.ApplyReceipt
		err     error
	}
	committed := make(chan commitResult, 1)
	publicationContext, cancelPublication := context.WithCancel(store.ctx)
	joined := make(chan struct{})
	t.Cleanup(func() {
		cancelPublication()
		select {
		case <-joined:
		case <-time.After(time.Minute):
			t.Error("publication did not stop within its cleanup deadline")
			return
		}
		for _, event := range collector.completed(t, "large-publication") {
			if event.Operation == observation.CatalogTransaction {
				t.Logf("publication catalog write: %s, outcome %s, boundary %s", event.Duration, event.Outcome, event.Data.Transaction.Boundary)
			}
		}
	})
	started := time.Now()
	go func() {
		defer close(joined)
		receipt, commitErr := store.library.CommitGeneration(observedRunContext(publicationContext, "large-publication"), key, seal)
		committed <- commitResult{receipt: receipt, err: commitErr}
	}()
	var result commitResult
	var maximumPage time.Duration
	pages := 0
	for {
		select {
		case result = <-committed:
			goto published
		default:
		}
		pageStarted := time.Now()
		page, err := peer.Search(store.ctx, request)
		maximumPage = max(maximumPage, time.Since(pageStarted))
		if err != nil || len(page.Hits) != 1 || page.Hits[0].ID.OwnerID != "seed" {
			t.Fatalf("continuation during publication after %s: %+v, %v", time.Since(pageStarted), page, err)
		}
		pages++
	}

published:
	if result.err != nil {
		t.Fatal(result.err)
	}
	if pages == 0 {
		t.Fatal("publication completed without a concurrent continuation")
	}
	t.Logf("published %d rows with %d scalars in %s; %d concurrent continuations, maximum %s",
		rowCount, scalarCount, time.Since(started), pages, maximumPage)
	if err := store.library.Stage(store.ctx, library.StageBatch{Key: key, Mode: library.Replace, Rows: rows[:searchLiveStageRows]}); err != nil {
		t.Fatalf("unchanged staged replay: %v", err)
	}
	replayed, err := store.library.CommitGeneration(store.ctx, key, seal)
	if err != nil || replayed != result.receipt {
		t.Fatalf("unchanged receipt: %+v, %v; want %+v", replayed, err, result.receipt)
	}
	stats, err := store.library.NamespaceStats(store.ctx, namespace)
	if err != nil || stats.Occurrences != rowCount+4 || stats.Owners != 2 {
		t.Fatalf("published counts: %+v, %v", stats, err)
	}
	conflict := rows[0]
	conflict.SourceText = "changed source"
	_, err = store.library.Apply(store.ctx, library.Batch{Namespace: namespace, OwnerID: "large", GenerationOrder: 2,
		IdempotencyToken: "conflict", Mode: library.Append, Rows: []library.Occurrence{conflict}})
	if !errors.Is(err, library.ErrAppendConflict) {
		t.Fatalf("append rewrite: %v; want ErrAppendConflict", err)
	}
	unlimited := store.open(t, func(config *library.Config) { config.MaxPageSize = 0 })
	request.Cursor = ""
	request.PageSize = rowCount + 4
	complete, err := unlimited.Search(store.ctx, request)
	if err != nil || len(complete.Hits) != rowCount+4 || complete.HasMore {
		t.Fatalf("complete published search: %d hits, HasMore %v, %v", len(complete.Hits), complete.HasMore, err)
	}
	byKey := make(map[string]library.Occurrence, len(rows))
	for _, row := range rows {
		byKey[row.RowKey] = row
	}
	seen := make(map[library.OccurrenceID]bool, len(complete.Hits))
	for _, hit := range complete.Hits {
		row, found := byKey[hit.ID.RowKey]
		if !found || seen[hit.ID] || hit.SourceText != row.SourceText || !reflect.DeepEqual(hit.Scalars, row.Scalars) {
			t.Fatalf("published hit differs from its source occurrence: %+v", hit)
		}
		seen[hit.ID] = true
	}
	request.PageSize = 100
	var got []library.SearchHit
	for {
		page, err := store.library.Search(store.ctx, request)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, page.Hits...)
		if !page.HasMore {
			break
		}
		request.Cursor = page.NextCursor
	}
	if !reflect.DeepEqual(got, complete.Hits) {
		t.Fatalf("paged search differs from complete search: %d and %d hits", len(got), len(complete.Hits))
	}
}
