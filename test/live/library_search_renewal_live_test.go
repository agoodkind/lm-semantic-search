//go:build live

package live

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/library"
)

func TestLibrarySearchRenewsActiveSnapshots(t *testing.T) {
	harness := newLibraryHarness(t)
	store := newSearchLiveStore(t, harness.context(), harness, "renewal")
	const ttl = 6 * time.Second
	configure := func(config *library.Config) { config.SnapshotTTL = ttl }
	active := store.open(t, configure)
	peer := store.open(t, configure)
	rows := make([]library.Occurrence, 8)
	for index := range rows {
		key := fmt.Sprintf("row-%02d", index)
		rows[index] = library.Occurrence{
			RowKey: key, SortKey: key, SourceText: "snapshot source " + key,
			SearchText: "snapshot renewal", EmbeddingInput: "snapshot renewal",
		}
	}
	store.replaceOwner(t, "owner", 1, rows)
	request := library.SearchRequest{Namespace: "search", Query: "snapshot renewal", PageSize: 100}
	complete, err := active.Search(store.ctx, request)
	if err != nil || len(complete.Hits) != len(rows) || complete.HasMore || complete.NextCursor != "" {
		t.Fatalf("complete search = %+v, %v; want eight hits and completion", complete, err)
	}
	request.PageSize = 1
	idle, err := peer.Search(store.ctx, request)
	if err != nil || idle.NextCursor == "" {
		t.Fatalf("idle snapshot creation = %+v, %v", idle, err)
	}
	first, err := active.Search(store.ctx, request)
	if err != nil || first.NextCursor == "" {
		t.Fatalf("active snapshot creation = %+v, %v", first, err)
	}
	started := time.Now()
	got := append([]library.SearchHit{}, first.Hits...)
	oldCursor := first.NextCursor
	request.Cursor = oldCursor
	for index := 1; index < len(rows); index++ {
		time.Sleep(2 * time.Second)
		type result struct {
			page library.SearchPage
			err  error
		}
		results := make(chan result, 2)
		for _, opened := range []*library.Library{active, peer} {
			go func() {
				page, searchErr := opened.Search(store.ctx, request)
				results <- result{page: page, err: searchErr}
			}()
		}
		left, right := <-results, <-results
		if left.err != nil || right.err != nil || !reflect.DeepEqual(left.page, right.page) {
			t.Fatalf("concurrent page %d = %+v, %+v", index, left, right)
		}
		page := left.page
		if len(page.Hits) != 1 || page.HasMore != (index < len(rows)-1) || (page.NextCursor != "") != page.HasMore {
			t.Fatalf("page %d = %+v; want one hit and exact continuation", index, page)
		}
		got = append(got, page.Hits...)
		request.Cursor = page.NextCursor
		if index == 3 {
			fresh := request
			fresh.Cursor = ""
			if _, err := peer.Search(store.ctx, fresh); err != nil {
				t.Fatalf("fresh search during active paging: %v", err)
			}
		}
	}
	if time.Since(started) <= ttl {
		t.Fatal("active traversal did not exceed the initial snapshot lifetime")
	}
	if !reflect.DeepEqual(got, complete.Hits) {
		t.Fatalf("active traversal = %+v; want %+v", got, complete.Hits)
	}
	request.Cursor = oldCursor
	replayed, err := peer.Search(store.ctx, request)
	if err != nil || !reflect.DeepEqual(replayed.Hits, complete.Hits[1:2]) {
		t.Fatalf("old cursor replay after renewal = %+v, %v", replayed, err)
	}
	short := store.open(t, func(config *library.Config) { config.SnapshotTTL = time.Second })
	if _, err := short.Search(store.ctx, request); err != nil {
		t.Fatalf("continuation with a shorter TTL: %v", err)
	}
	time.Sleep(2 * time.Second)
	if _, err := peer.Search(store.ctx, request); err != nil {
		t.Fatalf("continuation after a shorter renewal: %v", err)
	}
	request.Cursor = idle.NextCursor
	assertRenewalCursorExpired(t, peer, store, request)
	assertRenewalCursorExpired(t, active, store, request)
	time.Sleep(ttl + time.Second)
	request.Cursor = oldCursor
	assertRenewalCursorExpired(t, active, store, request)
	assertRenewalCursorExpired(t, peer, store, request)
}

func assertRenewalCursorExpired(t *testing.T, opened *library.Library, store *searchLiveStore, request library.SearchRequest) {
	t.Helper()
	page, err := opened.Search(store.ctx, request)
	if !errors.Is(err, library.ErrCursorExpired) || len(page.Hits) != 0 || page.HasMore || page.NextCursor != "" {
		t.Fatalf("inactive cursor = %+v, %v; want ErrCursorExpired and no page", page, err)
	}
}
