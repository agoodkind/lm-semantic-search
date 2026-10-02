//go:build live

package library_test

import (
	"cmp"
	"context"
	"database/sql"
	_ "embed"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/embedding"
	librarymilvus "goodkind.io/lm-semantic-search/library/milvus"
	"goodkind.io/lm-semantic-search/library/observation"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

//go:embed search_verified_vectors.sql
var verifiedVectorsStatement string

//go:embed search_verified_snapshots.sql
var verifiedSnapshotsStatement string

type verifiedSearchEvents struct {
	mutex    sync.Mutex
	active   map[uint64]bool
	admitted chan struct{}
}

func (events *verifiedSearchEvents) Observe(event observation.Event) {
	if event.Operation != observation.VerifiedExactScoring {
		return
	}
	events.mutex.Lock()
	defer events.mutex.Unlock()
	if event.Phase == observation.Started {
		events.active[event.Scope.OperationID] = true
		select {
		case events.admitted <- struct{}{}:
		default:
		}
	} else {
		delete(events.active, event.Scope.OperationID)
	}
}

type verifiedSearchFixture struct {
	ctx     context.Context
	client  *milvusclient.Client
	store   *librarymilvus.Store
	config  library.Config
	library *library.Library
	events  *verifiedSearchEvents
	rows    []library.Occurrence
}

func newVerifiedSearchFixture(t *testing.T) *verifiedSearchFixture {
	t.Helper()
	address := os.Getenv("LMS_VERIFIED_SEARCH_MILVUS_ADDRESS")
	if address == "" {
		t.Skip("the real Milvus verified-search fixture is not enabled")
	}
	if address != "localhost:39630" && address != "127.0.0.1:39630" {
		t.Fatal("verified search requires the admitted isolated Milvus endpoint")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 180*time.Second)
	t.Cleanup(cancel)
	admin, err := milvusclient.New(ctx, &milvusclient.ClientConfig{Address: address})
	if err != nil {
		t.Fatal(err)
	}
	database := "lms_verified_search_" + fmt.Sprintf("%x", randomCommittedReaderBytes(t))
	prior, err := admin.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil || slices.Contains(prior, database) {
		t.Fatalf("private database prior absence: %v", err)
	}
	if err := admin.CreateDatabase(ctx, milvusclient.NewCreateDatabaseOption(database)); err != nil {
		t.Fatal(err)
	}
	t.Logf("private database %s", database)
	t.Cleanup(func() {
		deleteCommittedReaderDatabase(t, admin, database)
		cleanup, stop := context.WithTimeout(context.Background(), 30*time.Second)
		defer stop()
		if err := admin.Close(cleanup); err != nil {
			t.Error(err)
		}
	})
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{Address: address, DBName: database})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, stop := context.WithTimeout(context.Background(), 30*time.Second)
		defer stop()
		if err := client.DropCollection(cleanup, milvusclient.NewDropCollectionOption("verified")); err != nil {
			t.Error(err)
		}
		if err := client.Close(cleanup); err != nil {
			t.Error(err)
		}
	})
	events := &verifiedSearchEvents{active: make(map[uint64]bool), admitted: make(chan struct{}, 16)}
	store, err := librarymilvus.New(client, librarymilvus.Config{Database: database, Collection: "verified", Observer: events})
	if err != nil {
		t.Fatal(err)
	}
	embedder, err := embedding.NewOpenAI(ctx, embedding.OpenAIConfig{BaseURL: "http://127.0.0.1:5400/v1", Model: "nvidia/NV-EmbedCode-7b-v1", Dimension: 4096, RequestTimeout: 30 * time.Second, MaxAttempts: 1})
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	fixture := &verifiedSearchFixture{
		ctx: ctx, client: client, store: store, events: events,
		config: library.Config{Store: library.StoreDescriptor{
			CatalogPath: filepath.Join(root, "catalog.sqlite"), LockPath: filepath.Join(root, "catalog.lock"), PoolID: database,
			EmbeddingModel: "nvidia/NV-EmbedCode-7b-v1", EmbeddingRevision: "2a97ba03aee57d4c2b146fbd74ee84b3a219ddfac13e5d4ea32f373638d2d97f", Dimension: 4096, Normalization: "l2",
		}, Vectors: store, Embedder: embedder, SearchMode: library.Dense, QueryBlockSize: 4, QueryWorkers: 2, QueryTimeout: 30 * time.Second},
	}
	fixture.reopen(t)
	t.Cleanup(func() {
		if err := fixture.library.Close(); err != nil {
			t.Error(err)
		}
	})
	for _, namespace := range []string{"verified", "unrelated"} {
		if err := fixture.library.RegisterNamespace(ctx, library.NamespaceSpec{ID: namespace, Policy: library.ReplaceAllowed}); err != nil {
			t.Fatal(err)
		}
	}
	for index := range 9 {
		fixture.rows = append(fixture.rows, library.Occurrence{RowKey: fmt.Sprintf("row%02d", index), SortKey: fmt.Sprintf("sort%02d", 8-index), SourceText: "Verified source " + searchTopics[index], SearchText: searchTopics[index], EmbeddingInput: searchTopics[index]})
	}
	fixture.publish(t, "verified", "owner", 1, fixture.rows)
	return fixture
}

func (fixture *verifiedSearchFixture) reopen(t *testing.T) {
	t.Helper()
	if fixture.library != nil {
		if err := fixture.library.Close(); err != nil {
			t.Fatal(err)
		}
	}
	opened, err := library.Open(fixture.ctx, fixture.config)
	if err != nil {
		t.Fatal(err)
	}
	fixture.library = opened
}

func (fixture *verifiedSearchFixture) publish(t *testing.T, namespace, owner string, order uint64, rows []library.Occurrence) {
	t.Helper()
	if _, err := fixture.library.Apply(fixture.ctx, library.Batch{Namespace: namespace, OwnerID: owner, GenerationOrder: order, IdempotencyToken: fmt.Sprintf("%s-%d", owner, order), Mode: library.Replace, Rows: rows}); err != nil {
		t.Fatal(err)
	}
}

func (fixture *verifiedSearchFixture) catalog(t *testing.T) *sql.DB {
	t.Helper()
	database, err := sql.Open("sqlite3", "file:"+fixture.config.Store.CatalogPath+"?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := database.Close(); err != nil {
			t.Error(err)
		}
	})
	return database
}

type verifiedSearchRow struct {
	owner, key, sortKey string
	identity            library.VectorIdentity
}

func (fixture *verifiedSearchFixture) identities(t *testing.T) []verifiedSearchRow {
	t.Helper()
	rows, err := fixture.catalog(t).QueryContext(fixture.ctx, verifiedVectorsStatement)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := rows.Close(); err != nil {
			t.Error(err)
		}
	}()
	var selected []verifiedSearchRow
	for rows.Next() {
		var row verifiedSearchRow
		if err := rows.Scan(&row.owner, &row.key, &row.sortKey, &row.identity.ID, &row.identity.IdentityDigest, &row.identity.Checksum); err != nil {
			t.Fatal(err)
		}
		selected = append(selected, row)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(selected) != 9 {
		t.Fatalf("published identities=%d, want 9", len(selected))
	}
	return selected
}

func testVerifiedNativeSearchParityPagingAndCache(t *testing.T, fixture *verifiedSearchFixture) {
	selected := fixture.identities(t)
	request := library.SearchRequest{Namespace: "verified", Query: searchTopics[3], PageSize: 2, MinScore: 0}
	query, err := fixture.config.Embedder.EmbedBatch(fixture.ctx, []string{request.Query})
	if err != nil {
		t.Fatal(err)
	}
	native := make(map[string]float64)
	for offset := 0; offset < len(selected); offset += 4 {
		end := min(offset+4, len(selected))
		identities := make([]library.VectorIdentity, 0, end-offset)
		ids := make([]string, 0, end-offset)
		for _, row := range selected[offset:end] {
			identities = append(identities, row.identity)
			ids = append(ids, row.identity.ID)
		}
		if err := fixture.store.VerifyStrong(fixture.ctx, identities); err != nil {
			t.Fatal(err)
		}
		separate, err := fixture.store.ScoreExact(fixture.ctx, query[0], ids)
		if err != nil {
			t.Fatal(err)
		}
		combined, err := fixture.store.ScoreExactVerified(fixture.ctx, query[0], identities)
		if err != nil {
			t.Fatal(err)
		}
		if len(combined) != len(separate) {
			t.Fatalf("combined scores=%d separate=%d", len(combined), len(separate))
		}
		for index, score := range separate {
			if score.ID != combined[index].ID || math.Float32bits(float32(score.Score)) != math.Float32bits(float32(combined[index].Score)) {
				t.Fatalf("native score mismatch for %s", score.ID)
			}
			native[score.ID] = score.Score
		}
	}
	got := pageAll(t, fixture.library, request, 2)
	fixture.assertCleanup(t, fixture.snapshots(t))
	slices.SortFunc(selected, func(left, right verifiedSearchRow) int {
		if result := cmp.Compare(native[right.identity.ID], native[left.identity.ID]); result != 0 {
			return result
		}
		if result := cmp.Compare(left.sortKey, right.sortKey); result != 0 {
			return result
		}
		return cmp.Compare(left.key, right.key)
	})
	if len(got) != len(selected) {
		t.Fatalf("hits=%d want %d", len(got), len(selected))
	}
	for index, hit := range got {
		row := selected[index]
		if hit.ID.OwnerID != row.owner || hit.ID.RowKey != row.key || hit.Score != native[row.identity.ID] {
			t.Fatalf("ranked hit %d does not match native score/order: ID=%+v score=%v", index, hit.ID, hit.Score)
		}
		for _, input := range fixture.rows {
			if input.RowKey == row.key && hit.SourceText != input.SourceText {
				t.Fatalf("source text mismatch for %s", row.key)
			}
		}
	}
	tail := fixture.identities(t)[8].identity
	record := fixture.readBackend(t, tail)
	changed := record
	changed.IdentityDigest = strings.Repeat("0", 64)
	fixture.writeBackend(t, changed)
	if page, err := fixture.library.Search(fixture.ctx, request); err != nil || len(page.Hits) != 2 {
		t.Fatalf("same-revision warm search: hits=%d error=%v", len(page.Hits), err)
	}
	unrelatedKey := fixture.identities(t)[0].key
	for _, row := range fixture.rows {
		if row.RowKey == unrelatedKey {
			fixture.publish(t, "unrelated", "new-owner", 1, []library.Occurrence{row})
		}
	}
	fixture.assertFailure(t, request, library.ErrVectorCorrupt)
	fixture.writeBackend(t, record)
}

func (fixture *verifiedSearchFixture) readBackend(t *testing.T, identity library.VectorIdentity) library.VectorRecord {
	t.Helper()
	result, err := fixture.client.Query(fixture.ctx, milvusclient.NewQueryOption("verified").WithIDs(column.NewColumnVarChar("vector_id", []string{identity.ID})).WithOutputFields("vector").WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		t.Fatal(err)
	}
	values, ok := result.GetColumn("vector").(*column.ColumnFloatVector)
	if !ok || result.ResultCount != 1 {
		t.Fatal("private backend vector is absent")
	}
	return library.VectorRecord{ID: identity.ID, IdentityDigest: identity.IdentityDigest, Checksum: identity.Checksum, Values: slices.Clone(values.Data()[0])}
}

func (fixture *verifiedSearchFixture) writeBackend(t *testing.T, record library.VectorRecord) {
	t.Helper()
	result, err := fixture.client.Upsert(fixture.ctx, milvusclient.NewColumnBasedInsertOption("verified", column.NewColumnVarChar("vector_id", []string{record.ID}), column.NewColumnVarChar("identity_digest", []string{record.IdentityDigest}), column.NewColumnVarChar("vector_checksum", []string{record.Checksum}), column.NewColumnFloatVector("vector", 4096, [][]float32{record.Values})))
	if err != nil || result.UpsertCount != 1 {
		t.Fatalf("private backend mutation: %v", err)
	}
}

func (fixture *verifiedSearchFixture) snapshots(t *testing.T) int64 {
	t.Helper()
	var count int64
	if err := fixture.catalog(t).QueryRowContext(fixture.ctx, verifiedSnapshotsStatement).Scan(&count); err != nil {
		t.Fatal(err)
	}
	return count
}

func (fixture *verifiedSearchFixture) assertFailure(t *testing.T, request library.SearchRequest, want error) {
	t.Helper()
	before := fixture.snapshots(t)
	page, err := fixture.library.Search(fixture.ctx, request)
	if !errors.Is(err, want) || len(page.Hits) != 0 || page.HasMore || page.NextCursor != "" {
		t.Fatalf("failed search hits=%d has_more=%t cursor_present=%t error=%v want %v", len(page.Hits), page.HasMore, page.NextCursor != "", err, want)
	}
	fixture.assertCleanup(t, before)
}

func (fixture *verifiedSearchFixture) assertCleanup(t *testing.T, snapshots int64) {
	t.Helper()
	if got := fixture.snapshots(t); got != snapshots {
		t.Fatalf("failed query changed snapshots %d to %d", snapshots, got)
	}
	files, err := filepath.Glob(filepath.Join(filepath.Dir(fixture.config.Store.CatalogPath), ".lms-query-*.sqlite*"))
	if err != nil || len(files) != 0 {
		t.Fatalf("query cleanup files=%v error=%v", files, err)
	}
	fixture.events.mutex.Lock()
	defer fixture.events.mutex.Unlock()
	if len(fixture.events.active) != 0 {
		t.Fatal("search returned with an admitted combined operation still active")
	}
}

func testVerifiedNativeSearchRejectsTailDamageAndJoinsCancellation(t *testing.T, fixture *verifiedSearchFixture) {
	tail := fixture.identities(t)[8].identity
	record := fixture.readBackend(t, tail)
	request := library.SearchRequest{Namespace: "verified", Query: searchTopics[3], PageSize: 2, MinScore: 0}
	for _, damage := range []string{"digest", "checksum", "bytes", "missing"} {
		t.Run(damage, func(t *testing.T) {
			fixture.reopen(t)
			changed := record
			changed.Values = slices.Clone(record.Values)
			want := library.ErrVectorCorrupt
			switch damage {
			case "digest":
				changed.IdentityDigest = strings.Repeat("0", 64)
			case "checksum":
				changed.Checksum = strings.Repeat("0", 64)
			case "bytes":
				changed.Values[0] += 0.125
			case "missing":
				want = library.ErrVectorMissing
			}
			if damage == "missing" {
				if _, err := fixture.client.Delete(fixture.ctx, milvusclient.NewDeleteOption("verified").WithStringIDs("vector_id", []string{tail.ID})); err != nil {
					t.Fatal(err)
				}
			} else {
				fixture.writeBackend(t, changed)
			}
			fixture.assertFailure(t, request, want)
			fixture.writeBackend(t, record)
		})
	}
	fixture.reopen(t)
	for len(fixture.events.admitted) > 0 {
		<-fixture.events.admitted
	}
	before := fixture.snapshots(t)
	ctx, cancel := context.WithCancel(fixture.ctx)
	defer cancel()
	type result struct {
		page library.SearchPage
		err  error
	}
	completed := make(chan result, 1)
	go func() { page, err := fixture.library.Search(ctx, request); completed <- result{page: page, err: err} }()
	select {
	case <-fixture.events.admitted:
		cancel()
	case response := <-completed:
		t.Fatalf("search completed before combined admission: %v", response.err)
	case <-fixture.ctx.Done():
		t.Fatal(fixture.ctx.Err())
	}
	select {
	case response := <-completed:
		if (!errors.Is(response.err, context.Canceled) && status.Code(response.err) != codes.Canceled) || len(response.page.Hits) != 0 || response.page.HasMore || response.page.NextCursor != "" {
			t.Fatalf("cancelled search hits=%d has_more=%t cursor_present=%t error=%v", len(response.page.Hits), response.page.HasMore, response.page.NextCursor != "", response.err)
		}
	case <-fixture.ctx.Done():
		t.Fatal("cancelled search did not join before its bounded caller deadline")
	}
	fixture.assertCleanup(t, before)
}

func TestVerifiedNativeSearch(t *testing.T) {
	fixture := newVerifiedSearchFixture(t)
	t.Run("parity paging and cache", func(t *testing.T) { testVerifiedNativeSearchParityPagingAndCache(t, fixture) })
	t.Run("tail damage and cancellation", func(t *testing.T) { testVerifiedNativeSearchRejectsTailDamageAndJoinsCancellation(t, fixture) })
	t.Run("filtered batches preserve bytes and frozen scalars", func(t *testing.T) { testVerifiedSearchFilteredBatches(t, fixture) })
}

func testVerifiedSearchFilteredBatches(t *testing.T, fixture *verifiedSearchFixture) {
	const namespace = "filtered-batches"
	const scalarText = "nul\x00 quote\" slash\\ Unicode雪🙂"
	spec := library.NamespaceSpec{ID: namespace, Policy: library.ReplaceAllowed, Scalars: []library.ScalarColumn{
		{Name: "label", Type: library.String, MaxLength: 256, Mutable: true, Nullable: true},
		{Name: "optional", Type: library.String, MaxLength: 256, Nullable: true},
		{Name: "number", Type: library.Int64, Mutable: true, Nullable: true},
		{Name: "flag", Type: library.Bool, Mutable: true, Nullable: true},
	}}
	if err := fixture.library.RegisterNamespace(fixture.ctx, spec); err != nil {
		t.Fatal(err)
	}
	owners := []string{"plain-owner" + strings.Repeat("p", 4096), "raw\xffowner\x00" + strings.Repeat("r", 4096)}
	projections := make([]library.ScalarProjection, 0, len(owners))
	for ownerIndex, owner := range owners {
		rowCount := 33 + ownerIndex
		rows := make([]library.Occurrence, 0, rowCount)
		projection := library.ScalarProjection{Namespace: namespace, OwnerID: owner, ProjectionOrder: 1, IdempotencyToken: "first", Rows: make(map[string]map[string]library.ScalarValue)}
		for index := range rowCount {
			row := fixture.rows[index%len(fixture.rows)]
			row.RowKey = fmt.Sprintf("row%03d\x00\"\\雪", index)
			row.SortKey = fmt.Sprintf("sort%03d", rowCount-index)
			row.Scalars = map[string]library.ScalarValue{"label": stringValue("original"), "number": {Type: library.Int64, Int64: math.MaxInt64}, "flag": {Type: library.Bool, Bool: true}}
			if index%2 == 0 {
				row.Scalars["optional"] = library.ScalarValue{Type: library.String, Null: true}
			}
			if ownerIndex == 1 && index >= rowCount-2 {
				row.Scalars["optional"] = stringValue("excluded")
			}
			rows = append(rows, row)
			projection.Rows[row.RowKey] = map[string]library.ScalarValue{"label": stringValue(scalarText)}
			if ownerIndex == 0 && index == 0 {
				projection.Rows[row.RowKey] = map[string]library.ScalarValue{
					"label":  {Type: library.String, Null: true},
					"number": {Type: library.Int64, Null: true},
					"flag":   {Type: library.Bool, Null: true},
				}
			}
		}
		fixture.publish(t, namespace, owner, 1, rows)
		if _, err := fixture.library.ReprojectScalars(fixture.ctx, projection); err != nil {
			t.Fatal(err)
		}
		projections = append(projections, projection)
	}
	request := library.SearchRequest{Namespace: namespace, Query: searchTopics[3], PageSize: 7, MinScore: 0}
	baseline := pageAll(t, fixture.library, request, request.PageSize)
	if len(baseline) != 67 {
		t.Fatalf("unfiltered occurrences=%d, want 67", len(baseline))
	}
	var expected []library.SearchHit
	var equalExpected []library.SearchHit
	for _, hit := range baseline {
		if hit.ID.OwnerID == owners[0] && hit.ID.RowKey == "row000\x00\"\\雪" {
			if !hit.Scalars["label"].Null || !hit.Scalars["number"].Null || !hit.Scalars["flag"].Null {
				t.Fatalf("explicit-null projection did not override published values for %+v", hit.ID)
			}
			continue
		}
		if hit.Scalars["label"].String != scalarText || hit.Scalars["number"].Int64 != math.MaxInt64 {
			t.Fatalf("unfiltered scalar bytes changed for %+v", hit.ID)
		}
		equalExpected = append(equalExpected, hit)
		value, present := hit.Scalars["optional"]
		if !present || value.Null {
			expected = append(expected, hit)
		}
	}
	if len(expected) != 64 {
		t.Fatalf("eligible occurrences=%d, want 64", len(expected))
	}
	for _, filter := range []library.Filter{
		{Op: library.Equal, Column: "label", Values: []library.ScalarValue{stringValue(scalarText)}},
		{Op: library.Equal, Column: "number", Values: []library.ScalarValue{{Type: library.Int64, Int64: math.MaxInt64}}},
		{Op: library.Equal, Column: "flag", Values: []library.ScalarValue{{Type: library.Bool, Bool: true}}},
	} {
		request.Filter = &filter
		assertHitsEqual(t, "equal "+filter.Column, pageAll(t, fixture.library, request, request.PageSize), equalExpected)
	}
	request.Filter = &library.Filter{Op: library.All, Children: []library.Filter{
		{Op: library.Equal, Column: "label", Values: []library.ScalarValue{stringValue(scalarText)}},
		{Op: library.Equal, Column: "number", Values: []library.ScalarValue{{Type: library.Int64, Int64: math.MaxInt64}}},
		{Op: library.Equal, Column: "flag", Values: []library.ScalarValue{{Type: library.Bool, Bool: true}}},
		{Op: library.Any, Children: []library.Filter{
			{Op: library.IsNull, Column: "optional"},
			{Op: library.Not, Children: []library.Filter{{Op: library.IsPresent, Column: "optional"}}},
		}},
	}}
	page, err := fixture.library.Search(fixture.ctx, request)
	if err != nil || len(page.Hits) != request.PageSize || !page.HasMore || page.NextCursor == "" {
		t.Fatalf("filtered first page hits=%d has_more=%t error=%v", len(page.Hits), page.HasMore, err)
	}
	for _, projection := range projections {
		projection.ProjectionOrder = 2
		projection.IdempotencyToken = "second"
		for key := range projection.Rows {
			projection.Rows[key] = map[string]library.ScalarValue{"label": stringValue("changed after snapshot")}
		}
		if _, err := fixture.library.ReprojectScalars(fixture.ctx, projection); err != nil {
			t.Fatal(err)
		}
	}
	got := slices.Clone(page.Hits)
	for page.HasMore {
		request.Cursor = page.NextCursor
		page, err = fixture.library.Search(fixture.ctx, request)
		if err != nil || len(page.Hits) == 0 || (page.HasMore && len(page.Hits) != request.PageSize) || (!page.HasMore && page.NextCursor != "") {
			t.Fatalf("filtered continuation hits=%d has_more=%t error=%v", len(page.Hits), page.HasMore, err)
		}
		got = append(got, page.Hits...)
	}
	assertHitsEqual(t, "filtered frozen batches", got, expected)
	request.Cursor = ""
	page, err = fixture.library.Search(fixture.ctx, request)
	if err != nil || len(page.Hits) != 0 || page.HasMore || page.NextCursor != "" {
		t.Fatalf("new search retained old projection: hits=%d error=%v", len(page.Hits), err)
	}
}
