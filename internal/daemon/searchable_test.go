package daemon

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/response"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc/codes"
	grpcstatus "google.golang.org/grpc/status"
)

// refreshDependencyHealth turns an active backend probe into the health record:
// a clean probe keeps the banner off, a probe failure degrades with the matching
// mode, and the debounce skips a repeat probe until the interval elapses.
func TestRefreshDependencyHealthProbe(t *testing.T) {
	manager, _, _ := newTestManager(t)

	manager.semantic = &fakeSemantic{}
	manager.refreshDependencyHealth(context.Background())
	if manager.DependencyHealth().Degraded() {
		t.Fatal("a clean probe must not degrade the health record")
	}

	resetProbeClock(manager)
	manager.semantic = &fakeSemantic{probeErr: adapterr.NewMilvusUnavailable(nil)}
	manager.refreshDependencyHealth(context.Background())
	if health := manager.DependencyHealth(); !health.Degraded() || health.Mode != dependencyStoreUnavailable {
		t.Fatalf("store probe failure: mode=%q degraded=%v, want %q degraded", health.Mode, health.Degraded(), dependencyStoreUnavailable)
	}

	// Within the debounce window the probe is skipped, so a now-healthy backend
	// does not yet clear the degraded mode.
	manager.semantic = &fakeSemantic{}
	manager.refreshDependencyHealth(context.Background())
	if !manager.DependencyHealth().Degraded() {
		t.Fatal("debounce must skip the probe and keep the prior degraded mode")
	}

	// Past the window the probe runs again and clears the mode.
	resetProbeClock(manager)
	manager.refreshDependencyHealth(context.Background())
	if manager.DependencyHealth().Degraded() {
		t.Fatal("a clean probe past the debounce window must clear the degraded mode")
	}
}

// A clean store probe clears a store outage but must not clear an embedder
// outage: ProbeHealth exercises only the store, while embedder health is observed
// from real embed outcomes. Clearing it on a store probe was a real bug.
func TestStoreProbeDoesNotClearEmbedderOutage(t *testing.T) {
	manager, _, _ := newTestManager(t)

	manager.mu.Lock()
	manager.noteDependencyFailureLocked(adapterr.NewEmbedderUnreachable(nil))
	manager.mu.Unlock()
	if got := manager.DependencyHealth().Mode; got != dependencyEmbedderUnreachable {
		t.Fatalf("setup: mode = %q, want %q", got, dependencyEmbedderUnreachable)
	}

	resetProbeClock(manager)
	manager.semantic = &fakeSemantic{}
	manager.refreshDependencyHealth(context.Background())
	if got := manager.DependencyHealth().Mode; got != dependencyEmbedderUnreachable {
		t.Fatalf("a clean store probe must not clear an embedder outage: mode = %q", got)
	}
}

// GetIndex reports searchable only when the path is indexed and the active
// backend probe succeeds. A store or embedder outage flips searchable to false
// even though the on-disk classification stays KIND_IN_SCOPE_INDEXED.
func TestGetIndexSearchableReflectsBackendHealth(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	server := NewGRPCServer(manager, nil)

	cases := []struct {
		name           string
		semantic       *fakeSemantic
		wantSearchable bool
	}{
		{"healthy backend", &fakeSemantic{}, true},
		{"embedder down", &fakeSemantic{probeErr: adapterr.NewEmbedderUnreachable(nil)}, false},
		{"store down", &fakeSemantic{unavailable: true}, false},
		{"collection loading", &fakeSemantic{collectionState: func(context.Context, string) (bool, bool, error) { return true, false, nil }}, true},
		{"collection load state unanswerable", &fakeSemantic{collectionState: func(context.Context, string) (bool, bool, error) {
			return false, false, adapterr.NewMilvusUnavailable(nil)
		}}, false},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			resetProbeClock(manager)
			manager.semantic = testCase.semantic
			resp, getErr := server.GetIndex(context.Background(), &pb.GetIndexRequest{Path: repoPath})
			if getErr != nil {
				t.Fatalf("GetIndex returned error: %v", getErr)
			}
			if resp.GetClassification().GetKind() != pb.PathClassification_KIND_IN_SCOPE_INDEXED {
				t.Fatalf("classification = %v, want KIND_IN_SCOPE_INDEXED", resp.GetClassification().GetKind())
			}
			if got := resp.GetSearchable(); got != testCase.wantSearchable {
				t.Fatalf("searchable = %v, want %v", got, testCase.wantSearchable)
			}
		})
	}
}

// GetIndex's searchable answer must survive JSON encoding as an explicit key,
// including the false case. protojson omits a plain proto3 bool at its zero
// value, so before this field carried explicit presence, a real false verdict
// and a key the daemon never set produced byte-identical JSON: no key at all.
// This asserts against the encoded bytes, because that omission happens in the
// encoder, not in the pb.GetIndexResponse struct fields a Go-level check would
// see.
func TestGetIndexSearchableSurvivesJSONEncoding(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	server := NewGRPCServer(manager, nil)

	cases := []struct {
		name     string
		semantic *fakeSemantic
		wantJSON string
	}{
		{"indexed and healthy encodes true", &fakeSemantic{}, `"searchable":true`},
		{"indexed but embedder down encodes false", &fakeSemantic{probeErr: adapterr.NewEmbedderUnreachable(nil)}, `"searchable":false`},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			resetProbeClock(manager)
			manager.semantic = testCase.semantic
			resp, getErr := server.GetIndex(context.Background(), &pb.GetIndexRequest{Path: repoPath})
			if getErr != nil {
				t.Fatalf("GetIndex returned error: %v", getErr)
			}
			encoded, marshalErr := response.MarshalCompactJSON(resp)
			if marshalErr != nil {
				t.Fatalf("MarshalCompactJSON returned error: %v", marshalErr)
			}
			if !strings.Contains(encoded, testCase.wantJSON) {
				t.Fatalf("encoded JSON = %q, want it to contain %q", encoded, testCase.wantJSON)
			}
		})
	}
}

// A per-path collection that is not loaded into query nodes, while the store
// itself is reachable, accepts search and reads loading, not the global
// store-unavailable banner. The per-path readiness drives searchable and the
// display; the global dependency banner stays off because no global probe failed.
// A separate genuine store outage still raises the banner (covered by the
// "store down" case in TestGetIndexSearchableReflectsBackendHealth).
func TestGetIndexCollectionNotLoadedReadsLoading(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed", CompletedAt: time.Now()}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	resetProbeClock(manager)
	// Store reachable (ProbeHealth nil) but this collection exists and is not loaded.
	manager.semantic = &fakeSemantic{collectionState: func(context.Context, string) (bool, bool, error) {
		return true, false, nil
	}}

	server := NewGRPCServer(manager, nil)
	resp, getErr := server.GetIndex(context.Background(), &pb.GetIndexRequest{Path: repoPath})
	if getErr != nil {
		t.Fatalf("GetIndex returned error: %v", getErr)
	}
	if !resp.GetSearchable() {
		t.Fatal("searchable = false for a loading collection, want true")
	}
	if got := resp.GetCodebase().GetDisplayStatus(); got != "loading" {
		t.Fatalf("display status = %q, want loading (not the global waiting banner)", got)
	}
	if dh := resp.GetDependencyHealth(); dh != nil && dh.GetMode() != "" {
		t.Fatalf("a per-path not-loaded collection must not set a global dependency mode, got %q", dh.GetMode())
	}
	if got := resp.GetCollectionReadiness(); got != "loading" {
		t.Fatalf("collection_readiness = %q, want loading", got)
	}
}

func TestGetIndexReportsIdleAsSearchableWithoutCountingRows(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	observationCalls := 0
	manager.semantic = &fakeSemantic{
		observeCollection: func(context.Context, string) (semantic.CollectionObservation, error) {
			observationCalls++
			return semantic.CollectionObservation{State: semantic.CollectionStateIdle}, nil
		},
		count: func(context.Context, string) (int32, error) {
			t.Fatal("idle status counted rows")
			return 0, nil
		},
	}

	response, getErr := NewGRPCServer(manager, nil).GetIndex(
		context.Background(),
		&pb.GetIndexRequest{Path: repoPath},
	)
	if getErr != nil {
		t.Fatalf("GetIndex returned error: %v", getErr)
	}
	if observationCalls != 1 {
		t.Fatalf("observation calls = %d, want 1", observationCalls)
	}
	if got := response.GetCollectionReadiness(); got != "idle" {
		t.Fatalf("collection readiness = %q, want idle", got)
	}
	if !response.GetSearchable() {
		t.Fatal("searchable = false, want true")
	}
	if got := response.GetCodebase().GetDisplayStatus(); got != "idle" {
		t.Fatalf("display status = %q, want idle", got)
	}
	if !strings.Contains(response.GetDisplayText(), "loads on demand") {
		t.Fatalf("idle display text does not explain on-demand loading:\n%s", response.GetDisplayText())
	}
}

func TestGetIndexAbsentCollectionNamesIndexRecovery(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	manager.semantic = &fakeSemantic{
		observeCollection: func(context.Context, string) (semantic.CollectionObservation, error) {
			return semantic.CollectionObservation{State: semantic.CollectionStateAbsent}, nil
		},
	}

	response, getErr := NewGRPCServer(manager, nil).GetIndex(
		context.Background(),
		&pb.GetIndexRequest{Path: repoPath},
	)
	if getErr != nil {
		t.Fatalf("GetIndex returned error: %v", getErr)
	}
	text := response.GetDisplayText()
	for _, want := range []string{"semantic collection is missing", "background repair", "index_codebase"} {
		if !strings.Contains(text, want) {
			t.Fatalf("absent collection guidance lacks %q:\n%s", want, text)
		}
	}
	if strings.Contains(text, "source directory is missing") {
		t.Fatalf("absent collection used filesystem recovery guidance:\n%s", text)
	}
}

func TestGetIndexUsesOneReadyObservationForReadinessAndRows(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	observationCalls := 0
	manager.semantic = &fakeSemantic{
		observeCollection: func(context.Context, string) (semantic.CollectionObservation, error) {
			observationCalls++
			return semantic.CollectionObservation{
				State:     semantic.CollectionStateReady,
				Rows:      359,
				RowsKnown: true,
			}, nil
		},
		count: func(context.Context, string) (int32, error) {
			t.Fatal("ready status issued a second row-count query")
			return 0, nil
		},
	}

	response, getErr := NewGRPCServer(manager, nil).GetIndex(
		context.Background(),
		&pb.GetIndexRequest{Path: repoPath},
	)
	if getErr != nil {
		t.Fatalf("GetIndex returned error: %v", getErr)
	}
	if observationCalls != 1 {
		t.Fatalf("observation calls = %d, want 1", observationCalls)
	}
	if !strings.Contains(response.GetDisplayText(), "current_index.total_chunks: 359") {
		t.Fatalf("display text did not use observed row count:\n%s", response.GetDisplayText())
	}
}

func TestListIndexesReportsIdleWithoutLoading(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	observationCalls := 0
	manager.semantic = &fakeSemantic{
		observeCollection: func(context.Context, string) (semantic.CollectionObservation, error) {
			observationCalls++
			return semantic.CollectionObservation{State: semantic.CollectionStateIdle}, nil
		},
		acquireCollection: func(context.Context, string) (semantic.CollectionLease, error) {
			t.Fatal("ListIndexes acquired a search lease")
			return nil, nil
		},
		count: func(context.Context, string) (int32, error) {
			t.Fatal("ListIndexes counted idle rows")
			return 0, nil
		},
	}

	response, listErr := NewGRPCServer(manager, nil).ListIndexes(
		context.Background(),
		&pb.ListIndexesRequest{},
	)
	if listErr != nil {
		t.Fatalf("ListIndexes returned error: %v", listErr)
	}
	if observationCalls != 1 {
		t.Fatalf("observation calls = %d, want 1", observationCalls)
	}
	if len(response.GetIndexes()) != 1 {
		t.Fatalf("indexes = %d, want 1", len(response.GetIndexes()))
	}
	if got := response.GetIndexes()[0].GetDisplayStatus(); got != "idle" {
		t.Fatalf("display status = %q, want idle", got)
	}
}

func TestSearchCodeHoldsCollectionLeaseThroughSearch(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	var leases atomic.Int32
	manager.semantic = &fakeSemantic{
		acquireCollection: func(_ context.Context, collectionName string) (semantic.CollectionLease, error) {
			if collectionName != "code_chunks_test" {
				t.Fatalf("collection name = %q, want code_chunks_test", collectionName)
			}
			leases.Add(1)
			return fakeCollectionLease{release: func() { leases.Add(-1) }}, nil
		},
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			if leases.Load() != 1 {
				t.Fatalf("active leases during search = %d, want 1", leases.Load())
			}
			return []model.StoredChunk{}, nil
		},
	}

	_, searchErr := NewGRPCServer(manager, nil).SearchCode(
		context.Background(),
		&pb.SearchCodeRequest{Path: repoPath, Query: "needle"},
	)
	if searchErr != nil {
		t.Fatalf("SearchCode returned error: %v", searchErr)
	}
	if leases.Load() != 0 {
		t.Fatalf("active leases after search = %d, want 0", leases.Load())
	}
}

func TestSearchCodeMapsCollectionLeaseErrors(t *testing.T) {
	tests := []struct {
		name        string
		err         error
		wantCode    codes.Code
		wantMessage string
	}{
		{
			name:        "load wait timeout",
			err:         semantic.ErrCollectionNotReady,
			wantCode:    codes.FailedPrecondition,
			wantMessage: "background collection load continues",
		},
		{
			name:     "Milvus outage",
			err:      adapterr.NewMilvusUnavailable(nil),
			wantCode: codes.Unavailable,
		},
		{name: "caller cancellation", err: context.Canceled, wantCode: codes.Canceled},
		{
			name:     "caller deadline",
			err:      context.DeadlineExceeded,
			wantCode: codes.DeadlineExceeded,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manager, _, repoPath := newTestManager(t)
			canonical, err := filepath.EvalSymlinks(repoPath)
			if err != nil {
				t.Fatalf("EvalSymlinks returned error: %v", err)
			}
			codebase := newCodebaseRecord(canonical)
			codebase.Status = model.CodebaseStatusIndexed
			manager.mu.Lock()
			manager.codebases[codebase.ID] = codebase
			manager.mu.Unlock()
			manager.semantic = &fakeSemantic{
				acquireCollection: func(context.Context, string) (semantic.CollectionLease, error) {
					return nil, test.err
				},
			}
			requestCtx := context.Background()
			cancelRequest := func() {}
			switch {
			case errors.Is(test.err, context.Canceled):
				requestCtx, cancelRequest = context.WithCancel(context.Background())
				cancelRequest()
			case errors.Is(test.err, context.DeadlineExceeded):
				requestCtx, cancelRequest = context.WithDeadline(
					context.Background(),
					time.Now().Add(-time.Second),
				)
			}
			defer cancelRequest()

			_, searchErr := NewGRPCServer(manager, nil).SearchCode(
				requestCtx,
				&pb.SearchCodeRequest{Path: repoPath, Query: "needle"},
			)
			if got := grpcstatus.Code(searchErr); got != test.wantCode {
				t.Fatalf("SearchCode status = %v, want %v: %v", got, test.wantCode, searchErr)
			}
			if test.wantMessage != "" && !strings.Contains(searchErr.Error(), test.wantMessage) {
				t.Fatalf("SearchCode error = %q, want %q", searchErr, test.wantMessage)
			}
		})
	}
}

func resetProbeClock(manager *Manager) {
	manager.mu.Lock()
	manager.lastDepProbeAt = time.Time{}
	manager.mu.Unlock()
}
