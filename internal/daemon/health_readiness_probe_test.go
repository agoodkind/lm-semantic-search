package daemon

import (
	"context"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/status"
)

// probeStartTimeout bounds how long a test waits for a surface to start its
// probe. A surface that never probes must fail the assertion quickly rather than
// wedging the run until the package-wide test timeout.
const probeStartTimeout = 10 * time.Second

// readinessSurface is one RPC that answers whether the daemon can serve. Each
// must reflect the current state of the shared dependencies on its own, without
// depending on some other caller having run first.
type readinessSurface struct {
	name string
	call func(context.Context, *GRPCServer, string) (string, error)
}

// readinessSurfaces are the three RPCs behind the readiness tools:
// get_indexing_status, list_indexing_statuses, and doctor_indexing. Each returns
// its display text so a test asserts on what the caller reads.
func readinessSurfaces() []readinessSurface {
	return []readinessSurface{
		{
			name: "GetIndex",
			call: func(ctx context.Context, server *GRPCServer, repoPath string) (string, error) {
				response, err := server.GetIndex(ctx, &pb.GetIndexRequest{Path: repoPath})
				if err != nil {
					return "", err
				}
				return response.GetDisplayText(), nil
			},
		},
		{
			name: "ListIndexes",
			call: func(ctx context.Context, server *GRPCServer, _ string) (string, error) {
				response, err := server.ListIndexes(ctx, &pb.ListIndexesRequest{})
				if err != nil {
					return "", err
				}
				return response.GetDisplayText(), nil
			},
		},
		{
			name: "Doctor",
			call: func(ctx context.Context, server *GRPCServer, _ string) (string, error) {
				response, err := server.Doctor(ctx, &pb.DoctorRequest{})
				if err != nil {
					return "", err
				}
				return response.GetDisplayText(), nil
			},
		},
	}
}

// newReadinessManager returns a manager with one indexed codebase and a store
// outage already on the record, recorded the way production records it: a search
// that failed against an unavailable store.
func newReadinessManager(t *testing.T) (*Manager, *GRPCServer, string) {
	t.Helper()

	manager, _, repoPath := newTestManager(t)
	canonicalPath, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonicalPath)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	manager.semantic = &fakeSemantic{
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return nil, semantic.ErrUnavailable
		},
	}
	server := NewGRPCServer(manager, nil)
	if _, searchErr := server.SearchCode(context.Background(), &pb.SearchCodeRequest{
		Path:  repoPath,
		Query: "anything",
		Limit: 5,
	}); searchErr == nil {
		t.Fatal("SearchCode succeeded against an unavailable store, so no outage was recorded")
	}
	if mode := manager.DependencyHealth().Mode; mode != dependencyStoreUnavailable {
		t.Fatalf("setup: dependency mode = %q, want %q", mode, dependencyStoreUnavailable)
	}
	return manager, server, repoPath
}

// Every readiness surface must answer about now. A store that recovers is
// reflected by the next readiness call on its own, with no search, no index job,
// and no other surface having run in between. Without a refresh of its own, a
// surface reports whatever was last recorded however old that is, which is not a
// bound on staleness at all.
func TestReadinessSurfacesReflectRecoveryWithoutAnotherCaller(t *testing.T) {
	for _, surface := range readinessSurfaces() {
		t.Run(surface.name, func(t *testing.T) {
			manager, server, repoPath := newReadinessManager(t)

			// The store returns. Nothing else runs: no search, no job, no other RPC.
			manager.semantic = &fakeSemantic{}
			resetProbeClock(manager)

			displayText, err := surface.call(context.Background(), server, repoPath)
			if err != nil {
				t.Fatalf("%s returned error: %v", surface.name, err)
			}
			if mode := manager.DependencyHealth().Mode; mode != dependencyHealthy {
				t.Fatalf("dependency mode = %q after %s, want it cleared by the surface's own probe", mode, surface.name)
			}
			headline := status.BannerHeadlineFor(dependencyStoreUnavailable)
			if strings.Contains(displayText, headline) {
				t.Fatalf("%s still shows the store banner after the store recovered:\n%s", surface.name, displayText)
			}
		})
	}
}

// A readiness surface must also report a live outage the record has never seen,
// so the probe answers in both directions rather than only clearing.
func TestReadinessSurfacesReportAnOutageTheRecordHasNotSeen(t *testing.T) {
	for _, surface := range readinessSurfaces() {
		t.Run(surface.name, func(t *testing.T) {
			manager, _, repoPath := newTestManager(t)
			canonicalPath, err := filepath.EvalSymlinks(repoPath)
			if err != nil {
				t.Fatalf("EvalSymlinks returned error: %v", err)
			}
			codebase := newCodebaseRecord(canonicalPath)
			codebase.Status = model.CodebaseStatusIndexed
			manager.mu.Lock()
			manager.codebases[codebase.ID] = codebase
			manager.mu.Unlock()
			manager.semantic = &fakeSemantic{probeErr: adapterr.NewMilvusUnavailable(nil)}
			resetProbeClock(manager)
			server := NewGRPCServer(manager, nil)

			displayText, callErr := surface.call(context.Background(), server, repoPath)
			if callErr != nil {
				t.Fatalf("%s returned error: %v", surface.name, callErr)
			}
			headline := status.BannerHeadlineFor(dependencyStoreUnavailable)
			if !strings.Contains(displayText, headline) {
				t.Fatalf("%s did not report a live store outage:\n%s", surface.name, displayText)
			}
		})
	}
}

// The debounce belongs to the manager, not to a caller, so sharing the probe
// across three surfaces must not multiply probe volume. This measures the count
// rather than assuming the debounce holds: many concurrent readiness calls cost
// one probe per interval in total.
func TestReadinessProbeVolumeStaysOneProbePerInterval(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonicalPath, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonicalPath)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	backend := &fakeSemantic{}
	manager.semantic = backend
	resetProbeClock(manager)
	server := NewGRPCServer(manager, nil)

	const callsPerSurface = 20
	surfaces := readinessSurfaces()
	hammerAllSurfaces := func() {
		var waitGroup sync.WaitGroup
		for _, surface := range surfaces {
			for range callsPerSurface {
				waitGroup.Add(1)
				go func() {
					defer waitGroup.Done()
					if _, callErr := surface.call(context.Background(), server, repoPath); callErr != nil {
						t.Errorf("%s returned error: %v", surface.name, callErr)
					}
				}()
			}
		}
		waitGroup.Wait()
	}

	callsPerInterval := int64(len(surfaces) * callsPerSurface)
	hammerAllSurfaces()
	if probes := backend.probeCount.Load(); probes != 1 {
		t.Fatalf("%d readiness calls inside one debounce interval issued %d probes, want exactly 1", callsPerInterval, probes)
	}

	// The interval elapses and the same load repeats. The rate is one probe per
	// interval, not one probe ever, so the record keeps tracking the store while
	// the cost stays independent of how many callers ask.
	resetProbeClock(manager)
	hammerAllSurfaces()
	if probes := backend.probeCount.Load(); probes != 2 {
		t.Fatalf("%d readiness calls across two debounce intervals issued %d probes, want exactly 2", 2*callsPerInterval, probes)
	}
}

// A probe that cannot complete must not become an indefinite wait. The probe
// runs under a deadline no later than dependencyProbeTimeout, and when it
// expires the caller still gets its answer, carrying the store banner the failed
// probe recorded.
func TestReadinessProbeIsBoundedByADeadline(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonicalPath, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonicalPath)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	deadlines := make(chan time.Duration, 1)
	manager.semantic = &fakeSemantic{
		probe: func(ctx context.Context) error {
			deadline, hasDeadline := ctx.Deadline()
			if !hasDeadline {
				deadlines <- 0
				return nil
			}
			deadlines <- time.Until(deadline)
			// A store that accepts the connection and never answers: block until the
			// bound expires rather than returning an error of our own.
			<-ctx.Done()
			return adapterr.NewMilvusUnavailable(ctx.Err())
		},
	}
	resetProbeClock(manager)
	server := NewGRPCServer(manager, nil)

	start := time.Now()
	response, listErr := server.ListIndexes(context.Background(), &pb.ListIndexesRequest{})
	elapsed := time.Since(start)
	if listErr != nil {
		t.Fatalf("ListIndexes returned error: %v", listErr)
	}

	budget := <-deadlines
	if budget <= 0 {
		t.Fatal("the probe ran with no deadline; a hung store would block the surface indefinitely")
	}
	if budget > dependencyProbeTimeout {
		t.Fatalf("probe deadline budget = %s, want no more than %s", budget, dependencyProbeTimeout)
	}
	if elapsed > dependencyProbeTimeout+5*time.Second {
		t.Fatalf("ListIndexes took %s against a hung probe, want it bounded near %s", elapsed, dependencyProbeTimeout)
	}
	headline := status.BannerHeadlineFor(dependencyStoreUnavailable)
	if !strings.Contains(response.GetDisplayText(), headline) {
		t.Fatalf("a probe that timed out left no store banner for the caller:\n%s", response.GetDisplayText())
	}
}

// A probe that passes must never clear a failure that arrived after it started.
// The probe's verdict describes the moment it began, so once several surfaces can
// trigger a refresh the in-flight window is hit far more often than it was with a
// single caller. The generation counter is what makes the stale verdict lose.
func TestPassingProbeDoesNotClearAFailureRecordedWhileItWasInFlight(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonicalPath, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonicalPath)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	probeStarted := make(chan struct{})
	releaseProbe := make(chan struct{})
	manager.semantic = &fakeSemantic{
		probe: func(context.Context) error {
			close(probeStarted)
			<-releaseProbe
			return nil
		},
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return nil, semantic.ErrUnavailable
		},
	}
	resetProbeClock(manager)
	server := NewGRPCServer(manager, nil)

	listDone := make(chan struct{})
	go func() {
		defer close(listDone)
		if _, listErr := server.ListIndexes(context.Background(), &pb.ListIndexesRequest{}); listErr != nil {
			t.Errorf("ListIndexes returned error: %v", listErr)
		}
	}()

	// The store dies after the probe read it as healthy but before the probe
	// returns, and a search records that newer failure.
	select {
	case <-probeStarted:
	case <-time.After(probeStartTimeout):
		close(releaseProbe)
		<-listDone
		t.Fatalf("ListIndexes never probed within %s, so it cannot answer a readiness question about now", probeStartTimeout)
	}
	if _, searchErr := server.SearchCode(context.Background(), &pb.SearchCodeRequest{
		Path:  repoPath,
		Query: "anything",
		Limit: 5,
	}); searchErr == nil {
		t.Fatal("SearchCode succeeded against an unavailable store, so no newer failure was recorded")
	}
	close(releaseProbe)
	<-listDone

	if mode := manager.DependencyHealth().Mode; mode != dependencyStoreUnavailable {
		t.Fatalf("dependency mode = %q, want %q; a probe that started before the failure must not clear it", mode, dependencyStoreUnavailable)
	}
}
