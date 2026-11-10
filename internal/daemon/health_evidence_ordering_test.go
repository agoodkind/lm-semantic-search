package daemon

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/status"
)

// newIndexedReadinessManager returns a manager holding one indexed codebase, the
// precondition every readiness surface needs before it will render a row.
func newIndexedReadinessManager(t *testing.T) (*Manager, *GRPCServer, string) {
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
	return manager, NewGRPCServer(manager, nil), repoPath
}

// The banner names one dependency and shows one last-reachable time beside it, so
// that time has to be a fact about that dependency. The store probe answers for
// the store alone, and an embedding endpoint that has not answered in this
// daemon's lifetime must not borrow the probe's round trip and read as reachable
// moments ago while it is the thing that is down.
func TestEmbedderBannerDoesNotShowTheStoreProbesReachability(t *testing.T) {
	manager, server, repoPath := newIndexedReadinessManager(t)

	// An embedder outage recorded the way production records it, from a search
	// whose query embed failed against the endpoint.
	manager.semantic = &fakeSemantic{
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return nil, adapterr.NewEmbedderUnreachable(errors.New("connection refused"))
		},
	}
	if _, searchErr := server.SearchCode(context.Background(), &pb.SearchCodeRequest{
		Path:  repoPath,
		Query: "anything",
		Limit: 5,
	}); searchErr == nil {
		t.Fatal("SearchCode succeeded against an unreachable embedder, so no outage was recorded")
	}
	if mode := manager.DependencyHealth().Mode; mode != dependencyEmbedderUnreachable {
		t.Fatalf("setup: dependency mode = %q, want %q", mode, dependencyEmbedderUnreachable)
	}

	// The store is healthy throughout and answers the probe this surface runs. The
	// embedder is still down and has still never answered.
	manager.semantic = &fakeSemantic{}
	resetProbeClock(manager)

	response, listErr := server.ListIndexes(context.Background(), &pb.ListIndexesRequest{})
	if listErr != nil {
		t.Fatalf("ListIndexes returned error: %v", listErr)
	}
	displayText := response.GetDisplayText()
	headline := status.BannerHeadlineFor(dependencyEmbedderUnreachable)
	if !strings.Contains(displayText, headline) {
		t.Fatalf("the embedder outage stopped showing after a store probe:\n%s", displayText)
	}
	if !strings.Contains(displayText, "last reachable unknown") {
		t.Fatalf("the embedder banner reports a last-reachable time that nothing proved about the embedder:\n%s", displayText)
	}
	if stamp := response.GetDependencyHealth().GetLastHealthyAt(); stamp != nil {
		t.Fatalf("last_healthy_at = %s beside mode %q, but the embedder has never answered", stamp.AsTime(), response.GetDependencyHealth().GetMode())
	}
}

// Every degraded mode has to be attributed to the dependency it is a fact
// about. The attribution switches carry a default that reads whichever
// dependency answered most recently, so a mode nobody listed borrows the store
// probe's round trip and reports it beside an embedder's name. This walks the
// whole set rather than the modes that happened to exist when it was written,
// so adding a mode without attributing it fails here.
func TestEveryDegradedModeIsAttributedToItsOwnDependency(t *testing.T) {
	storeAnsweredAt := time.Date(2026, 7, 31, 12, 0, 0, 0, time.UTC)
	embedderAnsweredAt := time.Date(2026, 7, 31, 9, 0, 0, 0, time.UTC)

	embedderModes := []dependencyMode{
		dependencyEmbedderUnreachable,
		dependencyEmbedderRejected,
		dependencyEmbedderPaused,
		dependencyEmbedderBusy,
	}

	for _, mode := range embedderModes {
		t.Run(string(mode), func(t *testing.T) {
			health := dependencyHealth{
				Mode:                mode,
				Since:               storeAnsweredAt,
				StoreReachableAt:    storeAnsweredAt,
				EmbedderReachableAt: embedderAnsweredAt,
			}
			if got := health.lastReachableAt(); !got.Equal(embedderAnsweredAt) {
				t.Fatalf("%s reports last reachable %s, want the embedder's own %s", mode, got, embedderAnsweredAt)
			}
			if covers := dependenciesFor(mode); !covers.Embedder || covers.Store {
				t.Fatalf("%s covers {Store:%t Embedder:%t}, want the embedder alone", mode, covers.Store, covers.Embedder)
			}
		})
	}

	t.Run(string(dependencyStoreUnavailable), func(t *testing.T) {
		health := dependencyHealth{
			Mode:                dependencyStoreUnavailable,
			Since:               embedderAnsweredAt,
			StoreReachableAt:    storeAnsweredAt,
			EmbedderReachableAt: embedderAnsweredAt,
		}
		if got := health.lastReachableAt(); !got.Equal(storeAnsweredAt) {
			t.Fatalf("store outage reports last reachable %s, want the store's own %s", got, storeAnsweredAt)
		}
		if covers := dependenciesFor(dependencyStoreUnavailable); !covers.Store || covers.Embedder {
			t.Fatalf("store outage covers {Store:%t Embedder:%t}, want the store alone", covers.Store, covers.Embedder)
		}
	})
}

// A store outage still carries the store's own reachability time, so fixing the
// embedder banner must not leave the store banner with nothing to show.
func TestStoreBannerShowsTheStoresOwnReachability(t *testing.T) {
	manager, server, repoPath := newIndexedReadinessManager(t)

	// A search reaches both dependencies, which is what proves the store answered.
	manager.semantic = &fakeSemantic{
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return []model.StoredChunk{}, nil
		},
	}
	if _, searchErr := server.SearchCode(context.Background(), &pb.SearchCodeRequest{
		Path:  repoPath,
		Query: "anything",
		Limit: 5,
	}); searchErr != nil {
		t.Fatalf("SearchCode returned error: %v", searchErr)
	}

	// The store then dies and its own probe says so.
	manager.semantic = &fakeSemantic{probeErr: adapterr.NewMilvusUnavailable(nil)}
	resetProbeClock(manager)

	response, listErr := server.ListIndexes(context.Background(), &pb.ListIndexesRequest{})
	if listErr != nil {
		t.Fatalf("ListIndexes returned error: %v", listErr)
	}
	displayText := response.GetDisplayText()
	if !strings.Contains(displayText, status.BannerHeadlineFor(dependencyStoreUnavailable)) {
		t.Fatalf("ListIndexes did not report the store outage:\n%s", displayText)
	}
	if strings.Contains(displayText, "last reachable unknown") {
		t.Fatalf("the store banner lost the time the store actually answered:\n%s", displayText)
	}
	if response.GetDependencyHealth().GetLastHealthyAt() == nil {
		t.Fatal("last_healthy_at is unset beside a store outage, but a search reached the store moments earlier")
	}
}

// Evidence is ordered by when it was gathered, in both directions. A probe that
// started before a search reached both dependencies must not reinstate the outage
// it saw, because the search is the newer evidence: telling an operator the store
// is down after something has since read from it is the same class of false
// answer as reporting healthy while it is down.
func TestFailingProbeDoesNotEraseASuccessRecordedWhileItWasInFlight(t *testing.T) {
	manager, server, repoPath := newIndexedReadinessManager(t)

	probeStarted := make(chan struct{})
	releaseProbe := make(chan struct{})
	manager.semantic = &fakeSemantic{
		probe: func(context.Context) error {
			close(probeStarted)
			<-releaseProbe
			return adapterr.NewMilvusUnavailable(errors.New("connection refused"))
		},
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return []model.StoredChunk{}, nil
		},
	}
	resetProbeClock(manager)

	listDone := make(chan struct{})
	go func() {
		defer close(listDone)
		if _, listErr := server.ListIndexes(context.Background(), &pb.ListIndexesRequest{}); listErr != nil {
			t.Errorf("ListIndexes returned error: %v", listErr)
		}
	}()

	// The store comes back after the probe read it as down but before the probe
	// returns, and a search that reached both dependencies records that recovery.
	select {
	case <-probeStarted:
	case <-time.After(probeStartTimeout):
		close(releaseProbe)
		<-listDone
		t.Fatalf("ListIndexes never probed within %s", probeStartTimeout)
	}
	if _, searchErr := server.SearchCode(context.Background(), &pb.SearchCodeRequest{
		Path:  repoPath,
		Query: "anything",
		Limit: 5,
	}); searchErr != nil {
		t.Fatalf("SearchCode returned error: %v", searchErr)
	}
	close(releaseProbe)
	<-listDone

	// ListJobs reads the record without probing, so it shows what the racing
	// writers left behind.
	response, jobsErr := server.ListJobs(context.Background(), &pb.ListJobsRequest{})
	if jobsErr != nil {
		t.Fatalf("ListJobs returned error: %v", jobsErr)
	}
	if mode := response.GetDependencyHealth().GetMode(); mode != "" {
		t.Fatalf("dependency mode = %q; a probe that started before the search must not reinstate an outage the search disproved", mode)
	}
	if headline := status.BannerHeadlineFor(dependencyStoreUnavailable); strings.Contains(response.GetDisplayText(), headline) {
		t.Fatalf("a stale probe put the store banner back after a search reached the store:\n%s", response.GetDisplayText())
	}
}

// A probe error the outage classes do not recognize is still a store that did not
// answer. Dropping it would let a readiness surface report healthy immediately
// after its own probe failed.
func TestProbeFailureWithAnUnclassifiedErrorStillReportsAnOutage(t *testing.T) {
	_, server, _ := newIndexedReadinessManagerWithProbe(t, func(context.Context) error {
		return errors.New("dial tcp: connection refused")
	})

	response, listErr := server.ListIndexes(context.Background(), &pb.ListIndexesRequest{})
	if listErr != nil {
		t.Fatalf("ListIndexes returned error: %v", listErr)
	}
	headline := status.BannerHeadlineFor(dependencyStoreUnavailable)
	if !strings.Contains(response.GetDisplayText(), headline) {
		t.Fatalf("a failed probe left no outage for the caller to see:\n%s", response.GetDisplayText())
	}
}

// A probe that answered is positive evidence whoever asked for it. When the
// caller walks away mid-call the round trip has already happened, and discarding
// its answer spends the probe interval on nothing while every other surface keeps
// reading an outage the store has already recovered from.
func TestProbeSuccessSurvivesACallerThatWalkedAway(t *testing.T) {
	manager, server, repoPath := newIndexedReadinessManager(t)

	manager.semantic = &fakeSemantic{
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return nil, adapterr.NewMilvusUnavailable(errors.New("connection refused"))
		},
	}
	if _, searchErr := server.SearchCode(context.Background(), &pb.SearchCodeRequest{
		Path:  repoPath,
		Query: "anything",
		Limit: 5,
	}); searchErr == nil {
		t.Fatal("SearchCode succeeded against an unavailable store, so no outage was recorded")
	}

	// The store returns and answers the probe, and the caller gives up while the
	// answer is on its way back.
	callerCtx, cancelCaller := context.WithCancel(context.Background())
	defer cancelCaller()
	manager.semantic = &fakeSemantic{
		probe: func(context.Context) error {
			cancelCaller()
			return nil
		},
	}
	resetProbeClock(manager)
	//nolint:errcheck // The abandoned caller's own result is not the subject; the record it left behind is.
	_, _ = server.ListIndexes(callerCtx, &pb.ListIndexesRequest{})

	response, jobsErr := server.ListJobs(context.Background(), &pb.ListJobsRequest{})
	if jobsErr != nil {
		t.Fatalf("ListJobs returned error: %v", jobsErr)
	}
	if mode := response.GetDependencyHealth().GetMode(); mode != "" {
		t.Fatalf("dependency mode = %q; the store answered a probe, so its answer is evidence whether or not the caller waited for it", mode)
	}
}

// newIndexedReadinessManagerWithProbe is newIndexedReadinessManager with the
// probe outcome a test wants and the debounce cleared, so the next readiness call
// probes.
func newIndexedReadinessManagerWithProbe(t *testing.T, probe func(context.Context) error) (*Manager, *GRPCServer, string) {
	t.Helper()

	manager, server, repoPath := newIndexedReadinessManager(t)
	manager.semantic = &fakeSemantic{probe: probe}
	resetProbeClock(manager)
	return manager, server, repoPath
}
