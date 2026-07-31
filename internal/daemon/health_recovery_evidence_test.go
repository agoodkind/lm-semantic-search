package daemon

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/status"
)

// A recorded store outage must survive a surface that does not probe. Job
// surfaces report job state and render the banner incidentally, so they read the
// record without refreshing it, and the absence of a probe must never be taken
// for recovery.
//
// The store client's availability flag is what made that mistake reachable. It
// is a boot-time latch that stays true after the store dies, so inferring
// recovery from it erased a true outage: the store went down, a search recorded
// it, and the next unprobed read cleared the banner while every codebase still
// rendered as indexed. This fake reproduces the latch, reporting itself
// available throughout.
func TestUnprobedSurfaceKeepsStoreOutageDespiteLatchedConnection(t *testing.T) {
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

	// The store is down but the client still reports itself available, which is
	// the state the latch leaves behind after a store that once dialled dies.
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

	// The debounce is clear, so nothing suppresses a refresh. ListJobs still must
	// not upgrade the record, because it never probes and holds no evidence.
	resetProbeClock(manager)

	response, listErr := server.ListJobs(context.Background(), &pb.ListJobsRequest{})
	if listErr != nil {
		t.Fatalf("ListJobs returned error: %v", listErr)
	}
	if mode := response.GetDependencyHealth().GetMode(); mode != string(dependencyStoreUnavailable) {
		t.Fatalf("dependency mode = %q, want %q; an unprobed read must not clear a recorded outage", mode, dependencyStoreUnavailable)
	}
	headline := status.BannerHeadlineFor(dependencyStoreUnavailable)
	if !strings.Contains(response.GetDisplayText(), headline) {
		t.Fatalf("job list display text lost the store banner %q, got:\n%s", headline, response.GetDisplayText())
	}
}

// A store probe that answers is positive evidence, so it still clears the store
// outage on the surface that runs it. Recovery must stay reachable after the
// latched-connection shortcut is gone.
func TestGetIndexClearsStoreOutageOnceTheProbeAnswers(t *testing.T) {
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

	// The store returns and answers the probe GetIndex runs.
	resetProbeClock(manager)
	manager.semantic = &fakeSemantic{}

	response, getErr := server.GetIndex(context.Background(), &pb.GetIndexRequest{Path: repoPath})
	if getErr != nil {
		t.Fatalf("GetIndex returned error: %v", getErr)
	}
	if mode := response.GetDependencyHealth().GetMode(); mode != "" {
		t.Fatalf("dependency mode = %q after a probe answered, want it cleared", mode)
	}
	headline := status.BannerHeadlineFor(dependencyStoreUnavailable)
	if strings.Contains(response.GetDisplayText(), headline) {
		t.Fatalf("the store banner outlived a successful probe, got:\n%s", response.GetDisplayText())
	}
}
