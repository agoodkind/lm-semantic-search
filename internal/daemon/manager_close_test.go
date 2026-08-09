package daemon

import (
	"context"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
)

type closeTrackingSemantic struct {
	semanticIndex
	closed          bool
	jobStopped      <-chan struct{}
	closedBeforeJob bool
}

func (semantic *closeTrackingSemantic) Close(context.Context) error {
	semantic.closed = true
	if semantic.jobStopped != nil {
		select {
		case <-semantic.jobStopped:
		default:
			semantic.closedBeforeJob = true
		}
	}
	return nil
}

func TestManagerCloseClosesSemanticBackend(t *testing.T) {
	manager, _, _ := newTestManager(t)
	semantic := &closeTrackingSemantic{semanticIndex: manager.semantic}
	manager.semantic = semantic

	if err := manager.Close(context.Background()); err != nil {
		t.Fatalf("Manager.Close returned error: %v", err)
	}
	if !semantic.closed {
		t.Fatal("Manager.Close did not close the semantic backend")
	}
}

func TestManagerCloseCancelsAndWaitsForDetachedJob(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 1)
	repoPath := newCapTestRepo(t)
	jobEntered := make(chan struct{}, 1)
	jobStopped := make(chan struct{})
	releaseJob := make(chan struct{})
	manager.runner = fakeRunner{indexOne: func(
		ctx context.Context,
		_ string,
		_ string,
		_ model.IndexConfig,
	) (indexer.OneFileResult, error) {
		jobEntered <- struct{}{}
		select {
		case <-ctx.Done():
			close(jobStopped)
			return indexer.OneFileResult{}, ctx.Err()
		case <-releaseJob:
			return indexer.OneFileResult{}, nil
		}
	}}
	semantic := &closeTrackingSemantic{
		semanticIndex: manager.semantic,
		jobStopped:    jobStopped,
	}
	manager.semantic = semantic
	_, job := seedBootstrapCodebase(t, manager, repoPath, defaultIndexConfig())
	requestContext, cancelRequest := context.WithCancel(context.Background())
	manager.runJobAsync(requestContext, job.ID)
	select {
	case <-jobEntered:
	case <-time.After(5 * time.Second):
		t.Fatal("detached job did not enter the runner")
	}
	cancelRequest()

	manager.mu.Lock()
	jobDone := manager.done[job.ID]
	manager.mu.Unlock()
	t.Cleanup(func() {
		close(releaseJob)
		if jobDone != nil {
			<-jobDone
		}
	})

	closeContext, cancelClose := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancelClose()
	if err := manager.Close(closeContext); err != nil {
		t.Fatalf("Manager.Close returned error: %v", err)
	}
	select {
	case <-jobStopped:
	default:
		t.Fatal("detached job did not observe manager close cancellation")
	}
	if semantic.closedBeforeJob {
		t.Fatal("semantic backend closed before the detached job stopped")
	}
	if !semantic.closed {
		t.Fatal("semantic backend did not close after the detached job stopped")
	}
}

func TestNewManagerClosesSemanticBackendWhenLoadFails(t *testing.T) {
	cfg, _ := newTestManagerConfig(t)
	cfg.JobsPath = t.TempDir()
	semantic := &closeTrackingSemantic{semanticIndex: &fakeSemantic{}}

	manager, err := newManagerWithSemanticFactory(
		context.Background(),
		cfg,
		func(context.Context, config.Config) (semanticIndex, error) {
			return semantic, nil
		},
	)
	if err == nil {
		if manager != nil {
			_ = manager.Close(context.Background())
		}
		t.Fatal("newManagerWithSemanticFactory returned no load error")
	}
	if !semantic.closed {
		t.Fatal("semantic backend remained open after manager load failed")
	}
}
