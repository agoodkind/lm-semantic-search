package daemon

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/cbm"
	"goodkind.io/lm-semantic-search/internal/model"
)

// TestRecordGraphIndexSkipsConcurrentDuplicate proves the in-flight guard keeps
// a second graph-index trigger for the same codebase from parsing the tree a
// second time while the first pass is still running. The injected graphIndex
// blocks the first pass; the second call must return without invoking it again.
func TestRecordGraphIndexSkipsConcurrentDuplicate(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	t.Cleanup(manager.CloseGraphEngines)

	codebase := newCodebaseRecord(repoPath)
	codebase.Kind = model.CodebaseKindCode
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	var indexCalls atomic.Int32
	entered := make(chan struct{})
	releaseWorker := make(chan struct{})
	manager.graphIndex = func(ctx context.Context, engine *cbm.Engine, canonicalPath string, mode string) error {
		indexCalls.Add(1)
		close(entered)
		<-releaseWorker
		return nil
	}

	firstDone := make(chan struct{})
	go func() {
		manager.recordGraphIndexNonFatal(context.Background(), codebase.ID, repoPath, "hash-1")
		close(firstDone)
	}()

	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("first graph index did not start")
	}

	secondDone := make(chan struct{})
	go func() {
		manager.recordGraphIndexNonFatal(context.Background(), codebase.ID, repoPath, "hash-2")
		close(secondDone)
	}()

	select {
	case <-secondDone:
	case <-time.After(2 * time.Second):
		t.Fatal("second graph index did not skip while the first was in flight")
	}
	if got := indexCalls.Load(); got != 1 {
		t.Fatalf("graphIndex call count = %d while first pass in flight, want 1", got)
	}

	close(releaseWorker)
	select {
	case <-firstDone:
	case <-time.After(2 * time.Second):
		t.Fatal("first graph index did not finish after release")
	}
	if got := indexCalls.Load(); got != 1 {
		t.Fatalf("graphIndex call count = %d after first pass finished, want 1", got)
	}
}

// TestRecordGraphIndexHoldsSlotWhileDetachedWorkerRuns proves the dupe-guard
// slot stays held after a cancelled pass detaches from its still-running
// worker: a new trigger during that window skips, and only after the worker
// exits can the next trigger index again.
func TestRecordGraphIndexHoldsSlotWhileDetachedWorkerRuns(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	t.Cleanup(manager.CloseGraphEngines)

	codebase := newCodebaseRecord(repoPath)
	codebase.Kind = model.CodebaseKindCode
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	var indexCalls atomic.Int32
	entered := make(chan struct{})
	releaseWorker := make(chan struct{})
	manager.graphIndex = func(ctx context.Context, engine *cbm.Engine, canonicalPath string, mode string) error {
		if indexCalls.Add(1) == 1 {
			close(entered)
			<-releaseWorker
		}
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	firstDone := make(chan struct{})
	go func() {
		manager.recordGraphIndexNonFatal(ctx, codebase.ID, repoPath, "hash-1")
		close(firstDone)
	}()

	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("first graph index did not start")
	}
	cancel()
	select {
	case <-firstDone:
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled pass did not return promptly while its worker was blocked")
	}

	manager.recordGraphIndexNonFatal(context.Background(), codebase.ID, repoPath, "hash-2")
	if got := indexCalls.Load(); got != 1 {
		t.Fatalf("graphIndex call count = %d while the detached worker still runs, want 1", got)
	}

	close(releaseWorker)
	deadline := time.Now().Add(2 * time.Second)
	for {
		manager.recordGraphIndexNonFatal(context.Background(), codebase.ID, repoPath, "hash-3")
		if indexCalls.Load() == 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("graphIndex call count = %d after worker exit, want 2", indexCalls.Load())
		}
		time.Sleep(10 * time.Millisecond)
	}
}
