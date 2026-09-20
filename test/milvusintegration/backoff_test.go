//go:build milvusintegration

package milvusintegration

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const (
	// backoffMilvusMemory caps the Milvus container. Together with
	// backoffMilvusConfig it leaves the query node about half a GiB of load
	// headroom, so the big collection below cannot finish loading while the
	// small one still can.
	backoffMilvusMemory = "3g"
	backoffBigName      = "lmstest_backoff_big"
	backoffBigRows      = 320_000
	backoffBigDimension = 2048
	backoffBigBatchRows = 500
	backoffSmallName    = "lmstest_backoff_small"
	backoffSmallRows    = 2_000
	backoffSmallDim     = 1024
	// backoffLoadBoundMS shortens the daemon's per-poll load bound so the
	// unrecovered outcome (two polls plus one re-issue) arrives in about half a
	// minute instead of the production four and a half.
	backoffLoadBoundMS  = "10000"
	backoffDeferWindow  = 3 * time.Second
	backoffPauseInitial = 30 * time.Second
)

// backoffMilvusConfig makes the throwaway Milvus hold raw vectors in memory
// the way the operator's restored store did, and lowers the load admission
// threshold so the refusal happens well under the container cap instead of at
// an OOM kill. The image defaults do neither: queryNode.mmap.vectorField is
// true, and tiered storage warms the vector field up lazily
// (queryNode.segcore.tieredStorage.warmup.vectorField "disable"), so a
// collection of 2.5 GiB of vectors loads in seconds with almost no resident
// memory. With mmap off and warmup on, the query node's load admission check
// (memory in use plus queryNode.loadMemoryUsageFactor, default 2, times the
// segment size, against queryCoord.overloadedMemoryThresholdPercentage of the
// cgroup limit) refuses the segments of the big collection once about half a
// GiB is in use, logs "load segment failed, OOM if load", and the query
// coordinator keeps retrying them, so the collection stays loading. The keys
// and defaults are read from the Milvus paramtable this repository already
// depends on (github.com/milvus-io/milvus/pkg/v2/util/paramtable).
const backoffMilvusConfig = `queryNode:
  mmap:
    mmapEnabled: false
    vectorField: false
    vectorIndex: false
    scalarField: false
    scalarIndex: false
    growingMmapEnabled: false
  lazyload:
    enabled: false
  segcore:
    tieredStorage:
      evictionEnabled: false
      warmup:
        vectorField: sync
        vectorIndex: sync
        scalarField: sync
        scalarIndex: sync
queryCoord:
  overloadedMemoryThresholdPercentage: 50
`

// A collection Milvus cannot fit in memory pauses every other load. The big
// collection's raw vectors (2.5 GiB) do not fit under the load admission
// threshold of a 3g container, so its load fails: either the proxy refuses it
// with a memory error, or the query node refuses the segment loads and the
// collection stays loading past every daemon bound, which is the shape the
// incident actually produced. Either way the daemon must then refuse a second,
// small collection's load without sending Milvus a LoadCollection for it, and
// Milvus must keep reporting that collection not loaded for the whole pause.
// Once the pause elapses the small collection loads normally.
func TestMemoryExhaustedLoadPausesOtherLoadsInMilvus(t *testing.T) {
	requireIntegration(t)
	stack := startThrowawayStackWithMilvusConfig(t, backoffMilvusMemory, backoffMilvusConfig)
	milvus := dialMilvus(t, stack)
	// The collection is not loaded while it is written, so no query node holds
	// its growing rows; the write nodes sync their buffers to storage on their
	// own, and one flush at the end seals the last of them.
	seedCollection(t, milvus, seedSpec{name: backoffBigName, rows: backoffBigRows, dimension: backoffBigDimension, batchRows: backoffBigBatchRows, flushEvery: 0})
	seedCollection(t, milvus, seedSpec{name: backoffSmallName, rows: backoffSmallRows, dimension: backoffSmallDim, batchRows: backoffSmallRows, flushEvery: 0})

	recorder := &milvusCallRecorder{}
	cfg := integrationConfig(t, stack, map[string]string{
		"CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_TIMEOUT_MS": backoffLoadBoundMS,
	})
	service := newService(t, cfg, recorder)

	t.Logf("Milvus container memory before the load: %s", containerMemory(t, stack))
	started := time.Now()
	bigLease, bigErr := service.AcquireCollection(context.Background(), backoffBigName)
	if bigLease != nil {
		bigLease.Release()
	}
	bigState := loadState(t, milvus, backoffBigName)
	t.Logf("Milvus container memory after the load attempt: %s", containerMemory(t, stack))
	if bigErr == nil {
		t.Fatalf("Milvus loaded %d MiB of raw vectors under a %s container limit in %s (load state %v); the memory refusal was not reproduced, so check that the mounted user.yaml took effect (mmap off, vector warmup sync, threshold 50%%)", int64(backoffBigRows)*backoffBigDimension*4>>20, backoffMilvusMemory, time.Since(started).Round(time.Second), bigState.State)
	}
	signal := classifyBigLoadFailure(bigErr)
	if signal == "" {
		t.Fatalf("big collection load failed after %s with an error that is neither a Milvus memory refusal nor an unrecovered load: %v", time.Since(started).Round(time.Second), bigErr)
	}
	t.Logf("big collection load failed after %s via %s; Milvus load state now %v progress %d; error: %v", time.Since(started).Round(time.Second), signal, bigState.State, bigState.Progress, bigErr)

	recorder.reset()
	sampler := startLoadStateSampler(t, milvus, []string{backoffSmallName})
	deferredStarted := time.Now()
	smallLease, smallErr := service.AcquireCollection(context.Background(), backoffSmallName)
	if smallLease != nil {
		smallLease.Release()
	}
	if !errors.Is(smallErr, semantic.ErrCollectionLoadDeferred) {
		t.Fatalf("AcquireCollection(%s) during the pause returned %v after %s, want ErrCollectionLoadDeferred", backoffSmallName, smallErr, time.Since(deferredStarted).Round(time.Millisecond))
	}
	if elapsed := time.Since(deferredStarted); elapsed > backoffDeferWindow {
		t.Fatalf("the deferred acquire took %s, want a fast refusal", elapsed.Round(time.Millisecond))
	}
	time.Sleep(backoffDeferWindow)
	sampler.finish()
	if calls := recorder.count("LoadCollection", backoffSmallName); calls != 0 {
		t.Fatalf("daemon sent %d LoadCollection requests for %s during the pause, want 0", calls, backoffSmallName)
	}
	samples := sampler.snapshot()
	if len(samples) == 0 {
		t.Fatal("the sampler recorded no load states during the pause")
	}
	for _, sample := range samples {
		if len(sample.loading) != 0 || len(sample.loaded) != 0 {
			t.Fatalf("Milvus reported %s loading=%v loaded=%v during the pause at %s, want not loaded", backoffSmallName, sample.loading, sample.loaded, sample.at.Format(time.RFC3339Nano))
		}
	}
	t.Logf("Milvus reported %s not loaded in all %d samples during the pause", backoffSmallName, len(samples))

	// Release the collection Milvus cannot fit, the way an operator would, so
	// the query node stops retrying it and frees what it managed to load before
	// the pause elapses.
	releaseCtx, cancelRelease := context.WithTimeout(context.Background(), milvusCallTimeout)
	if err := milvus.ReleaseCollection(releaseCtx, milvusclient.NewReleaseCollectionOption(backoffBigName)); err != nil {
		cancelRelease()
		t.Fatalf("release %s: %v", backoffBigName, err)
	}
	cancelRelease()
	waitForLoadState(t, milvus, backoffBigName, entity.LoadStateNotLoad)
	t.Logf("Milvus container memory after releasing %s: %s", backoffBigName, containerMemory(t, stack))

	time.Sleep(backoffPauseInitial)
	resumedLease, resumedErr := service.AcquireCollection(context.Background(), backoffSmallName)
	if resumedErr != nil {
		t.Fatalf("AcquireCollection(%s) after the pause returned error: %v", backoffSmallName, resumedErr)
	}
	defer resumedLease.Release()
	if state := loadState(t, milvus, backoffSmallName); state.State != entity.LoadStateLoaded {
		t.Fatalf("Milvus reports %s load state %v after the pause, want loaded", backoffSmallName, state.State)
	}
	if calls := recorder.count("LoadCollection", backoffSmallName); calls != 1 {
		t.Fatalf("daemon sent %d LoadCollection requests for %s after the pause, want 1", calls, backoffSmallName)
	}
}

// classifyBigLoadFailure names which memory signal the daemon acted on, so
// the log says whether Milvus refused the load outright or let it hang. It
// returns "" for a failure that is neither, which the backoff must not treat
// as a memory signal.
func classifyBigLoadFailure(err error) string {
	lowered := strings.ToLower(err.Error())
	switch {
	case strings.Contains(lowered, "memory limit exceeded"), strings.Contains(lowered, "resource insufficient"), strings.Contains(lowered, "resourcetype=memory"), strings.Contains(lowered, "oom if load"):
		return "a direct Milvus memory refusal"
	case errors.Is(err, semantic.ErrCollectionNotReady):
		return "an unrecovered load (Milvus kept the collection loading past every bound)"
	default:
		return ""
	}
}
