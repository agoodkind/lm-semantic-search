//go:build milvusintegration

package milvusintegration

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/semantic/milvusgrpc"
	"google.golang.org/protobuf/proto"
)

const (
	seedIDField        = "id"
	seedVectorField    = "vector"
	seedIndexName      = "vector_flat"
	milvusCallTimeout  = 2 * time.Minute
	milvusReadyPoll    = 2 * time.Second
	loadStatePoll      = 25 * time.Millisecond
	loadStateWait      = 2 * time.Minute
	storeCallStackSkip = 2
)

// milvusCall is one unary request the daemon's own client sent to Milvus,
// captured on the wire by the transport's call observer. Nothing is stubbed:
// the request still reaches the real store.
type milvusCall struct {
	method          string
	collectionNames []string
	recordedAt      time.Time
}

// milvusCallRecorder keeps every call the daemon's Milvus client made.
type milvusCallRecorder struct {
	mutex sync.Mutex
	calls []milvusCall
}

func (recorder *milvusCallRecorder) observe(method string, _ string, request proto.Message) {
	collectionNames := make([]string, 0, 2)
	if named, ok := request.(interface{ GetCollectionName() string }); ok && named.GetCollectionName() != "" {
		collectionNames = append(collectionNames, named.GetCollectionName())
	}
	if separator := strings.LastIndex(method, "/"); separator >= 0 {
		method = method[separator+1:]
	}
	recorder.mutex.Lock()
	recorder.calls = append(recorder.calls, milvusCall{
		method:          method,
		collectionNames: collectionNames,
		recordedAt:      time.Now(),
	})
	recorder.mutex.Unlock()
}

func (recorder *milvusCallRecorder) snapshot() []milvusCall {
	recorder.mutex.Lock()
	defer recorder.mutex.Unlock()
	return slices.Clone(recorder.calls)
}

func (recorder *milvusCallRecorder) reset() {
	recorder.mutex.Lock()
	recorder.calls = nil
	recorder.mutex.Unlock()
}

func (recorder *milvusCallRecorder) count(method string, collectionName string) int {
	count := 0
	for _, call := range recorder.snapshot() {
		if call.method == method && (collectionName == "" || slices.Contains(call.collectionNames, collectionName)) {
			count++
		}
	}
	return count
}

// observedContext returns a Milvus construction context that reports every
// unary call to recorder. It is the same seam the daemon's live suite uses.
func observedContext(recorder *milvusCallRecorder) context.Context {
	return context.WithValue(
		context.Background(),
		milvusgrpc.CallObserverContextKey{},
		milvusgrpc.CallObserver(recorder.observe),
	)
}

// dialMilvus connects a test-owned client to the throwaway stack, retrying
// until the proxy answers a list call, and closes it at cleanup.
func dialMilvus(t *testing.T, stack throwawayStack) *milvusclient.Client {
	t.Helper()
	deadline := time.Now().Add(stackReadyTimeout)
	for {
		dialCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		client, err := milvusclient.New(dialCtx, &milvusclient.ClientConfig{
			Address:     stack.milvusAddress(),
			DialOptions: milvusgrpc.DialOptions(context.Background(), slog.Default(), milvusgrpc.DefaultCallTimeouts()),
		})
		if err == nil {
			_, err = client.ListCollections(dialCtx, milvusclient.NewListCollectionOption())
		}
		cancel()
		if err == nil {
			t.Cleanup(func() {
				closeCtx, closeCancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer closeCancel()
				_ = client.Close(closeCtx)
			})
			return client
		}
		if time.Now().After(deadline) {
			t.Fatalf("milvus at %s not answering after %s: %v", stack.milvusAddress(), stackReadyTimeout, err)
		}
		time.Sleep(milvusReadyPoll)
	}
}

// seedSpec describes one collection the test writes into the throwaway store.
type seedSpec struct {
	name      string
	rows      int64
	dimension int
	batchRows int64
	// flushEvery flushes after this many batches so growing data never piles
	// up in memory; zero flushes once at the end.
	flushEvery int
}

// bytes is the raw float32 vector payload the seed writes, which is also what
// a FLAT index holds in memory once the collection loads.
func (spec seedSpec) bytes() int64 {
	return spec.rows * int64(spec.dimension) * 4
}

// seedCollection creates the collection, inserts its rows in batches, flushes,
// builds a FLAT vector index (raw vectors in memory, no build time), waits for
// Milvus to report the index built, and leaves the collection not loaded.
func seedCollection(t *testing.T, client *milvusclient.Client, spec seedSpec) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	started := time.Now()

	schema := entity.NewSchema().
		WithField(entity.NewField().WithName(seedIDField).WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName(seedVectorField).WithDataType(entity.FieldTypeFloatVector).WithDim(int64(spec.dimension)))
	if err := client.CreateCollection(ctx, milvusclient.NewCreateCollectionOption(spec.name, schema)); err != nil {
		t.Fatalf("create collection %s: %v", spec.name, err)
	}

	batches := 0
	for start := int64(0); start < spec.rows; start += spec.batchRows {
		end := min(start+spec.batchRows, spec.rows)
		ids := make([]int64, 0, end-start)
		vectors := make([][]float32, 0, end-start)
		for id := start; id < end; id++ {
			ids = append(ids, id)
			vectors = append(vectors, seedVector(id, spec.dimension))
		}
		insertOption := milvusclient.NewColumnBasedInsertOption(spec.name).
			WithInt64Column(seedIDField, ids).
			WithFloatVectorColumn(seedVectorField, spec.dimension, vectors)
		if _, err := client.Insert(ctx, insertOption); err != nil {
			t.Fatalf("insert rows %d..%d into %s: %v", start, end, spec.name, err)
		}
		batches++
		if spec.flushEvery > 0 && batches%spec.flushEvery == 0 {
			flushCollection(t, ctx, client, spec.name)
		}
	}
	flushCollection(t, ctx, client, spec.name)

	indexTask, err := client.CreateIndex(ctx, milvusclient.NewCreateIndexOption(spec.name, seedVectorField, index.NewFlatIndex(entity.COSINE)).WithIndexName(seedIndexName))
	if err != nil {
		t.Fatalf("create FLAT index on %s: %v", spec.name, err)
	}
	if err := indexTask.Await(ctx); err != nil {
		t.Fatalf("await FLAT index on %s: %v", spec.name, err)
	}
	ensureNotLoaded(t, client, spec.name)
	t.Logf("seeded %s: %d rows x dim %d (%d MiB raw) in %s", spec.name, spec.rows, spec.dimension, spec.bytes()>>20, time.Since(started).Round(time.Second))
}

func flushCollection(t *testing.T, ctx context.Context, client *milvusclient.Client, name string) {
	t.Helper()
	flushTask, err := client.Flush(ctx, milvusclient.NewFlushOption(name))
	if err != nil {
		t.Fatalf("flush %s: %v", name, err)
	}
	if err := flushTask.Await(ctx); err != nil {
		t.Fatalf("await flush of %s: %v", name, err)
	}
}

// seedVector derives a deterministic vector from the row id.
func seedVector(id int64, dimension int) []float32 {
	vector := make([]float32, dimension)
	for i := range vector {
		vector[i] = float32((id*31+int64(i)*17)%1000) / 1000
	}
	return vector
}

// ensureNotLoaded releases the collection if Milvus reports it loaded and then
// waits for the not-loaded state, so a test starts from a cold collection.
func ensureNotLoaded(t *testing.T, client *milvusclient.Client, name string) {
	t.Helper()
	if state := loadState(t, client, name); state.State != entity.LoadStateNotLoad {
		ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
		defer cancel()
		if err := client.ReleaseCollection(ctx, milvusclient.NewReleaseCollectionOption(name)); err != nil {
			t.Fatalf("release %s: %v", name, err)
		}
	}
	waitForLoadState(t, client, name, entity.LoadStateNotLoad)
}

// loadState reads one collection's load state from Milvus.
func loadState(t *testing.T, client *milvusclient.Client, name string) entity.LoadState {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	state, err := client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(name))
	if err != nil {
		t.Fatalf("get load state for %s: %v", name, err)
	}
	return state
}

func waitForLoadState(t *testing.T, client *milvusclient.Client, name string, want entity.LoadStateCode) {
	t.Helper()
	deadline := time.Now().Add(loadStateWait)
	for {
		state := loadState(t, client, name)
		if state.State == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("collection %s load state = %v after %s, want %v", name, state.State, loadStateWait, want)
		}
		time.Sleep(loadStatePoll)
	}
}

// loadSample is one reading of every watched collection's load state.
type loadSample struct {
	at      time.Time
	loading []string
	loaded  []string
}

// loadStateSampler polls Milvus for the load state of a set of collections on
// a short interval, so a test can see how many were loading at the same
// instant from the store's own point of view.
type loadStateSampler struct {
	client  *milvusclient.Client
	names   []string
	mutex   sync.Mutex
	samples []loadSample
	stop    chan struct{}
	done    chan struct{}
}

func startLoadStateSampler(t *testing.T, client *milvusclient.Client, names []string) *loadStateSampler {
	t.Helper()
	sampler := &loadStateSampler{
		client:  client,
		names:   slices.Clone(names),
		mutex:   sync.Mutex{},
		samples: nil,
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
	}
	go sampler.run()
	t.Cleanup(sampler.finish)
	return sampler
}

func (sampler *loadStateSampler) run() {
	defer close(sampler.done)
	ticker := time.NewTicker(loadStatePoll)
	defer ticker.Stop()
	for {
		select {
		case <-sampler.stop:
			return
		case <-ticker.C:
			sampler.sampleOnce()
		}
	}
}

func (sampler *loadStateSampler) sampleOnce() {
	ctx, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	sample := loadSample{at: time.Now(), loading: nil, loaded: nil}
	for _, name := range sampler.names {
		state, err := sampler.client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(name))
		if err != nil {
			return
		}
		switch state.State {
		case entity.LoadStateLoading:
			sample.loading = append(sample.loading, name)
		case entity.LoadStateLoaded:
			sample.loaded = append(sample.loaded, name)
		case entity.LoadStateUnloading, entity.LoadStateNotLoad:
		}
	}
	sampler.mutex.Lock()
	sampler.samples = append(sampler.samples, sample)
	sampler.mutex.Unlock()
}

// finish stops sampling and waits for the last poll.
func (sampler *loadStateSampler) finish() {
	select {
	case <-sampler.stop:
	default:
		close(sampler.stop)
	}
	<-sampler.done
}

func (sampler *loadStateSampler) snapshot() []loadSample {
	sampler.mutex.Lock()
	defer sampler.mutex.Unlock()
	return slices.Clone(sampler.samples)
}

// peakLoading reports the largest number of collections Milvus reported
// loading at one instant, the number of samples that saw any load in flight,
// and a compact trace of the samples that saw the peak.
func (sampler *loadStateSampler) peakLoading() (int, int, string) {
	peak := 0
	busy := 0
	var trace strings.Builder
	for _, sample := range sampler.snapshot() {
		if len(sample.loading) > 0 {
			busy++
		}
		if len(sample.loading) > peak {
			peak = len(sample.loading)
			trace.Reset()
		}
		if len(sample.loading) == peak && peak > 0 && trace.Len() < 2000 {
			_, _ = fmt.Fprintf(&trace, "%s loading=%v loaded=%d\n", sample.at.Format(time.RFC3339Nano), sample.loading, len(sample.loaded))
		}
	}
	return peak, busy, trace.String()
}
