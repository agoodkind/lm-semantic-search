package semantic

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/config"
)

const (
	// defaultCollectionLoadBound bounds one wait for a collection to become
	// queryable when the operator configured no value. Ninety seconds matches the
	// release wait in mmap.go, which is the other place a Milvus state transition
	// is polled, and is long enough for a large collection to materialize on a
	// healthy query node.
	defaultCollectionLoadBound = 90 * time.Second
	// collectionLoadPollInterval is the gap between load-state reads while a load
	// is in flight. It is coarse on purpose: the state rarely flips within a
	// quarter second and each read is a round trip to Milvus.
	collectionLoadPollInterval = 500 * time.Millisecond
)

// collectionLoadState is one poll's view of a collection's load, reduced to the
// facts the wait acts on. Milvus reports a progress percentage only while a load
// is in flight, so ProgressKnown says whether Progress carries a real reading
// rather than the zero value of a state that exposes none.
type collectionLoadState struct {
	Loaded        bool
	Progress      int64
	ProgressKnown bool
}

// collectionLoadProbe reads one collection's current load state.
type collectionLoadProbe func(ctx context.Context) (collectionLoadState, error)

// collectionLoadRequest re-issues the load for one collection. Milvus treats a
// repeated LoadCollection on an already-loading or already-loaded collection as
// a no-op, which is what makes the single recovery attempt safe to run.
type collectionLoadRequest func(ctx context.Context) error

type collectionLoadOperation func(ctx context.Context) error

type collectionLoadFlight struct {
	done chan struct{}
	err  error
}

// collectionLoadCoordinator shares one in-flight load operation per collection
// name. Every caller observes that load through its own wait, so a caller that
// cancels ends only its own wait.
type collectionLoadCoordinator struct {
	mutex   sync.Mutex
	flights map[string]*collectionLoadFlight
}

// Do joins the in-flight load for collectionName or starts one, and returns
// when either that load finishes or ctx ends.
//
// The caller that starts the load does not run it inline. A load run on the
// first caller's context would fail for the whole cohort the moment that one
// caller cancelled, which is the opposite of what sharing the load is for, so
// the operation runs detached from every caller and ceiling is what ends it
// once nobody is waiting on it any more.
func (coordinator *collectionLoadCoordinator) Do(
	ctx context.Context,
	collectionName string,
	ceiling time.Duration,
	operation collectionLoadOperation,
) error {
	flight, starting := coordinator.joinOrStart(collectionName)
	if starting {
		coordinator.runFlight(ctx, collectionName, ceiling, flight, operation)
	}
	return waitForCollectionLoadFlight(ctx, collectionName, flight)
}

// joinOrStart returns the flight this caller waits on and whether this caller
// is the one that must run it.
func (coordinator *collectionLoadCoordinator) joinOrStart(
	collectionName string,
) (*collectionLoadFlight, bool) {
	coordinator.mutex.Lock()
	defer coordinator.mutex.Unlock()
	if flight, found := coordinator.flights[collectionName]; found {
		return flight, false
	}
	if coordinator.flights == nil {
		coordinator.flights = make(map[string]*collectionLoadFlight)
	}
	flight := &collectionLoadFlight{done: make(chan struct{}), err: nil}
	coordinator.flights[collectionName] = flight
	return flight, true
}

// runFlight runs the shared load on a context that keeps the starting caller's
// values, so correlation ids and spans survive, but none of its cancellation or
// deadline. The load is left running when its starting caller leaves, which is
// the right trade because LoadCollection is idempotent and the collection it
// finishes loading serves whoever asks next.
//
// The outcome is published from inside this goroutine rather than on the
// starting caller's return path. Publishing earlier would retire the flight
// while the load was still running and the next caller would start a second
// concurrent load for the same collection. The deferred publish also means a
// panicking operation cannot leave followers parked until their own contexts
// expire.
func (coordinator *collectionLoadCoordinator) runFlight(
	ctx context.Context,
	collectionName string,
	ceiling time.Duration,
	flight *collectionLoadFlight,
	operation collectionLoadOperation,
) {
	loadCtx, cancelLoad := context.WithTimeout(context.WithoutCancel(ctx), ceiling)
	go func() {
		defer cancelLoad()
		var operationErr error
		defer func() {
			if recovered := recover(); recovered != nil {
				operationErr = fmt.Errorf(
					"shared load of collection %s panicked: %v",
					collectionName,
					recovered,
				)
				slog.ErrorContext(loadCtx, "semantic.collection_load_panic",
					"component", "semantic",
					"subcomponent", "load",
					"collection", collectionName,
					"err", operationErr,
				)
			}
			coordinator.finishFlight(collectionName, flight, operationErr)
		}()
		operationErr = operation(loadCtx)
	}()
}

// finishFlight publishes one flight's outcome and retires it, so the next
// caller starts a fresh load rather than joining a finished one.
func (coordinator *collectionLoadCoordinator) finishFlight(
	collectionName string,
	flight *collectionLoadFlight,
	operationErr error,
) {
	coordinator.mutex.Lock()
	defer coordinator.mutex.Unlock()
	flight.err = operationErr
	delete(coordinator.flights, collectionName)
	close(flight.done)
}

func waitForCollectionLoadFlight(
	ctx context.Context,
	collectionName string,
	flight *collectionLoadFlight,
) error {
	select {
	case <-ctx.Done():
		slog.WarnContext(ctx, "semantic.collection_load_shared_wait_cancelled",
			"component", "semantic",
			"subcomponent", "load",
			"collection", collectionName,
			"err", ctx.Err(),
		)
		return fmt.Errorf(
			"wait for shared collection load %s: %w",
			collectionName,
			ctx.Err(),
		)
	case <-flight.done:
		return flight.err
	}
}

// waitForCollectionLoad blocks until collectionName reports loaded, or reports a
// stuck load once the bound elapses twice: once on the initial wait, and once
// more on a single re-issued load.
//
// The recovery is deliberately one attempt against a fixed bound. Re-issuing the
// load is idempotent and clears the case where the original load request was
// lost or the collection was released underneath the caller, but a query node
// that cannot materialize the collection will not be fixed by asking again, so
// looping on it would only convert a bounded failure back into an unbounded
// wait. The two polls and bounded recovery request take at most three times
// bound, after which the caller gets a typed error naming the collection, how
// long it waited, and the last progress Milvus reported.
func waitForCollectionLoad(
	ctx context.Context,
	collectionName string,
	bound time.Duration,
	probe collectionLoadProbe,
	reissue collectionLoadRequest,
) error {
	startedAt := clock.Now()

	state, err := pollCollectionLoad(ctx, bound, probe)
	if err != nil {
		return err
	}
	if state.Loaded {
		return nil
	}
	slog.ErrorContext(ctx, "semantic.collection_load_stuck",
		"component", "semantic",
		"subcomponent", "load",
		"collection", collectionName,
		"waited_ms", clock.Now().Sub(startedAt).Milliseconds(),
		"bound_ms", bound.Milliseconds(),
		"load_progress", loadProgressField(state),
		"err", ErrCollectionNotReady,
	)

	if err := reissueWithin(ctx, bound, reissue); err != nil {
		return err
	}
	state, err = pollCollectionLoad(ctx, bound, probe)
	if err != nil {
		return err
	}
	waited := clock.Now().Sub(startedAt)
	if state.Loaded {
		slog.WarnContext(ctx, "semantic.collection_load_recovered",
			"component", "semantic",
			"subcomponent", "load",
			"collection", collectionName,
			"waited_ms", waited.Milliseconds(),
		)
		return nil
	}

	stuck := fmt.Errorf(
		"collection %s did not become queryable within %s after a re-issued load (load progress %s): %w",
		collectionName, waited, loadProgressField(state), ErrCollectionNotReady,
	)
	slog.ErrorContext(ctx, "semantic.collection_load_unrecovered",
		"component", "semantic",
		"subcomponent", "load",
		"collection", collectionName,
		"waited_ms", waited.Milliseconds(),
		"bound_ms", bound.Milliseconds(),
		"load_progress", loadProgressField(state),
		"err", stuck,
	)
	return stuck
}

// reissueWithin bounds the single recovery load request under the same budget
// as one wait, so a store that accepts the connection but never answers the
// request cannot stretch the total past three times the configured bound.
func reissueWithin(ctx context.Context, bound time.Duration, reissue collectionLoadRequest) error {
	boundedCtx, cancel := context.WithTimeout(ctx, bound)
	defer cancel()
	return reissue(boundedCtx)
}

// pollCollectionLoad reads the load state until the collection is loaded, the
// bound elapses, or the caller's context ends. Exhausting the bound is not an
// error here: it returns the last observed state so the caller decides whether
// to recover or report. A probe that fails because the bound expired mid-call is
// the same outcome as the bound expiring between calls, so it is reported as an
// unfinished load rather than as a store outage; a genuine probe failure is
// returned as the classified error the probe produced.
func pollCollectionLoad(
	ctx context.Context,
	bound time.Duration,
	probe collectionLoadProbe,
) (collectionLoadState, error) {
	boundedCtx, cancel := context.WithTimeout(ctx, bound)
	defer cancel()

	ticker := time.NewTicker(collectionLoadPollInterval)
	defer ticker.Stop()

	last := collectionLoadState{Loaded: false, Progress: 0, ProgressKnown: false}
	for {
		state, err := probe(boundedCtx)
		if err != nil {
			if boundedCtx.Err() != nil && ctx.Err() == nil {
				return last, nil
			}
			return last, err
		}
		last = state
		if state.Loaded {
			return state, nil
		}
		select {
		case <-boundedCtx.Done():
			if cancelled := ctx.Err(); cancelled != nil {
				slog.WarnContext(ctx, "semantic.collection_load_wait_cancelled",
					"component", "semantic",
					"subcomponent", "load",
					"err", cancelled,
				)
				return last, fmt.Errorf("wait for collection load: %w", cancelled)
			}
			return last, nil
		case <-ticker.C:
		}
	}
}

// loadProgressField renders the reported load progress for a log field and an
// error message, so a state that exposes no percentage reads as unknown instead
// of as a misleading zero.
func loadProgressField(state collectionLoadState) string {
	if !state.ProgressKnown {
		return "unknown"
	}
	return fmt.Sprintf("%d%%", state.Progress)
}

// collectionLoadBound returns the configured bound for one collection load,
// falling back to the built-in bound when the operator set none or set a value
// the conversion rejected.
func (service *Service) collectionLoadBound() time.Duration {
	if bound := config.MilvusCollectionLoadTimeout(service.cfg.MilvusCollectionLoadTimeoutMS); bound > 0 {
		return bound
	}
	return defaultCollectionLoadBound
}

// sharedCollectionLoadCeiling is the whole-operation bound for one shared load.
// The shared load answers to no caller's cancellation, so this is the only
// thing that ends it: one initial LoadCollection under the store's metadata
// bound, then awaitCollectionLoaded's worst case of two polls and one recovery
// request, each under the configured load bound. Stating the sum once here
// keeps the guarantee readable, and keeps it correct if any one of those bounds
// changes later.
func (service *Service) sharedCollectionLoadCeiling() time.Duration {
	return service.callTimeouts().Metadata + 3*service.collectionLoadBound()
}

// awaitCollectionLoaded waits for collectionName to become queryable under the
// configured bound, reading the authoritative GetLoadState enum rather than the
// client's own LoadTask.Await, whose wait is unbounded. GetLoadState also
// carries the loading progress percentage, which is what makes a stuck load
// diagnosable.
func (service *Service) awaitCollectionLoaded(ctx context.Context, collectionName string) error {
	probe := func(probeCtx context.Context) (collectionLoadState, error) {
		state, err := service.milvus.GetLoadState(probeCtx, milvusclient.NewGetLoadStateOption(collectionName))
		if err != nil {
			return collectionLoadState{Loaded: false, Progress: 0, ProgressKnown: false},
				wrapStoreError(ctx, err, "get load state for "+collectionName)
		}
		return collectionLoadState{
			Loaded:        state.State == entity.LoadStateLoaded,
			Progress:      state.Progress,
			ProgressKnown: state.State == entity.LoadStateLoading,
		}, nil
	}
	reissue := func(reissueCtx context.Context) error {
		if _, err := service.milvus.LoadCollection(reissueCtx, milvusclient.NewLoadCollectionOption(collectionName)); err != nil {
			return wrapStoreError(ctx, err, "re-issue load of Milvus collection "+collectionName)
		}
		return nil
	}
	return waitForCollectionLoad(ctx, collectionName, service.collectionLoadBound(), probe, reissue)
}
