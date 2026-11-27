package scheduling

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/platformactivity"
)

type fakeActivity struct {
	available bool
	idle      atomic.Int64
	closed    bool
}

func (a *fakeActivity) Sample(context.Context) platformactivity.Snapshot {
	return platformactivity.Snapshot{
		InputAvailable: a.available,
		InputIdleFor:   time.Duration(a.idle.Load()),
	}
}

func (a *fakeActivity) Close() {
	a.closed = true
}

func TestQuietSchedulerAdmitsBatchesWhenIdle(t *testing.T) {
	synctest.Test(
		t,
		func(t *testing.T) {
			activity := &fakeActivity{available: true}
			scheduler := newQuiet(t.Context(), activity)
			defer scheduler.Close()
			first := startAcquire(t.Context(), scheduler)
			synctest.Wait()
			assertWaiting(t, scheduler, model.SchedulingReasonUserActive)

			activity.idle.Store(int64(model.DefaultIdleAfterSeconds) * int64(time.Second))
			awaitActivitySample()
			firstResult := <-first
			if firstResult.err != nil {
				t.Fatal(firstResult.err)
			}

			activity.idle.Store(0)
			awaitActivitySample()
			assertWaiting(t, scheduler, model.SchedulingReasonUnspecified)
			second := startAcquire(t.Context(), scheduler)
			synctest.Wait()
			assertWaiting(t, scheduler, model.SchedulingReasonUserActive)
			firstResult.release()
			synctest.Wait()
			assertWaiting(t, scheduler, model.SchedulingReasonUserActive)

			scheduler.Close()
			if result := <-second; !errors.Is(result.err, context.Canceled) {
				t.Fatalf("acquire after close returned %v, want context cancellation", result.err)
			}

			if !activity.closed {
				t.Fatal("close did not stop the activity source")
			}
		},
	)
}

func TestQuietSchedulerWaitsWhenActivityIsUnavailable(t *testing.T) {
	synctest.Test(
		t,
		func(t *testing.T) {
			activity := &fakeActivity{}
			scheduler := newQuiet(t.Context(), activity)
			defer scheduler.Close()
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			pending := startAcquire(ctx, scheduler)
			synctest.Wait()
			assertWaiting(t, scheduler, model.SchedulingReasonActivityUnavailable)

			cancel()
			if result := <-pending; !errors.Is(result.err, context.Canceled) {
				t.Fatalf("canceled acquisition returned %v, want context cancellation", result.err)
			}

			assertWaiting(t, scheduler, model.SchedulingReasonUnspecified)
		},
	)
}

type acquireResult struct {
	release func()
	err     error
}

func startAcquire(ctx context.Context, scheduler *Scheduler) <-chan acquireResult {
	result := make(chan acquireResult, 1)
	go func() {
		release, err := scheduler.Acquire(ctx)
		result <- acquireResult{release: release, err: err}
	}()

	return result
}

func assertWaiting(t *testing.T, scheduler *Scheduler, want model.SchedulingReason) {
	t.Helper()
	if got := scheduler.WaitingReason(); got != string(want) {
		t.Fatalf("waiting reason = %q, want %q", got, want)
	}
}

// The fake clock advances until both the sampler and this timer have fired.
func awaitActivitySample() {
	const settleAfterSample = 3 * time.Second
	<-time.After(settleAfterSample)
	synctest.Wait()
}
