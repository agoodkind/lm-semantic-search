package updateopts

import (
	"context"
	"io"
	"log/slog"
	"testing"
	"time"
)

func TestRunApplySchedulerRecoversPanicPerIteration(t *testing.T) {
	originalDelay := nextUpdateDelayFunc
	originalApply := runScheduledApplyFunc
	t.Cleanup(func() {
		nextUpdateDelayFunc = originalDelay
		runScheduledApplyFunc = originalApply
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	callCount := 0
	nextUpdateDelayFunc = func(overrides Overrides) time.Duration {
		_ = overrides
		return time.Millisecond
	}
	runScheduledApplyFunc = func(ctx context.Context, overrides Overrides, log *slog.Logger) (bool, error) {
		_ = ctx
		_ = overrides
		_ = log
		callCount++
		if callCount == 1 {
			panic("scheduled apply panic")
		}
		cancel()
		return false, nil
	}

	done := make(chan struct{})
	go func() {
		logger := slog.New(slog.NewTextHandler(io.Discard, nil))
		RunApplyScheduler(ctx, Overrides{Log: logger}, nil)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatalf("RunApplyScheduler did not return after context cancellation")
	}
	if callCount < 2 {
		t.Fatalf("scheduled apply call count = %d, want scheduler to continue after panic", callCount)
	}
}
