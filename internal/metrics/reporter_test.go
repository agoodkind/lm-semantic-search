package metrics

import (
	"context"
	"log/slog"
	"sync"
	"testing"
	"time"
)

// captureHandler is a minimal [slog.Handler] that records every emitted
// record so reporter tests can assert on message and attribute keys
// without depending on output formatting.
type captureHandler struct {
	mu      sync.Mutex
	records []slog.Record
}

func (h *captureHandler) Enabled(context.Context, slog.Level) bool {
	return true
}

func (h *captureHandler) Handle(_ context.Context, record slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.records = append(h.records, record.Clone())
	return nil
}

func (h *captureHandler) WithAttrs([]slog.Attr) slog.Handler {
	return h
}

func (h *captureHandler) WithGroup(string) slog.Handler {
	return h
}

func (h *captureHandler) countMessage(message string) int {
	h.mu.Lock()
	defer h.mu.Unlock()
	count := 0
	for _, record := range h.records {
		if record.Message == message {
			count++
		}
	}
	return count
}

func (h *captureHandler) firstMessage(message string) (slog.Record, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, record := range h.records {
		if record.Message == message {
			return record, true
		}
	}
	return slog.Record{}, false
}

// withCaptureLogger swaps the default slog logger for one backed by a
// captureHandler and restores the original when the test ends.
func withCaptureLogger(t *testing.T) *captureHandler {
	t.Helper()
	handler := &captureHandler{}
	previous := slog.Default()
	slog.SetDefault(slog.New(handler))
	t.Cleanup(func() {
		slog.SetDefault(previous)
	})
	return handler
}

func TestStartReporterNonPositiveIntervalEmitsNothing(t *testing.T) {
	handler := withCaptureLogger(t)

	StartReporter(context.Background(), 0)
	StartReporter(context.Background(), -time.Second)

	time.Sleep(50 * time.Millisecond)
	if got := handler.countMessage(reportMessage); got != 0 {
		t.Fatalf("non-positive interval emitted %d %q records, want 0", got, reportMessage)
	}
}

func TestStartReporterEmitsRecordWithExpectedKeys(t *testing.T) {
	handler := withCaptureLogger(t)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	StartReporter(ctx, 5*time.Millisecond)

	deadline := time.After(2 * time.Second)
	for handler.countMessage(reportMessage) == 0 {
		select {
		case <-deadline:
			t.Fatal("reporter did not emit a record before the deadline")
		default:
			time.Sleep(2 * time.Millisecond)
		}
	}

	record, ok := handler.firstMessage(reportMessage)
	if !ok {
		t.Fatal("expected at least one perf_counters record")
	}

	present := map[string]bool{}
	record.Attrs(func(attr slog.Attr) bool {
		present[attr.Key] = true
		return true
	})

	wantKeys := []string{
		"embed_batches_total",
		"embed_batches_failed",
		"embed_vectors_total",
		"embed_latency_ms_sum",
		"embed_inflight",
		"converge_upsert_total",
		"converge_remove_total",
		"converge_copy_chunks_total",
		"sweep_runs_total",
		"sweep_changed_total",
		"sync_skipped_inflight_total",
		"jobs_completed_total",
		"jobs_failed_total",
		"jobs_cancelled_total",
		"boot_resumes_total",
		"jobs_active",
		"num_goroutine",
		"heap_alloc_bytes",
		"heap_inuse_bytes",
		"num_gc",
	}
	for _, key := range wantKeys {
		if !present[key] {
			t.Errorf("perf_counters record missing field %q", key)
		}
	}
}

func TestStartReporterStopsOnContextCancel(t *testing.T) {
	handler := withCaptureLogger(t)

	ctx, cancel := context.WithCancel(context.Background())
	StartReporter(ctx, 5*time.Millisecond)

	deadline := time.After(2 * time.Second)
	for handler.countMessage(reportMessage) == 0 {
		select {
		case <-deadline:
			t.Fatal("reporter did not emit before cancel")
		default:
			time.Sleep(2 * time.Millisecond)
		}
	}

	cancel()
	// Allow the goroutine to observe cancellation and exit its loop.
	time.Sleep(50 * time.Millisecond)
	stableCount := handler.countMessage(reportMessage)
	time.Sleep(60 * time.Millisecond)
	if got := handler.countMessage(reportMessage); got != stableCount {
		t.Fatalf("reporter kept emitting after cancel: %d -> %d", stableCount, got)
	}
}
