package semantic

import (
	"context"
	"log/slog"
	"sync"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

// capturedRecord is one log record reduced to the parts a span assertion needs:
// the message and the attributes rendered as strings.
type capturedRecord struct {
	Message string
	Attrs   map[string]string
}

// capturingHandler collects every record the default logger emits, so a test can
// assert on the phase spans and their count lines without parsing a log file.
type capturingHandler struct {
	mu      sync.Mutex
	records []capturedRecord
}

func (handler *capturingHandler) Enabled(_ context.Context, _ slog.Level) bool { return true }

func (handler *capturingHandler) Handle(_ context.Context, record slog.Record) error {
	attrs := make(map[string]string, record.NumAttrs())
	record.Attrs(func(attr slog.Attr) bool {
		attrs[attr.Key] = attr.Value.String()
		return true
	})
	handler.mu.Lock()
	defer handler.mu.Unlock()
	handler.records = append(handler.records, capturedRecord{Message: record.Message, Attrs: attrs})
	return nil
}

func (handler *capturingHandler) WithAttrs(_ []slog.Attr) slog.Handler { return handler }

func (handler *capturingHandler) WithGroup(_ string) slog.Handler { return handler }

func (handler *capturingHandler) reset() {
	handler.mu.Lock()
	defer handler.mu.Unlock()
	handler.records = nil
}

// find returns the first record with the given message, and whether one exists.
func (handler *capturingHandler) find(message string) (capturedRecord, bool) {
	handler.mu.Lock()
	defer handler.mu.Unlock()
	for _, record := range handler.records {
		if record.Message == message {
			return record, true
		}
	}
	return capturedRecord{}, false
}

// findSpanCompletion returns the completion record for the named span.
func (handler *capturingHandler) findSpanCompletion(name string) (capturedRecord, bool) {
	handler.mu.Lock()
	defer handler.mu.Unlock()
	for _, record := range handler.records {
		if record.Message == "daemon.span.completed" && record.Attrs["span"] == name {
			return record, true
		}
	}
	return capturedRecord{}, false
}

// captureDefaultLogger routes the default logger into a capturing handler for
// the duration of one test.
func captureDefaultLogger(t *testing.T) *capturingHandler {
	t.Helper()
	handler := &capturingHandler{}
	previous := slog.Default()
	slog.SetDefault(slog.New(handler))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return handler
}

// TestEmbedPhaseSpanReportsInputCountsForRateAttribution pins the embedding
// phase boundary: the embedder round trip carries its own span with the input
// and estimated-token counts beside it, so a reindex's embedding rate is
// readable from the logs alone.
func TestEmbedPhaseSpanReportsInputCountsForRateAttribution(t *testing.T) {
	handler := captureDefaultLogger(t)
	service := &Service{embedder: &countingEmbedder{}}

	chunks := []model.StoredChunk{{Content: "reused-A"}, {Content: "fresh"}}
	reuse := map[string][]float32{contentVectorKey("reused-A"): {1}}

	if _, err := service.embedChunkBatch(context.Background(), chunks, reuse); err != nil {
		t.Fatalf("embedChunkBatch returned error: %v", err)
	}

	started, found := handler.find("semantic.embed_batch_started")
	if !found {
		t.Fatal("no semantic.embed_batch_started record; the embedding phase is untimed")
	}
	if started.Attrs["inputs"] != "1" {
		t.Fatalf("inputs = %q, want \"1\" (only the miss is sent)", started.Attrs["inputs"])
	}
	if started.Attrs["reused"] != "1" {
		t.Fatalf("reused = %q, want \"1\"", started.Attrs["reused"])
	}
	// "fresh" is five bytes, which the packer's estimate rounds up to two tokens.
	if started.Attrs["estimated_tokens"] != "2" {
		t.Fatalf("estimated_tokens = %q, want \"2\"", started.Attrs["estimated_tokens"])
	}

	completed, found := handler.findSpanCompletion("semantic.embedBatch")
	if !found {
		t.Fatal("no completed semantic.embedBatch span; embedding duration is not attributable")
	}
	if _, hasDuration := completed.Attrs["duration_ms"]; !hasDuration {
		t.Fatalf("semantic.embedBatch completion has no duration_ms: %v", completed.Attrs)
	}
}

// TestEmbedPhaseSpanIsSilentForAnAllReuseBatch keeps the log volume
// proportionate. A batch served entirely from the reuse map never calls the
// embedder, so it must not open an embedding span that would report a phase
// that did not run.
func TestEmbedPhaseSpanIsSilentForAnAllReuseBatch(t *testing.T) {
	handler := captureDefaultLogger(t)
	service := &Service{embedder: &countingEmbedder{}}

	chunks := []model.StoredChunk{{Content: "reused-A"}}
	reuse := map[string][]float32{contentVectorKey("reused-A"): {1}}

	if _, err := service.embedChunkBatch(context.Background(), chunks, reuse); err != nil {
		t.Fatalf("embedChunkBatch returned error: %v", err)
	}

	if _, found := handler.find("semantic.embed_batch_started"); found {
		t.Fatal("an all-reuse batch opened an embedding span; no embedder call happened")
	}
	if _, found := handler.findSpanCompletion("semantic.embedBatch"); found {
		t.Fatal("an all-reuse batch completed a semantic.embedBatch span")
	}
}

// TestEmbedPhaseSpanCountsEveryInputWhenNothingIsReused covers the cold path,
// where the estimated-token count is the sum across the whole request.
func TestEmbedPhaseSpanCountsEveryInputWhenNothingIsReused(t *testing.T) {
	handler := captureDefaultLogger(t)
	handler.reset()
	service := &Service{embedder: &countingEmbedder{}}

	chunks := []model.StoredChunk{chunkOfBytes(400), chunkOfBytes(400)}

	if _, err := service.embedChunkBatch(context.Background(), chunks, nil); err != nil {
		t.Fatalf("embedChunkBatch returned error: %v", err)
	}

	started, found := handler.find("semantic.embed_batch_started")
	if !found {
		t.Fatal("no semantic.embed_batch_started record for a cold batch")
	}
	if started.Attrs["inputs"] != "2" {
		t.Fatalf("inputs = %q, want \"2\"", started.Attrs["inputs"])
	}
	// Two 400-byte chunks estimate at 100 tokens each.
	if started.Attrs["estimated_tokens"] != "200" {
		t.Fatalf("estimated_tokens = %q, want \"200\"", started.Attrs["estimated_tokens"])
	}
	if started.Attrs["reused"] != "0" {
		t.Fatalf("reused = %q, want \"0\"", started.Attrs["reused"])
	}
}
