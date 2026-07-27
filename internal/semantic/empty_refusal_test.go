package semantic

import (
	"context"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/model"
)

// emptyRefusingEmbedder refuses any input carrying nothing to embed and embeds
// the rest, which is what both real providers now do. It records every batch so
// a test can prove a refused input is never offered again.
type emptyRefusingEmbedder struct {
	batches [][]string
}

func (embedder *emptyRefusingEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return []float32{0}, nil
}

func (embedder *emptyRefusingEmbedder) EmbedBatch(
	_ context.Context,
	texts []string,
) (embedding.BatchResult, error) {
	embedder.batches = append(embedder.batches, append([]string(nil), texts...))
	vectors := make([][]float32, len(texts))
	var skipped []embedding.SkippedInput
	for index, text := range texts {
		if strings.TrimSpace(text) == "" {
			skipped = append(skipped, embedding.SkippedInput{
				Index:          index,
				Reason:         adapterr.EmbedRejectionEmptyContent,
				ReportedTokens: adapterr.UnreportedFigure(),
				MaxTokens:      adapterr.UnreportedFigure(),
			})
			continue
		}
		vectors[index] = []float32{float32(len(text))}
	}
	return embedding.BatchResult{Vectors: vectors, Skipped: skipped}, nil
}

func (embedder *emptyRefusingEmbedder) ProviderName() model.EmbeddingProvider {
	return "empty-refusing"
}

func (embedder *emptyRefusingEmbedder) Health(_ context.Context) error { return nil }

// TestInsertChunksBatchedNeverCountsAnEmptyRefusalAsADrop is the counting
// guarantee. ChunksDropped means content the index lost, and a chunk that
// carried nothing was never content, so a refusal must leave that count at zero.
// If a refusal reached the split-retry path it would be counted there as an
// unclassified drop, which is the failure this locks out: the drop counter would
// fill with inputs that were never content and stop being usable for finding the
// real losses.
func TestInsertChunksBatchedNeverCountsAnEmptyRefusalAsADrop(t *testing.T) {
	handler := captureDefaultLogger(t)
	embedder := &emptyRefusingEmbedder{}
	service := &Service{
		cfg: config.Config{
			EmbeddingBatchSize:        32,
			EmbeddingBatchTokenBudget: 6000,
		},
		embedder: embedder,
	}
	chunks := []model.StoredChunk{
		{Content: "", RelativePath: "conv/thread-1/4", ConversationID: "thread-1", MessageIndex: 4, Role: "assistant"},
		{Content: "   \n\t ", RelativePath: "conv/thread-1/5", ConversationID: "thread-1", MessageIndex: 5, Role: "assistant"},
	}
	var reports []Progress

	err := service.insertChunksBatched(
		context.Background(),
		"test_collection",
		chunks,
		true,
		"test",
		func(progress Progress) { reports = append(reports, progress) },
		nil,
		StoreColumnSetConversation,
	)
	if err != nil {
		t.Fatalf("insertChunksBatched returned error: %v", err)
	}

	if len(reports) != 1 {
		t.Fatalf("progress reports = %d, want 1", len(reports))
	}
	if reports[0].ChunksDropped != 0 {
		t.Fatalf("ChunksDropped = %d, want 0; an input that carried nothing is not lost content", reports[0].ChunksDropped)
	}
	if reports[0].CollectionRowsWritten != 0 {
		t.Fatalf("CollectionRowsWritten = %d, want 0; a refused input must not be stored", reports[0].CollectionRowsWritten)
	}

	// One batch only. A second would mean the refusal reached the splitter, which
	// could only divide nothing and offer the same empty input again.
	if len(embedder.batches) != 1 {
		t.Fatalf("embed calls = %d, want 1; a refusal must not be retried through the splitter", len(embedder.batches))
	}

	if _, found := handler.find("semantic.embed_input_dropped"); found {
		t.Fatal("a refusal was logged as a dropped input, which is the record of lost content")
	}
	if _, found := handler.find("semantic.embed_inputs_dropped_summary"); found {
		t.Fatal("a refusal reached the dropped-inputs summary")
	}

	refusal, found := handler.find("semantic.embed_inputs_refused_empty")
	if !found {
		t.Fatal("no semantic.embed_inputs_refused_empty record; a refusal must be visible, not silent")
	}
	if refusal.Attrs["refused_inputs"] != "2" {
		t.Fatalf("refused_inputs = %q, want \"2\"", refusal.Attrs["refused_inputs"])
	}
	// The identity is what makes the count actionable: it names who sent it.
	if refusal.Attrs["conversation_id"] != "thread-1" {
		t.Fatalf("conversation_id = %q, want thread-1 so the offending caller is nameable", refusal.Attrs["conversation_id"])
	}
	if refusal.Attrs["relative_path"] != "conv/thread-1/4" {
		t.Fatalf("relative_path = %q, want conv/thread-1/4", refusal.Attrs["relative_path"])
	}

	summary, found := handler.find("semantic.embed_inputs_refused_empty_summary")
	if !found {
		t.Fatal("no job-level refusal summary; a feeder sending these at volume must be visible per run")
	}
	if summary.Attrs["refused_inputs"] != "2" {
		t.Fatalf("summary refused_inputs = %q, want \"2\"", summary.Attrs["refused_inputs"])
	}
}

// TestInsertChunksBatchedStillWritesContentBesideARefusal proves the refusal is
// confined to the input it refused. A batch carrying one empty chunk and one
// real chunk must still embed and keep the real one, so a feeder defect never
// costs the content sitting next to it.
func TestInsertChunksBatchedStillWritesContentBesideARefusal(t *testing.T) {
	embedder := &emptyRefusingEmbedder{}
	service := &Service{
		cfg: config.Config{
			EmbeddingBatchSize:        32,
			EmbeddingBatchTokenBudget: 6000,
		},
		embedder: embedder,
	}
	chunks := []model.StoredChunk{
		{Content: "", RelativePath: "conv/thread-2/0", ConversationID: "thread-2"},
		{Content: "a real message worth indexing", RelativePath: "conv/thread-2/1", ConversationID: "thread-2"},
	}

	keptChunks, keptVectors := filterEmbeddedChunks(chunks, [][]float32{nil, {29}})
	if len(keptChunks) != 1 || keptChunks[0].RelativePath != "conv/thread-2/1" {
		t.Fatalf("kept %#v, want only the chunk that carried content", keptChunks)
	}
	if len(keptVectors) != 1 {
		t.Fatalf("kept %d vectors, want 1 index-aligned with the kept chunk", len(keptVectors))
	}

	embedded, err := service.embedChunkBatch(context.Background(), chunks, nil)
	if err != nil {
		t.Fatalf("embedChunkBatch returned error: %v", err)
	}
	if !embedded.refusedEmpty[0] {
		t.Fatal("the empty chunk was not marked refused")
	}
	if embedded.refusedEmpty[1] {
		t.Fatal("the chunk carrying content was marked refused")
	}
	if embedded.vectors[1] == nil {
		t.Fatal("the chunk carrying content got no vector")
	}
	if oversized := collectOversizedChunks(chunks, embedded); len(oversized) != 0 {
		t.Fatalf("collectOversizedChunks returned %#v, want none; a refusal must not enter the splitter", oversized)
	}
}
