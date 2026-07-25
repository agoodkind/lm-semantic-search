package semantic

import (
	"context"
	"log/slog"
	"unicode/utf8"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
)

// expandOverTokenBudget splits any chunk longer than the embedding byte budget
// into byte-budgeted sub-chunks, so no input reaches the embedder above the
// model's input limit and gets dropped as context_length_exceeded. The budget is
// derived from the model's hard input-token limit at a conservative
// bytes-per-token ratio, so it is always positive and the split runs even when
// EmbeddingMaxTokens is unset. It is the single backstop covering every embed
// path, including the conversation tool payloads that split by syntax rather than
// by the byte budget. Unlike the varchar guardrail, a token split is expected for
// large content, so it logs at info level.
func (service *Service) expandOverTokenBudget(ctx context.Context, codebasePath string, chunks []model.StoredChunk, operation string) []model.StoredChunk {
	byteBudget := config.EmbedChunkByteBudget(service.cfg.EmbeddingMaxTokens)
	if byteBudget <= 0 {
		return chunks
	}
	overBudget := false
	for _, chunk := range chunks {
		if len(chunk.Content) > byteBudget {
			overBudget = true
			break
		}
	}
	if !overBudget {
		return chunks
	}
	out := make([]model.StoredChunk, 0, len(chunks))
	splitChunks := 0
	for _, chunk := range chunks {
		if len(chunk.Content) <= byteBudget {
			out = append(out, chunk)
			continue
		}
		splitChunks++
		for _, piece := range splitBytes(chunk.Content, byteBudget) {
			child := chunk
			child.Content = piece
			out = append(out, child)
		}
	}
	slog.InfoContext(ctx, "semantic.split_over_token_budget", "codebase_path", codebasePath, "operation", operation, "byte_budget", byteBudget, "chunks_split", splitChunks, "expanded_from", len(chunks), "expanded_to", len(out))
	return out
}

// splitBytes cuts value into sub-strings of at most maxBytes bytes, each ending
// on a UTF-8 codepoint boundary. A non-positive maxBytes returns value unsplit.
func splitBytes(value string, maxBytes int) []string {
	if maxBytes <= 0 {
		return []string{value}
	}
	out := make([]string, 0, (len(value)+maxBytes-1)/maxBytes)
	start := 0
	for start < len(value) {
		end := start + maxBytes
		if end >= len(value) {
			out = append(out, value[start:])
			break
		}
		for end > start && !utf8.RuneStart(value[end]) {
			end--
		}
		if end == start {
			_, size := utf8.DecodeRuneInString(value[start:])
			end = start + size
		}
		out = append(out, value[start:end])
		start = end
	}
	return out
}
