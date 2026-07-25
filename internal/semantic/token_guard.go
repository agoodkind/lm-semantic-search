package semantic

import (
	"context"
	"log/slog"
	"unicode/utf8"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
)

// expandOverTokenBudget splits any chunk whose estimated token count exceeds the
// configured per-chunk cap into byte-budgeted sub-chunks, so no input reaches
// the embedder above its input limit and gets silently truncated. It is a no-op
// when EmbeddingMaxTokens is unset (cap 0). Unlike the varchar guardrail, a
// token split is expected for large content, so it logs at info level.
func (service *Service) expandOverTokenBudget(ctx context.Context, codebasePath string, chunks []model.StoredChunk, operation string) []model.StoredChunk {
	tokenCap := config.EffectiveEmbedTokenCap(service.cfg.EmbeddingMaxTokens)
	if tokenCap <= 0 {
		return chunks
	}
	overBudget := false
	for _, chunk := range chunks {
		if estimatedTokenCount(chunk.Content) > tokenCap {
			overBudget = true
			break
		}
	}
	if !overBudget {
		return chunks
	}
	byteBudget := config.EmbedChunkByteBudget(service.cfg.EmbeddingMaxTokens)
	out := make([]model.StoredChunk, 0, len(chunks))
	splitChunks := 0
	for _, chunk := range chunks {
		if estimatedTokenCount(chunk.Content) <= tokenCap {
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
	slog.InfoContext(ctx, "semantic.split_over_token_budget", "codebase_path", codebasePath, "operation", operation, "token_cap", tokenCap, "byte_budget", byteBudget, "chunks_split", splitChunks, "expanded_from", len(chunks), "expanded_to", len(out))
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
