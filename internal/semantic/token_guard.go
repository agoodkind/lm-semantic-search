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
// model's input limit and gets dropped as context_length_exceeded or silently
// truncated. The budget is derived from the active provider's hard input-token
// limit at a conservative bytes-per-token ratio, so it is always positive and the
// split runs even when EmbeddingMaxTokens is unset. It is the single backstop
// covering every Milvus embed path, including the conversation tool payloads that
// split by syntax rather than by the byte budget. Unlike the varchar guardrail, a
// token split is expected for large content, so it logs at info level.
func (service *Service) expandOverTokenBudget(ctx context.Context, codebasePath string, chunks []model.StoredChunk, operation string) []model.StoredChunk {
	byteBudget := config.EmbedChunkByteBudgetForLimit(service.cfg.EmbeddingMaxTokens, config.ActiveEmbedTokenLimit(service.cfg))
	out, splitCount := SplitChunksToByteBudget(chunks, byteBudget)
	if splitCount == 0 {
		return chunks
	}
	slog.InfoContext(ctx, "semantic.split_over_token_budget", "codebase_path", codebasePath, "operation", operation, "byte_budget", byteBudget, "chunks_split", splitCount, "expanded_from", len(chunks), "expanded_to", len(out))
	return out
}

// SplitChunksToByteBudget splits any chunk whose content exceeds byteBudget into
// byte-budgeted children that each end on a UTF-8 codepoint boundary and carry a
// distinct SplitPart, so no input reaches the embedder above the model's input
// limit and identical pieces never collide on the primary key. It is the shared
// pre-embed split used by both the Milvus and local backends, parameterized by
// the caller's byte budget: the OpenAI-compatible model limit for Milvus and the
// offline preset limit for the local ONNX backend. A non-positive budget returns
// chunks unchanged. The returned count reports how many input chunks were split.
func SplitChunksToByteBudget(chunks []model.StoredChunk, byteBudget int) ([]model.StoredChunk, int) {
	if byteBudget <= 0 {
		return chunks, 0
	}
	overBudget := false
	for index := range chunks {
		if len(chunks[index].Content) > byteBudget {
			overBudget = true
			break
		}
	}
	if !overBudget {
		return chunks, 0
	}
	out := make([]model.StoredChunk, 0, len(chunks))
	splitCount := 0
	for index := range chunks {
		if len(chunks[index].Content) <= byteBudget {
			out = append(out, chunks[index])
			continue
		}
		splitCount++
		out = append(out, splitChunkAtBudget(chunks[index], byteBudget)...)
	}
	return out, splitCount
}

// splitChunkAtBudget splits chunk.Content into sub-chunks of at most budget bytes,
// each ending on a UTF-8 codepoint boundary, assigning every child a SplitPart
// that encodes its byte offset within the original parent content plus one, so
// identical pieces of repeated content never share a primary key. A child of an
// already-split chunk keeps offsets relative to the original parent by adding the
// parent child's base offset, so nested re-splits during retry stay unique. The
// parent relativePath and line range are preserved on every child.
func splitChunkAtBudget(chunk model.StoredChunk, budget int) []model.StoredChunk {
	baseOffset := 0
	if chunk.SplitPart > 0 {
		baseOffset = int(chunk.SplitPart) - 1
	}
	pieces, offsets := splitBytesWithOffsets(chunk.Content, budget)
	out := make([]model.StoredChunk, 0, len(pieces))
	for index := range pieces {
		child := chunk
		child.Content = pieces[index]
		child.SplitPart = safeInt32FromInt(baseOffset + offsets[index] + 1)
		out = append(out, child)
	}
	return out
}

// splitBytesWithOffsets cuts value into sub-strings of at most maxBytes bytes,
// each ending on a UTF-8 codepoint boundary, and reports each piece's start byte
// offset within value. A non-positive maxBytes returns value unsplit at offset
// zero.
func splitBytesWithOffsets(value string, maxBytes int) ([]string, []int) {
	if maxBytes <= 0 {
		return []string{value}, []int{0}
	}
	pieceCount := (len(value) + maxBytes - 1) / maxBytes
	pieces := make([]string, 0, pieceCount)
	offsets := make([]int, 0, pieceCount)
	start := 0
	for start < len(value) {
		end := start + maxBytes
		if end >= len(value) {
			pieces = append(pieces, value[start:])
			offsets = append(offsets, start)
			break
		}
		for end > start && !utf8.RuneStart(value[end]) {
			end--
		}
		if end == start {
			_, size := utf8.DecodeRuneInString(value[start:])
			end = start + size
		}
		pieces = append(pieces, value[start:end])
		offsets = append(offsets, start)
		start = end
	}
	return pieces, offsets
}

// splitBytes cuts value into sub-strings of at most maxBytes bytes, each ending
// on a UTF-8 codepoint boundary. A non-positive maxBytes returns value unsplit.
func splitBytes(value string, maxBytes int) []string {
	pieces, _ := splitBytesWithOffsets(value, maxBytes)
	return pieces
}
