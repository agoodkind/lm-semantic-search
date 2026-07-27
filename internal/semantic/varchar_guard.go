package semantic

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"unicode/utf8"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/spans"
)

// sanitizeUTF8 returns a copy of value with invalid UTF-8 byte sequences
// replaced by the Unicode replacement character. Milvus rejects VarChar
// payloads with invalid UTF-8 at the gRPC marshal boundary, so any chunk
// content that survives the file-level skip but still slices through a
// multi-byte codepoint (for example from a tree-sitter byte-offset
// boundary) gets repaired here. The second return value reports whether
// the input needed repair so callers can log the event.
func sanitizeUTF8(value string) (string, bool) {
	if utf8.ValidString(value) {
		return value, false
	}
	return strings.ToValidUTF8(value, "�"), true
}

// milvusVarcharMaxBytes mirrors the schema's WithMaxLength(65535) for VarChar
// fields. Chunks longer than this fail the Milvus insert with "length of
// varchar field content exceeds max length". The splitter is supposed to
// keep every chunk under chunk_size (default 2500), so an oversize chunk
// at insert time signals a splitter regression. The expansion in
// expandOversizeChunks turns the splitter regression into multiple rows
// rather than a dropped insert, so no content is lost.
const milvusVarcharMaxBytes = 65000

// guardrailExpand wraps expandOversizeChunks with logging. Each oversize
// chunk hitting this path signals an upstream splitter regression that
// emitted content longer than chunk_size. The log carries the codebase
// path, the operation that requested the embed, and the relative path of
// the offending file so the regression can be diagnosed without losing
// the data.
func (service *Service) guardrailExpand(ctx context.Context, codebasePath string, chunks []model.StoredChunk, operation string) []model.StoredChunk {
	ctx, done := spans.Open(ctx, "semantic.guardrailExpand")
	defer done(nil)

	// Split chunks over the embedding token budget first, so no input reaches the
	// embedder above its input limit and gets silently truncated. This runs
	// before the varchar guardrail because the token budget is the tighter cap.
	chunks = service.expandOverTokenBudget(ctx, codebasePath, chunks, operation)

	expanded, changed := expandOversizeChunks(chunks)
	if !changed {
		return chunks
	}
	offenders := make([]string, 0)
	for _, chunk := range chunks {
		if len(chunk.Content) > milvusVarcharMaxBytes {
			offenders = append(offenders, fmt.Sprintf("%s:%d-%d (%d bytes)", chunk.RelativePath, chunk.StartLine, chunk.EndLine, len(chunk.Content)))
		}
	}
	slog.WarnContext(ctx, "semantic.expanded_oversize_chunks", "codebase_path", codebasePath, "operation", operation, "expanded_from", len(chunks), "expanded_to", len(expanded), "max_bytes", milvusVarcharMaxBytes, "offenders", offenders)
	return expanded
}

// expandOversizeChunks returns a list where any chunk over
// milvusVarcharMaxBytes has been split into multiple chunks aligned to
// codepoint boundaries. The boolean reports whether any expansion
// happened so the caller can log the upstream regression.
func expandOversizeChunks(chunks []model.StoredChunk) ([]model.StoredChunk, bool) {
	expanded := false
	for _, chunk := range chunks {
		if len(chunk.Content) > milvusVarcharMaxBytes {
			expanded = true
			break
		}
	}
	if !expanded {
		return chunks, false
	}
	out := make([]model.StoredChunk, 0, len(chunks))
	for _, chunk := range chunks {
		if len(chunk.Content) <= milvusVarcharMaxBytes {
			out = append(out, chunk)
			continue
		}
		out = append(out, splitChunkAtBudget(chunk, milvusVarcharMaxBytes)...)
	}
	return out, true
}
