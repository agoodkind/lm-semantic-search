package semantic

import (
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
)

const (
	boolBytes                         = 1
	float32Bytes                      = 4
	int64Bytes                        = 8
	insertRowProtobufFramingAllowance = 40
	insertBatchEstimatedByteBudget    = 192 << 20
)

// estimatedTokenCount approximates the embedding server's tokenizer count at
// four bytes per token with a floor of one. The server enforces exact counts;
// this estimate only shapes request granularity.
func estimatedTokenCount(content string) int {
	count := (len(content) + 3) / 4
	if count < 1 {
		return 1
	}
	return count
}

// packChunksByEstimatedTokens groups consecutive chunks for one embedding
// request. A group closes when adding the next chunk would push the estimated
// token sum past tokenBudget or the row count past maxRows. A single chunk
// above the budget ships alone. Order is preserved and every chunk lands in
// exactly one group.
//
// Only the chunks that will actually reach the embedder are charged against
// tokenBudget. A chunk whose content already has a vector in reuse is stripped
// by embedChunkBatch before the request is built, so counting it would close
// the group over tokens the request never carries and leave the row ceiling
// unreachable: a re-ingest where most content is unchanged would then pay a
// full round trip to embed a handful of inputs. Charging only the misses keeps
// every request inside the same token budget while letting the group grow to
// maxRows, so the request that is sent is the one the budget describes.
func packChunksByEstimatedTokens(
	chunks []model.StoredChunk,
	maxRows int,
	tokenBudget int,
	reuse map[string][]float32,
) [][]model.StoredChunk {
	if maxRows < 1 {
		maxRows = 1
	}
	if tokenBudget < 1 {
		tokenBudget = 1
	}
	groups := make([][]model.StoredChunk, 0)
	current := make([]model.StoredChunk, 0, maxRows)
	currentTokens := 0
	for _, chunk := range chunks {
		tokens := embeddedTokenCount(chunk, reuse)
		overBudget := currentTokens+tokens > tokenBudget
		overRows := len(current) >= maxRows
		if len(current) > 0 && (overBudget || overRows) {
			groups = append(groups, current)
			current = make([]model.StoredChunk, 0, maxRows)
			currentTokens = 0
		}
		current = append(current, chunk)
		currentTokens += tokens
	}
	if len(current) > 0 {
		groups = append(groups, current)
	}
	return groups
}

// embeddedTokenCount returns the estimated tokens one chunk contributes to an
// embedding request. A chunk whose vector is already in reuse contributes zero,
// because it is served from the map and never sent. The lookup mirrors
// embedChunkBatch's, so the packer's view of a request matches what is built.
func embeddedTokenCount(chunk model.StoredChunk, reuse map[string][]float32) int {
	if len(reuse) > 0 {
		if _, hit := reuse[contentVectorKey(chunk.Content)]; hit {
			return 0
		}
	}
	return estimatedTokenCount(chunk.Content)
}

// packChunksByEstimatedInsertBytes groups consecutive chunks for one store
// insert. Every row is charged because reused vectors still cross the store
// transport. The estimate includes every base column and the optional
// conversation scalar columns, using the values insertBatch sends after its
// string transformations.
func packChunksByEstimatedInsertBytes(
	chunks []model.StoredChunk,
	vectorDimension int,
	byteBudget int,
	columnSet StoreColumnSet,
) [][]model.StoredChunk {
	if vectorDimension < 0 {
		vectorDimension = 0
	}
	if byteBudget < 1 {
		byteBudget = 1
	}

	groups := make([][]model.StoredChunk, 0)
	current := make([]model.StoredChunk, 0)
	currentBytes := 0
	for _, chunk := range chunks {
		rowBytes := estimatedInsertRowBytes(chunk, vectorDimension, columnSet)
		overBudget := currentBytes+rowBytes > byteBudget
		if len(current) > 0 && overBudget {
			groups = append(groups, current)
			current = make([]model.StoredChunk, 0)
			currentBytes = 0
		}
		current = append(current, chunk)
		currentBytes += rowBytes
	}
	if len(current) > 0 {
		groups = append(groups, current)
	}
	return groups
}

// estimatedInsertRowBytes charges the raw values in every column insertBatch
// sends. The framing allowance covers the row-scaling protobuf overhead that is
// not part of those values. Ten schema-valid string columns need at most 28
// bytes for their repeated-field tags and length varints. Four int64 columns
// can each need two bytes beyond their eight-byte raw width when encoded as
// varints. Rounding that 36-byte maximum to 40 leaves four bytes per row for
// implementation framing. Per-column and per-request wrappers do not scale
// with rows and fit inside the 64 MiB gap to the transport limit.
func estimatedInsertRowBytes(
	chunk model.StoredChunk,
	vectorDimension int,
	columnSet StoreColumnSet,
) int {
	content, _ := sanitizeUTF8(chunk.Content)
	relativePath, _ := sanitizeUTF8(chunk.RelativePath)
	fileExtension, _ := sanitizeUTF8(chunk.FileExtension)
	metadataValue, _ := sanitizeUTF8(encodeMetadata(chunk))

	rowBytes := len(generateID(chunk, 0)) +
		len(content) +
		len(relativePath) +
		int64Bytes +
		int64Bytes +
		len(fileExtension) +
		len(metadataValue) +
		vectorDimension*float32Bytes +
		insertRowProtobufFramingAllowance
	if columnSet.ConversationScalars() {
		rowBytes += len(chunk.ConversationID) +
			len(chunk.ParentConversationID) +
			len(strings.ToLower(chunk.Role)) +
			len(providerFromConversationID(chunk.ConversationID)) +
			len(chunk.WorkspaceRoot) +
			boolBytes +
			int64Bytes +
			int64Bytes
	}
	return rowBytes
}
