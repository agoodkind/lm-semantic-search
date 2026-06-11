package semantic

import "goodkind.io/lm-semantic-search/internal/model"

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
func packChunksByEstimatedTokens(
	chunks []model.StoredChunk,
	maxRows int,
	tokenBudget int,
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
		tokens := estimatedTokenCount(chunk.Content)
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
