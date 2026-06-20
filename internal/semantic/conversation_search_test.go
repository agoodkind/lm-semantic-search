package semantic

import (
	"context"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

func rankedChunk(conversationID string, score float64) model.StoredChunk {
	return model.StoredChunk{ConversationID: conversationID, Score: score}
}

// pagerOver returns a conversationPageSearch that serves ranked in score order,
// slicing it by offset and pageLimit, and counts the calls it receives.
func pagerOver(ranked []model.StoredChunk, calls *int) conversationPageSearch {
	return func(_ context.Context, offset int, pageLimit int) ([]model.StoredChunk, error) {
		*calls++
		if offset >= len(ranked) {
			return nil, nil
		}
		end := offset + pageLimit
		if end > len(ranked) {
			end = len(ranked)
		}
		return append([]model.StoredChunk(nil), ranked[offset:end]...), nil
	}
}

func conversationCounts(chunks []model.StoredChunk) map[string]int {
	counts := make(map[string]int)
	for _, chunk := range chunks {
		counts[chunk.ConversationID]++
	}
	return counts
}

// TestFillCappedConversationSearchFillsAcrossPages proves the worked example: a
// query whose top results are dominated by three conversations still returns the
// full limit under a per-conversation cap by paging until the cap is satisfied.
func TestFillCappedConversationSearchFillsAcrossPages(t *testing.T) {
	t.Parallel()

	order := []string{
		"A", "A", "B", "A", "B", "C", "A", "B", "A", "C",
		"B", "A", "C", "B", "A", "C", "B", "A", "C", "B",
		"D", "C", "B", "E", "C", "F", "G", "C", "H", "I",
	}
	ranked := make([]model.StoredChunk, len(order))
	for index, conversationID := range order {
		ranked[index] = rankedChunk(conversationID, float64(len(order)-index))
	}

	calls := 0
	survivors, err := fillCappedConversationSearchWith(context.Background(), 10, 2, 0, pagerOver(ranked, &calls))
	if err != nil {
		t.Fatalf("fill returned error: %v", err)
	}
	if len(survivors) != 10 {
		t.Fatalf("survivors = %d, want 10 (the cap must page to fill the limit)", len(survivors))
	}
	for conversationID, count := range conversationCounts(survivors) {
		if count > 2 {
			t.Fatalf("conversation %q kept %d hits, want at most 2", conversationID, count)
		}
	}
	if calls != 3 {
		t.Fatalf("page searches = %d, want 3 (page 10 each over a 30-row window)", calls)
	}
}

// TestFillCappedConversationSearchReturnsPartialOnExhaustion proves the fill
// returns the honest smaller count when the corpus cannot supply limit
// per-conversation-distinct matches, and stops at the short page.
func TestFillCappedConversationSearchReturnsPartialOnExhaustion(t *testing.T) {
	t.Parallel()

	ranked := []model.StoredChunk{
		rankedChunk("A", 0.9),
		rankedChunk("A", 0.8),
		rankedChunk("A", 0.7),
		rankedChunk("B", 0.6),
		rankedChunk("B", 0.5),
	}
	calls := 0
	survivors, err := fillCappedConversationSearchWith(context.Background(), 10, 2, 0, pagerOver(ranked, &calls))
	if err != nil {
		t.Fatalf("fill returned error: %v", err)
	}
	if len(survivors) != 4 {
		t.Fatalf("survivors = %d, want 4 (A and B capped at 2 each, no more rows)", len(survivors))
	}
	if calls != 1 {
		t.Fatalf("page searches = %d, want 1 (a short page ends the fill)", calls)
	}
}

// TestFillCappedConversationSearchMinScoreEarlyStop proves a page whose frontier
// falls below the score floor ends the fill, since later pages rank lower.
func TestFillCappedConversationSearchMinScoreEarlyStop(t *testing.T) {
	t.Parallel()

	ranked := []model.StoredChunk{
		rankedChunk("A", 0.9),
		rankedChunk("B", 0.8),
		rankedChunk("C", 0.7),
		rankedChunk("D", 0.6),
		rankedChunk("E", 0.5),
		rankedChunk("F", 0.45),
		rankedChunk("G", 0.4),
		rankedChunk("H", 0.35),
		rankedChunk("I", 0.3),
		rankedChunk("J", 0.25),
	}
	calls := 0
	survivors, err := fillCappedConversationSearchWith(context.Background(), 10, 0, 0.5, pagerOver(ranked, &calls))
	if err != nil {
		t.Fatalf("fill returned error: %v", err)
	}
	if len(survivors) != 5 {
		t.Fatalf("survivors = %d, want 5 (only scores >= 0.5 clear the floor)", len(survivors))
	}
	if calls != 1 {
		t.Fatalf("page searches = %d, want 1 (the floor frontier ends the fill)", calls)
	}
}

// TestFillCappedConversationSearchSinglePageWhenCapLoose proves the common case
// runs exactly one search when the cap does not bind.
func TestFillCappedConversationSearchSinglePageWhenCapLoose(t *testing.T) {
	t.Parallel()

	ranked := []model.StoredChunk{
		rankedChunk("A", 0.9),
		rankedChunk("B", 0.8),
		rankedChunk("C", 0.7),
		rankedChunk("D", 0.6),
		rankedChunk("E", 0.5),
	}
	calls := 0
	survivors, err := fillCappedConversationSearchWith(context.Background(), 5, 2, 0, pagerOver(ranked, &calls))
	if err != nil {
		t.Fatalf("fill returned error: %v", err)
	}
	if len(survivors) != 5 {
		t.Fatalf("survivors = %d, want 5", len(survivors))
	}
	if calls != 1 {
		t.Fatalf("page searches = %d, want 1 (one embed, one page in the common case)", calls)
	}
}

// TestCapConversationChunks proves the reduction keeps the highest-ranked hits
// per conversation, floors by minScore, and truncates to limit.
func TestCapConversationChunks(t *testing.T) {
	t.Parallel()

	chunks := []model.StoredChunk{
		rankedChunk("a", 0.9),
		rankedChunk("a", 0.8),
		rankedChunk("a", 0.7),
		rankedChunk("b", 0.6),
		rankedChunk("c", 0.2),
	}

	capped := capConversationChunks(chunks, 1, 0, 0)
	if len(capped) != 3 {
		t.Fatalf("per-conversation cap kept %d, want 3 (one per conversation)", len(capped))
	}
	if capped[0].ConversationID != "a" || capped[0].Score != 0.9 {
		t.Fatalf("cap kept %+v first, want a's top-ranked hit", capped[0])
	}

	floored := capConversationChunks(chunks, 0, 0.5, 0)
	if len(floored) != 4 {
		t.Fatalf("min-score floor kept %d, want 4 (0.2 dropped)", len(floored))
	}

	limited := capConversationChunks(chunks, 0, 0, 2)
	if len(limited) != 2 {
		t.Fatalf("limit kept %d, want 2", len(limited))
	}
}
