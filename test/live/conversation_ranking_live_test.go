//go:build live

package live

import (
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

const (
	rankingQuery             = "needle"
	rankingDenseConversation = "claude:ranking-dense"
	rankingDenseMessages     = 30
	rankingOtherCount        = 20
	rankingLimit             = 10
	rankingCap               = 2
	rankingFullLimit         = 1000
	rankingLargeScopeSize    = 20000
	rankingRepeatCount       = 3
	rankingVisibilityTimeout = 60 * time.Second
	rankingVisibilityPoll    = 500 * time.Millisecond
	rankingStoredRows        = rankingDenseMessages + rankingOtherCount
)

func rankingOtherConversationID(index int) string {
	return fmt.Sprintf("claude:ranking-other-%02d", index)
}

// rankingCorpus has one dense conversation with thirty messages equal to the
// query and twenty other one-message conversations. The dense rows take the
// first ranks.
func rankingCorpus() map[string][]*pb.ConversationDocument {
	conversations := map[string][]*pb.ConversationDocument{}
	for messageIndex := range rankingDenseMessages {
		conversations[rankingDenseConversation] = append(conversations[rankingDenseConversation], &pb.ConversationDocument{
			ConversationId: rankingDenseConversation,
			MessageIndex:   int32(messageIndex),
			Role:           "user",
			TimestampUnix:  int64(1_600_000_000 + messageIndex),
			Text:           rankingQuery,
		})
	}
	for index := range rankingOtherCount {
		conversationID := rankingOtherConversationID(index)
		conversations[conversationID] = []*pb.ConversationDocument{{
			ConversationId: conversationID,
			MessageIndex:   0,
			Role:           "assistant",
			TimestampUnix:  int64(1_700_000_000 + index),
			Text:           fmt.Sprintf("other conversation %02d covers topic %d", index, index%4),
		}}
	}
	return conversations
}

func (h *harness) rankingSearch(limit int32, perConversationLimit int32, conversationIDs []string) []*pb.ConversationSearchResult {
	h.t.Helper()
	response, err := h.client.SearchConversations(correlatedContext(), &pb.SearchConversationsRequest{
		CollectionId:         h.collectionID,
		Query:                rankingQuery,
		Limit:                limit,
		PerConversationLimit: perConversationLimit,
		Filter:               &pb.ConversationSearchFilter{ConversationIds: conversationIDs},
	})
	if err != nil {
		h.t.Fatalf("SearchConversations returned error: %v", err)
	}
	return response.GetResults()
}

// newRankingHarness ingests the ranking corpus and waits until every row is
// searchable, then returns the harness and the full uncapped ranking.
func newRankingHarness(t *testing.T) (*harness, []*pb.ConversationSearchResult) {
	t.Helper()
	h := newHarness(t)
	ingested := h.upsert(rankingCorpus(), pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, false, false)
	requireCompleted(t, ingested, "ranking corpus ingest")
	deadline := time.Now().Add(rankingVisibilityTimeout)
	full := h.rankingSearch(rankingFullLimit, 0, nil)
	for len(full) < rankingStoredRows && time.Now().Before(deadline) {
		time.Sleep(rankingVisibilityPoll)
		full = h.rankingSearch(rankingFullLimit, 0, nil)
	}
	if len(full) != rankingStoredRows {
		t.Fatalf("full ranking has %d rows, want %d", len(full), rankingStoredRows)
	}
	return h, full
}

func rankingKey(result *pb.ConversationSearchResult) string {
	return fmt.Sprintf("conv/%s/%d", result.GetConversationId(), result.GetMessageIndex())
}

func rankingKeys(results []*pb.ConversationSearchResult) []string {
	keys := make([]string, 0, len(results))
	for _, result := range results {
		keys = append(keys, rankingKey(result))
	}
	return keys
}

// capRanking applies a per-conversation cap and a limit to a full ranking.
func capRanking(full []*pb.ConversationSearchResult, perConversationLimit int, limit int) []string {
	kept := make([]string, 0, limit)
	perConversation := map[string]int{}
	for _, result := range full {
		if perConversation[result.GetConversationId()] >= perConversationLimit {
			continue
		}
		perConversation[result.GetConversationId()]++
		kept = append(kept, rankingKey(result))
		if len(kept) == limit {
			break
		}
	}
	return kept
}

// TestConversationSearchCapFillsPastAnOverfilledTop proves the
// per-conversation cap fills the limit when the top ranks belong to one
// conversation. Thirty dense rows equal to the query rank first, and a limit
// of ten with a cap of two returns ten rows: two dense rows and the eight best
// other rows of the full ranking, in ranking order.
func TestConversationSearchCapFillsPastAnOverfilledTop(t *testing.T) {
	h, full := newRankingHarness(t)
	denseInTop := 0
	for _, result := range full[:rankingLimit] {
		if result.GetConversationId() == rankingDenseConversation {
			denseInTop++
		}
	}
	if denseInTop <= rankingCap {
		t.Fatalf("top %d ranks have %d dense rows, want more than the cap %d", rankingLimit, denseInTop, rankingCap)
	}
	capped := h.rankingSearch(rankingLimit, rankingCap, nil)
	if got, want := rankingKeys(capped), capRanking(full, rankingCap, rankingLimit); !slices.Equal(got, want) {
		t.Fatalf("capped rows = %v, want %v", got, want)
	}
}

// TestConversationSearchRankingIsStable proves one query returns the same rows
// in the same order across repeated calls, and a smaller limit returns a
// prefix of a larger limit under the same cap.
func TestConversationSearchRankingIsStable(t *testing.T) {
	h, full := newRankingHarness(t)
	for range rankingRepeatCount {
		if again := rankingKeys(h.rankingSearch(rankingFullLimit, 0, nil)); !slices.Equal(again, rankingKeys(full)) {
			t.Fatalf("repeated full ranking = %v, want %v", again, rankingKeys(full))
		}
	}
	larger := rankingKeys(h.rankingSearch(20, rankingCap, nil))
	for _, limit := range []int32{1, 3, 5, 10, 15} {
		smaller := rankingKeys(h.rankingSearch(limit, rankingCap, nil))
		if len(smaller) > len(larger) || !slices.Equal(smaller, larger[:len(smaller)]) {
			t.Fatalf("limit %d rows %v are not a prefix of limit 20 rows %v", limit, smaller, larger)
		}
	}
	uncappedPrefix := rankingKeys(h.rankingSearch(rankingLimit, 0, nil))
	if !slices.Equal(uncappedPrefix, rankingKeys(full)[:rankingLimit]) {
		t.Fatalf("uncapped limit %d rows %v are not a prefix of the full ranking", rankingLimit, uncappedPrefix)
	}
}

// TestConversationSearchLargeScopeRunsOneSearch proves a conversation id scope
// of twenty thousand ids runs as one ranking search and returns exactly the
// scoped rows in the full ranking's order.
func TestConversationSearchLargeScopeRunsOneSearch(t *testing.T) {
	h, full := newRankingHarness(t)
	scope := make([]string, 0, rankingLargeScopeSize)
	for index := range rankingLargeScopeSize {
		scope = append(scope, fmt.Sprintf("claude:ranking-missing-%05d", index))
	}
	scoped := map[string]bool{}
	for index := 0; index < rankingOtherCount; index += 2 {
		position := index * (rankingLargeScopeSize / rankingOtherCount)
		scope[position] = rankingOtherConversationID(index)
		scoped[rankingOtherConversationID(index)] = true
	}
	scope[rankingLargeScopeSize-1] = rankingDenseConversation
	scoped[rankingDenseConversation] = true

	want := make([]string, 0, len(full))
	for _, result := range full {
		if scoped[result.GetConversationId()] {
			want = append(want, rankingKey(result))
		}
	}
	h.callRecorder.reset()
	got := rankingKeys(h.rankingSearch(rankingFullLimit, 0, scope))
	searches := h.callRecorder.count("Search", h.collectionName) + h.callRecorder.count("HybridSearch", h.collectionName)
	if searches != 1 {
		t.Fatalf("large scope ran %d ranking searches, want 1", searches)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("large scope rows = %d %v, want %d in full ranking order", len(got), got, len(want))
	}
	for _, key := range got {
		if !strings.HasPrefix(key, "conv/"+rankingDenseConversation+"/") && !strings.HasPrefix(key, "conv/claude:ranking-other-") {
			t.Fatalf("large scope returned row %s outside the scope", key)
		}
	}
}
