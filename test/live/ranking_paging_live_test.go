//go:build live

package live

import (
	"fmt"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	pagingQuery               = "needle"
	pagingConversationCount   = 10
	pagingBaseMessages        = 15
	pagingScopedConversations = 4
	pagingRestartPageSize     = 10
	pagingRestartPages        = 6

	invalidationFirstMessages  = 5
	invalidationAddedMessages  = 3
	invalidationRewriteCount   = 5
	invalidationSearchLimit    = 50
	invalidationRewrittenToken = "rewritten"
)

var pagingPageSizes = []int{1, 10, 100}

func pagingConversationID(index int) string {
	return fmt.Sprintf("claude:paging-%02d", index)
}

// pagingCorpus has ten conversations with 15 to 33 messages each. Every
// message contains the query word once, and the message at global position n
// adds n filler words. Each message then has a distinct length and a distinct
// BM25 score, and a distinct text with a distinct fake embedding.
func pagingCorpus() (map[string][]*pb.ConversationDocument, int) {
	conversations := map[string][]*pb.ConversationDocument{}
	position := 0
	for conversationIndex := range pagingConversationCount {
		conversationID := pagingConversationID(conversationIndex)
		messageCount := pagingBaseMessages + 2*conversationIndex
		for messageIndex := range messageCount {
			filler := make([]string, 0, position)
			for word := range position {
				filler = append(filler, fmt.Sprintf("w%d", word))
			}
			conversations[conversationID] = append(conversations[conversationID], &pb.ConversationDocument{
				ConversationId: conversationID,
				MessageIndex:   int32(messageIndex),
				Role:           "user",
				TimestampUnix:  int64(1_650_000_000 + position),
				Text:           strings.TrimSpace(pagingQuery + " " + strings.Join(filler, " ")),
			})
			position++
		}
	}
	return conversations, position
}

// conversationPage runs one SearchConversations request.
func (h *harness) conversationPage(limit int32, conversationIDs []string) *pb.SearchConversationsResponse {
	h.t.Helper()
	response, err := h.searchConversationsAt(limit, 0, conversationIDs)
	if err != nil {
		h.t.Fatalf("SearchConversations(limit %d) returned error: %v", limit, err)
	}
	return response
}

// searchConversationsAt runs one SearchConversations request with an offset.
func (h *harness) searchConversationsAt(limit int32, offset int32, conversationIDs []string) (*pb.SearchConversationsResponse, error) {
	h.t.Helper()
	return h.client.SearchConversations(correlatedContext(), &pb.SearchConversationsRequest{
		CollectionId:         h.collectionID,
		Query:                pagingQuery,
		Limit:                limit,
		PerConversationLimit: 0,
		Filter:               &pb.ConversationSearchFilter{ConversationIds: conversationIDs},
		Offset:               offset,
	})
}

// pageByOffset pages one query with the offset field: each request asks for
// pageSize rows at the number of rows already read. It stops at the first page
// shorter than pageSize and returns the kept keys and each request's duration.
func (h *harness) pageByOffset(pageSize int, conversationIDs []string) ([]string, []time.Duration) {
	h.t.Helper()
	kept := make([]string, 0)
	durations := make([]time.Duration, 0)
	for {
		started := time.Now()
		response, err := h.searchConversationsAt(int32(pageSize), int32(len(kept)), conversationIDs)
		durations = append(durations, time.Since(started))
		if err != nil {
			h.t.Fatalf("SearchConversations(offset %d, limit %d) returned error: %v", len(kept), pageSize, err)
		}
		page := rankingKeys(response.GetResults())
		if len(page) > pageSize {
			h.t.Fatalf("SearchConversations(offset %d, limit %d) returned %d rows", len(kept), pageSize, len(page))
		}
		kept = append(kept, page...)
		if len(page) < pageSize {
			return kept, durations
		}
	}
}

// pageInClydeShape pages one query the way Clyde does: each request asks for
// offset plus pageSize rows and keeps the rows from offset on. It starts at
// startOffset and stops at the first page shorter than pageSize or after
// maxPages pages when maxPages is positive. It returns the kept keys and each
// request's duration.
func (h *harness) pageInClydeShape(pageSize int, conversationIDs []string, startOffset int, maxPages int) ([]string, []time.Duration) {
	h.t.Helper()
	kept := make([]string, 0)
	durations := make([]time.Duration, 0)
	for offset, pages := startOffset, 0; maxPages <= 0 || pages < maxPages; offset, pages = offset+pageSize, pages+1 {
		started := time.Now()
		results := h.conversationPage(int32(offset+pageSize), conversationIDs).GetResults()
		durations = append(durations, time.Since(started))
		if len(results) <= offset {
			break
		}
		kept = append(kept, rankingKeys(results[offset:])...)
		if len(results) < offset+pageSize {
			break
		}
	}
	return kept, durations
}

func requireSamePages(t *testing.T, label string, paged []string, full []string) {
	t.Helper()
	seen := make(map[string]int, len(paged))
	for position, key := range paged {
		if earlier, repeated := seen[key]; repeated {
			t.Fatalf("%s: row %s repeats at positions %d and %d", label, key, earlier, position)
		}
		seen[key] = position
	}
	omitted := 0
	for _, key := range full {
		if _, found := seen[key]; !found {
			omitted++
		}
	}
	if omitted != 0 || len(paged) != len(full) {
		t.Fatalf("%s: %d paged rows, %d full rows, %d omitted", label, len(paged), len(full), omitted)
	}
	if !slices.Equal(paged, full) {
		t.Fatalf("%s: paged key order differs from the full ranking", label)
	}
}

func durationSummary(durations []time.Duration) string {
	if len(durations) == 0 {
		return "no requests"
	}
	sorted := slices.Clone(durations)
	sort.Slice(sorted, func(first int, second int) bool { return sorted[first] < sorted[second] })
	return fmt.Sprintf("%d requests, p50 %s, max %s", len(sorted), sorted[len(sorted)/2], sorted[len(sorted)-1])
}

// newPagingHarness ingests the paging corpus through the daemon.
func newPagingHarness(t *testing.T) (*harness, int, []string, int) {
	t.Helper()
	h := newRealEmbeddingHarness(t)
	corpus, total := pagingCorpus()
	started := time.Now()
	requireCompleted(t, h.upsert(corpus, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, false, false), "paging corpus ingest")
	t.Logf("paging corpus: %d rows ingested in %s into database %s", total, time.Since(started), h.databaseName)
	scope := make([]string, 0, pagingScopedConversations)
	scopedRows := 0
	for index := range pagingScopedConversations {
		conversationID := pagingConversationID(2 * index)
		scope = append(scope, conversationID)
		scopedRows += len(corpus[conversationID])
	}
	return h, total, scope, scopedRows
}

// TestConversationSearchPagesToExhaustion pages one query to its end in the
// Clyde request shape at page sizes 1, 10, and 100, unfiltered and filtered by
// conversation ids. The pages equal the product's own one ranking, a single
// request with limit equal to the eligible row count, with no row repeated or
// omitted and the same key order.
func TestConversationSearchPagesToExhaustion(t *testing.T) {
	h, total, scope, scopedRows := newPagingHarness(t)
	cases := []struct {
		name     string
		scope    []string
		eligible int
	}{
		{name: "unfiltered", scope: nil, eligible: total},
		{name: "conversation ids", scope: scope, eligible: scopedRows},
	}
	for _, testCase := range cases {
		fullResponse := h.conversationPage(int32(testCase.eligible), testCase.scope)
		full := rankingKeys(fullResponse.GetResults())
		if len(full) != testCase.eligible {
			t.Fatalf("%s: full ranking has %d rows, want the %d eligible rows", testCase.name, len(full), testCase.eligible)
		}
		if fullResponse.GetRankingTruncated() {
			t.Fatalf("%s: ranking_truncated is true for %d eligible rows", testCase.name, testCase.eligible)
		}
		for _, pageSize := range pagingPageSizes {
			paged, durations := h.pageInClydeShape(pageSize, testCase.scope, 0, 0)
			requireSamePages(t, fmt.Sprintf("%s page size %d", testCase.name, pageSize), paged, full)
			t.Logf("%s page size %d: %s", testCase.name, pageSize, durationSummary(durations))
		}
	}
}

// TestConversationSearchPagesAcrossRestart pages part of one query, restarts
// the daemon, which empties the ranking cache, and pages the rest. The new
// daemon ranks again on unchanged rows. The pages from both daemons together
// equal the full ranking from before the restart.
func TestConversationSearchPagesAcrossRestart(t *testing.T) {
	h, total, scope, scopedRows := newPagingHarness(t)
	cases := []struct {
		name     string
		scope    []string
		eligible int
	}{
		{name: "unfiltered", scope: nil, eligible: total},
		{name: "conversation ids", scope: scope, eligible: scopedRows},
	}
	fulls := make([][]string, 0, len(cases))
	firstHalves := make([][]string, 0, len(cases))
	for _, testCase := range cases {
		full := rankingKeys(h.conversationPage(int32(testCase.eligible), testCase.scope).GetResults())
		if len(full) != testCase.eligible {
			t.Fatalf("%s: full ranking has %d rows, want %d", testCase.name, len(full), testCase.eligible)
		}
		firstHalf, _ := h.pageInClydeShape(pagingRestartPageSize, testCase.scope, 0, pagingRestartPages)
		fulls = append(fulls, full)
		firstHalves = append(firstHalves, firstHalf)
	}
	h.restart(nil)
	for index, testCase := range cases {
		secondHalf, durations := h.pageInClydeShape(pagingRestartPageSize, testCase.scope, pagingRestartPageSize*pagingRestartPages, 0)
		paged := append(slices.Clone(firstHalves[index]), secondHalf...)
		requireSamePages(t, testCase.name+" across restart", paged, fulls[index])
		t.Logf("%s after restart: %s", testCase.name, durationSummary(durations))
	}
}

// TestConversationSearchKeysRankingsByFilter searches two conversation id
// filters that match the same number of rows. The eligible count check cannot
// tell the two rankings apart, and only the filter in the ranking key keeps the
// second search from reading the first search's cached ranking.
func TestConversationSearchKeysRankingsByFilter(t *testing.T) {
	h, _, _, _ := newPagingHarness(t)
	corpus, _ := pagingCorpus()
	first := []string{pagingConversationID(0), pagingConversationID(3)}
	second := []string{pagingConversationID(1), pagingConversationID(2)}
	firstRows := len(corpus[first[0]]) + len(corpus[first[1]])
	secondRows := len(corpus[second[0]]) + len(corpus[second[1]])
	if firstRows != secondRows {
		t.Fatalf("filters match %d and %d rows, want equal counts", firstRows, secondRows)
	}
	for _, scope := range [][]string{first, second} {
		results := h.conversationPage(int32(firstRows), scope).GetResults()
		if len(results) != firstRows {
			t.Fatalf("filter %v returned %d rows, want %d", scope, len(results), firstRows)
		}
		for _, result := range results {
			if !slices.Contains(scope, result.GetConversationId()) {
				t.Fatalf("filter %v returned row %s from conversation %s", scope, rankingKey(result), result.GetConversationId())
			}
		}
	}
}

// TestConversationSearchPagesByOffset pages one query with the offset request
// field at page sizes 1, 10, and 100, unfiltered and filtered by conversation
// ids. Each request returns only its page. The pages equal the full ranking
// with no row repeated or omitted. A negative offset is an invalid argument.
func TestConversationSearchPagesByOffset(t *testing.T) {
	h, total, scope, scopedRows := newPagingHarness(t)
	cases := []struct {
		name     string
		scope    []string
		eligible int
	}{
		{name: "unfiltered", scope: nil, eligible: total},
		{name: "conversation ids", scope: scope, eligible: scopedRows},
	}
	for _, testCase := range cases {
		full := rankingKeys(h.conversationPage(int32(testCase.eligible), testCase.scope).GetResults())
		if len(full) != testCase.eligible {
			t.Fatalf("%s: full ranking has %d rows, want %d", testCase.name, len(full), testCase.eligible)
		}
		for _, pageSize := range pagingPageSizes {
			paged, durations := h.pageByOffset(pageSize, testCase.scope)
			requireSamePages(t, fmt.Sprintf("%s offset page size %d", testCase.name, pageSize), paged, full)
			t.Logf("%s offset page size %d: %s", testCase.name, pageSize, durationSummary(durations))
		}
	}
	if _, err := h.searchConversationsAt(10, -1, nil); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("offset -1 returned %v, want InvalidArgument", err)
	}
}

func invalidationMessages(conversationID string, count int, token string) []*pb.ConversationDocument {
	documents := make([]*pb.ConversationDocument, 0, count)
	for messageIndex := range count {
		documents = append(documents, &pb.ConversationDocument{
			ConversationId: conversationID,
			MessageIndex:   int32(messageIndex),
			Role:           "user",
			TimestampUnix:  int64(1_670_000_000 + messageIndex),
			Text:           fmt.Sprintf("%s %s message %d", pagingQuery, token, messageIndex),
		})
	}
	return documents
}

// TestConversationSearchSeesDaemonWritesBetweenSearches searches one key,
// writes through the daemon, and searches the same key again. Rows the write
// adds appear with the new indexed fingerprint. A rewrite that keeps the
// matching row count returns the rewritten rows.
func TestConversationSearchSeesDaemonWritesBetweenSearches(t *testing.T) {
	h := newRealEmbeddingHarness(t)
	grown := "claude:invalidation-grown"
	rewritten := "claude:invalidation-rewritten"
	first := map[string][]*pb.ConversationDocument{
		grown:     invalidationMessages(grown, invalidationFirstMessages, "original"),
		rewritten: invalidationMessages(rewritten, invalidationRewriteCount, "original"),
	}
	requireCompleted(t, h.upsert(first, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, false, false), "invalidation corpus ingest")

	within := func() *pb.SearchWithinConversationResponse {
		t.Helper()
		response, err := h.client.SearchWithinConversation(correlatedContext(), &pb.SearchWithinConversationRequest{
			CollectionId:   h.collectionID,
			ConversationId: grown,
			Query:          pagingQuery,
			Limit:          invalidationSearchLimit,
		})
		if err != nil {
			t.Fatalf("SearchWithinConversation returned error: %v", err)
		}
		return response
	}
	before := within()
	if len(before.GetResults()) != invalidationFirstMessages || before.GetIndexedFingerprint() != fingerprint(first[grown]) {
		t.Fatalf("before the write: %d rows with fingerprint %q, want %d rows with %q", len(before.GetResults()), before.GetIndexedFingerprint(), invalidationFirstMessages, fingerprint(first[grown]))
	}
	rewrittenBefore := h.conversationPage(invalidationSearchLimit, []string{rewritten}).GetResults()
	if len(rewrittenBefore) != invalidationRewriteCount {
		t.Fatalf("before the rewrite: %d rows, want %d", len(rewrittenBefore), invalidationRewriteCount)
	}

	second := map[string][]*pb.ConversationDocument{
		grown:     invalidationMessages(grown, invalidationFirstMessages+invalidationAddedMessages, "original"),
		rewritten: invalidationMessages(rewritten, invalidationRewriteCount, invalidationRewrittenToken),
	}
	requireCompleted(t, h.upsert(map[string][]*pb.ConversationDocument{grown: second[grown]}, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, false, false), "invalidation append")
	// The message-level delta embeds only new message indexes. A forced
	// ingest rebuilds the conversation and replaces every message row.
	requireCompleted(t, h.upsert(map[string][]*pb.ConversationDocument{rewritten: second[rewritten]}, pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN, false, true), "invalidation rewrite")

	after := within()
	if len(after.GetResults()) != invalidationFirstMessages+invalidationAddedMessages {
		t.Fatalf("after the write: %d rows, want %d", len(after.GetResults()), invalidationFirstMessages+invalidationAddedMessages)
	}
	if after.GetIndexedFingerprint() != fingerprint(second[grown]) || after.GetIndexedFingerprint() == before.GetIndexedFingerprint() {
		t.Fatalf("after the write: fingerprint %q, want the new fingerprint %q", after.GetIndexedFingerprint(), fingerprint(second[grown]))
	}
	rewrittenAfter := h.conversationPage(invalidationSearchLimit, []string{rewritten}).GetResults()
	if len(rewrittenAfter) != invalidationRewriteCount {
		t.Fatalf("after the rewrite: %d rows, want %d", len(rewrittenAfter), invalidationRewriteCount)
	}
	for _, result := range rewrittenAfter {
		if !strings.Contains(result.GetContent(), invalidationRewrittenToken) {
			t.Fatalf("after the rewrite: row %s content %q, want the rewritten text", rankingKey(result), result.GetContent())
		}
	}
}
