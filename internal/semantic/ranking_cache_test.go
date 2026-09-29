package semantic

import (
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
)

// TestRankingDepthBoundaries proves the ranking depth is every eligible row up
// to CollectionRankingDepth, and truncation starts one row past it.
func TestRankingDepthBoundaries(t *testing.T) {
	t.Parallel()

	cases := []struct {
		eligible      int64
		wantDepth     int
		wantTruncated bool
	}{
		{eligible: 0, wantDepth: 0, wantTruncated: false},
		{eligible: 1, wantDepth: 1, wantTruncated: false},
		{eligible: 16_383, wantDepth: 16_383, wantTruncated: false},
		{eligible: 16_384, wantDepth: 16_384, wantTruncated: false},
		{eligible: 16_385, wantDepth: 16_384, wantTruncated: true},
		{eligible: 3_500_000, wantDepth: 16_384, wantTruncated: true},
	}
	for _, testCase := range cases {
		if got := RankingDepth(testCase.eligible); got != testCase.wantDepth {
			t.Fatalf("RankingDepth(%d) = %d, want %d", testCase.eligible, got, testCase.wantDepth)
		}
		if got := RankingTruncated(testCase.eligible); got != testCase.wantTruncated {
			t.Fatalf("RankingTruncated(%d) = %t, want %t", testCase.eligible, got, testCase.wantTruncated)
		}
	}
}

// testRankingCollection is the collection name of the rankings the cache
// tests store at generation 0.
const testRankingCollection = "conv_chunks_cache_test"

func baseRankingKey() rankingKey {
	return rankingKey{
		CollectionName:  "conv_chunks_base",
		CollectionID:    461,
		WriteGeneration: 3,
		Query:           "needle",
		Hybrid:          true,
		Filter: compiledFilter{
			Expression: `conversationId in {p0} and provider in {p1} and role == "user" and timestampUnix >= 100 and timestampUnix < 200`,
			Params: []filterTemplateParam{
				{Name: "p0", Type: model.ScalarTypeString, Strings: []string{"claude:a", "claude:b"}, Bools: nil, Int64s: nil},
				{Name: "p1", Type: model.ScalarTypeString, Strings: []string{"claude"}, Bools: nil, Int64s: nil},
			},
		},
		MinScore:      0.25,
		GroupColumn:   "conversationId",
		PerGroupLimit: 2,
		CallerState:   "fp-1",
	}
}

// TestRankingKeyDigestCoversEveryField proves each key field changes the
// digest, and a membership set sent in another order keeps it.
func TestRankingKeyDigestCoversEveryField(t *testing.T) {
	t.Parallel()

	base := baseRankingKey().digest()
	mutations := map[string]func(*rankingKey){
		"collection name":  func(key *rankingKey) { key.CollectionName = "conv_chunks_other" },
		"collection id":    func(key *rankingKey) { key.CollectionID = 462 },
		"write generation": func(key *rankingKey) { key.WriteGeneration = 4 },
		"query":            func(key *rankingKey) { key.Query = "needles" },
		"hybrid":           func(key *rankingKey) { key.Hybrid = false },
		"expression": func(key *rankingKey) {
			key.Filter.Expression = `conversationId in {p0} and provider in {p1} and role == "user" and timestampUnix >= 100 and timestampUnix < 201`
		},
		"conversation id set": func(key *rankingKey) {
			key.Filter.Params[0].Strings = []string{"claude:a", "claude:c"}
		},
		"conversation id added": func(key *rankingKey) {
			key.Filter.Params[0].Strings = []string{"claude:a", "claude:b", "claude:c"}
		},
		"provider set":    func(key *rankingKey) { key.Filter.Params[1].Strings = []string{"codex"} },
		"param type":      func(key *rankingKey) { key.Filter.Params[1].Type = model.ScalarTypeBool },
		"min score":       func(key *rankingKey) { key.MinScore = 0.26 },
		"group column":    func(key *rankingKey) { key.GroupColumn = "" },
		"per group limit": func(key *rankingKey) { key.PerGroupLimit = 3 },
		"caller state":    func(key *rankingKey) { key.CallerState = "fp-2" },
	}
	for name, mutate := range mutations {
		key := baseRankingKey()
		key.Filter.Params = slices.Clone(key.Filter.Params)
		for index := range key.Filter.Params {
			key.Filter.Params[index].Strings = slices.Clone(key.Filter.Params[index].Strings)
		}
		mutate(&key)
		if key.digest() == base {
			t.Fatalf("changing the %s kept the digest %s", name, base)
		}
	}

	reordered := baseRankingKey()
	reordered.Filter.Params[0].Strings = []string{"claude:b", "claude:a"}
	if reordered.digest() != base {
		t.Fatal("a reordered conversation id set changed the digest")
	}
}

type fakeRankingClock struct {
	now time.Time
}

func (clock *fakeRankingClock) read() time.Time {
	return clock.now
}

func rankingOf(eligible int64, count int, prefix string) collectionRanking {
	candidates := make([]rankedCandidate, 0, count)
	for index := range count {
		candidates = append(candidates, candidate(
			fmt.Sprintf("%s-%04d", prefix, index),
			fmt.Sprintf("conv/%s/%d", prefix, index),
			fmt.Sprintf("claude:%s-%d", prefix, index%7),
			1-float64(index)/float64(count+1),
		))
	}
	return collectionRanking{Candidates: candidates, Eligible: eligible, Truncated: RankingTruncated(eligible), CallerState: ""}
}

// TestRankingCacheMissesOtherKeysAndCounts proves a ranking serves only its
// own key and eligible count: another conversation id set, another score
// floor, and another eligible count each miss.
func TestRankingCacheMissesOtherKeysAndCounts(t *testing.T) {
	t.Parallel()

	clock := &fakeRankingClock{now: time.Unix(1_700_000_000, 0)}
	cache := newRankingCache(clock.read, RankingCacheMaxBytes)
	base := baseRankingKey()
	cache.put(base.digest(), testRankingCollection, 0, rankingOf(40, 40, "base"))

	if _, found := cache.get(base.digest(), 40); !found {
		t.Fatal("the stored key missed")
	}
	otherScope := baseRankingKey()
	otherScope.Filter.Params[0].Strings = []string{"claude:a"}
	if _, found := cache.get(otherScope.digest(), 40); found {
		t.Fatal("another conversation id set hit the stored ranking")
	}
	otherFloor := baseRankingKey()
	otherFloor.MinScore = 0.5
	if _, found := cache.get(otherFloor.digest(), 40); found {
		t.Fatal("another min_score hit the stored ranking")
	}
	if _, found := cache.get(base.digest(), 41); found {
		t.Fatal("another eligible count hit the stored ranking")
	}
}

// TestRankingCacheWriteGenerationChangesKey proves a noted write changes the
// generation that later keys include.
func TestRankingCacheWriteGenerationChangesKey(t *testing.T) {
	t.Parallel()

	cache := newRankingCache(time.Now, RankingCacheMaxBytes)
	key := baseRankingKey()
	key.WriteGeneration = cache.writeGeneration(key.CollectionName)
	cache.put(key.digest(), key.CollectionName, key.WriteGeneration, rankingOf(10, 10, "gen"))
	cache.noteWrite("conv_chunks_unrelated")
	if generation := cache.writeGeneration(key.CollectionName); generation != key.WriteGeneration {
		t.Fatalf("a write to another collection moved the generation to %d", generation)
	}
	cache.noteWrite(key.CollectionName)
	after := key
	after.WriteGeneration = cache.writeGeneration(key.CollectionName)
	if after.WriteGeneration == key.WriteGeneration {
		t.Fatal("a write kept the generation")
	}
	if _, found := cache.get(after.digest(), 10); found {
		t.Fatal("the key after a write hit the ranking stored before it")
	}
}

// TestRankingCacheWritePurgesCollectionRankings proves a noted write removes
// every stored ranking of its collection and keeps other collections'
// rankings, and that a ranking computed before a write is not stored after it.
func TestRankingCacheWritePurgesCollectionRankings(t *testing.T) {
	t.Parallel()

	cache := newRankingCache(time.Now, RankingCacheMaxBytes)
	cache.put("written-a", "conv_chunks_written", 0, rankingOf(10, 10, "a"))
	cache.put("written-b", "conv_chunks_written", 0, rankingOf(10, 10, "b"))
	cache.put("other", "conv_chunks_other", 0, rankingOf(10, 10, "c"))
	other := rankingOf(10, 10, "c").estimatedBytes()

	cache.noteWrite("conv_chunks_written")
	if len(cache.entries) != 1 || cache.usedBytes != other {
		t.Fatalf("after a write the cache stores %d rankings and %d bytes, want 1 and %d", len(cache.entries), cache.usedBytes, other)
	}
	if _, found := cache.get("other", 10); !found {
		t.Fatal("a write to one collection removed another collection's ranking")
	}
	cache.put("late", "conv_chunks_written", 0, rankingOf(10, 10, "late"))
	if _, found := cache.get("late", 10); found {
		t.Fatal("a ranking computed before a write was stored after it")
	}
}

// TestRankingCacheExpiresAfterTTL proves an entry serves until RankingCacheTTL
// passes without a read, that every read restarts that period, and that the
// entry misses once a full period passes without a read.
func TestRankingCacheExpiresAfterTTL(t *testing.T) {
	t.Parallel()

	clock := &fakeRankingClock{now: time.Unix(1_700_000_000, 0)}
	cache := newRankingCache(clock.read, RankingCacheMaxBytes)
	digest := baseRankingKey().digest()
	cache.put(digest, testRankingCollection, 0, rankingOf(5, 5, "ttl"))

	for read := range 3 {
		clock.now = clock.now.Add(RankingCacheTTL - time.Second)
		if _, found := cache.get(digest, 5); !found {
			t.Fatalf("read %d missed one second before the TTL after the previous read", read)
		}
	}
	clock.now = clock.now.Add(RankingCacheTTL)
	if _, found := cache.get(digest, 5); found {
		t.Fatal("the entry hit a full TTL after its last read")
	}
	if cache.usedBytes != 0 {
		t.Fatalf("expired entry left %d bytes counted", cache.usedBytes)
	}
}

// TestRankingCacheSharesOneComputation proves concurrent requests that miss
// one digest and count run the ranking computation once and all receive its
// ranking.
func TestRankingCacheSharesOneComputation(t *testing.T) {
	t.Parallel()

	clock := &fakeRankingClock{now: time.Unix(1_700_000_000, 0)}
	cache := newRankingCache(clock.read, RankingCacheMaxBytes)
	digest := baseRankingKey().digest()
	const requests = 8
	release := make(chan struct{})
	var computations atomic.Int32
	compute := func() (collectionRanking, error) {
		computations.Add(1)
		<-release
		return rankingOf(5, 5, "shared"), nil
	}
	var started sync.WaitGroup
	var finished sync.WaitGroup
	results := make([]collectionRanking, requests)
	failures := make([]error, requests)
	for request := range requests {
		started.Add(1)
		finished.Add(1)
		go func() {
			defer finished.Done()
			started.Done()
			results[request], failures[request] = cache.rank(digest, testRankingCollection, 0, 5, compute)
		}()
	}
	started.Wait()
	for computations.Load() == 0 {
		time.Sleep(time.Millisecond)
	}
	time.Sleep(20 * time.Millisecond)
	close(release)
	finished.Wait()

	if got := computations.Load(); got != 1 {
		t.Fatalf("%d concurrent misses ran %d computations, want 1", requests, got)
	}
	for request := range requests {
		if failures[request] != nil {
			t.Fatalf("request %d failed: %v", request, failures[request])
		}
		if len(results[request].Candidates) != 5 {
			t.Fatalf("request %d received %d candidates, want 5", request, len(results[request].Candidates))
		}
	}
	if _, found := cache.get(digest, 5); !found {
		t.Fatal("the shared computation did not store its ranking")
	}
}

// TestRankingCacheEvictsLeastRecentlyUsed proves storing past the byte bound
// evicts the ranking read longest ago and keeps the rest.
func TestRankingCacheEvictsLeastRecentlyUsed(t *testing.T) {
	t.Parallel()

	first := rankingOf(100, 100, "a")
	second := rankingOf(100, 100, "b")
	third := rankingOf(100, 100, "c")
	bound := first.estimatedBytes() + second.estimatedBytes() + third.estimatedBytes()/2
	clock := &fakeRankingClock{now: time.Unix(1_700_000_000, 0)}
	cache := newRankingCache(clock.read, bound)
	cache.put("first", testRankingCollection, 0, first)
	cache.put("second", testRankingCollection, 0, second)
	if _, found := cache.get("first", 100); !found {
		t.Fatal("first ranking missed before eviction")
	}
	cache.put("third", testRankingCollection, 0, third)

	if _, found := cache.get("second", 100); found {
		t.Fatal("the least recently used ranking survived eviction")
	}
	if _, found := cache.get("first", 100); !found {
		t.Fatal("the recently read ranking was evicted")
	}
	if _, found := cache.get("third", 100); !found {
		t.Fatal("the newest ranking was evicted")
	}
	if cache.usedBytes > bound {
		t.Fatalf("cache counts %d bytes, above the bound %d", cache.usedBytes, bound)
	}

	oversized := newRankingCache(clock.read, first.estimatedBytes()-1)
	oversized.put("first", testRankingCollection, 0, first)
	if _, found := oversized.get("first", 100); found {
		t.Fatal("a ranking larger than the bound was stored")
	}
}

// TestCachedRankingPagesInClydeShape pages a cached ranking the way Clyde
// does: each request asks for offset plus the page size and keeps the rows
// past the offset. The pages together equal the whole selection, with no row
// repeated or omitted, at page sizes 1, 10, and 100, and reading the cached
// ranking leaves it unchanged.
func TestCachedRankingPagesInClydeShape(t *testing.T) {
	t.Parallel()

	const rankedRows = 437
	cache := newRankingCache(time.Now, RankingCacheMaxBytes)
	stored := rankingOf(rankedRows, rankedRows, "page")
	sortRankedCandidates(stored.Candidates)
	storedKeys := candidateKeys(stored.Candidates)
	cache.put("page", testRankingCollection, 0, stored)

	for _, groupLimit := range []int32{0, 3} {
		want := candidateKeys(selectRankedCandidates(stored.Candidates, groupLimit, 0.1, 0))
		for _, pageSize := range []int{1, 10, 100} {
			paged := make([]string, 0, len(want))
			for offset := 0; ; offset += pageSize {
				ranking, found := cache.get("page", rankedRows)
				if !found {
					t.Fatalf("page at offset %d missed the cache", offset)
				}
				selected := selectRankedCandidates(ranking.Candidates, groupLimit, 0.1, int32(offset+pageSize))
				if len(selected) <= offset {
					break
				}
				paged = append(paged, candidateKeys(selected[offset:])...)
			}
			if !slices.Equal(paged, want) {
				t.Fatalf("group limit %d page size %d pages = %d rows, want %d rows in ranking order", groupLimit, pageSize, len(paged), len(want))
			}
		}
	}
	ranking, _ := cache.get("page", rankedRows)
	if !slices.Equal(candidateKeys(ranking.Candidates), storedKeys) {
		t.Fatal("paging changed the cached ranking")
	}
}

// TestPrimaryKeyColumnKeepsCallerKeys proves the id column the ranking passes
// to the Milvus client WithIDs option leaves the caller's keys unchanged,
// although the option quotes the column values in place.
func TestPrimaryKeyColumnKeepsCallerKeys(t *testing.T) {
	t.Parallel()

	keys := []string{"chunk_a", "chunk_b"}
	idColumn := primaryKeyColumn(keys)
	milvusclient.NewQueryOption("conv_chunks_ids").WithIDs(idColumn)
	if !slices.Equal(keys, []string{"chunk_a", "chunk_b"}) {
		t.Fatalf("keys after WithIDs = %v, want them unchanged", keys)
	}
	quoted, err := idColumn.GetAsString(0)
	if err != nil {
		t.Fatalf("read id column: %v", err)
	}
	if quoted != `"chunk_a"` {
		t.Fatalf("id column value after WithIDs = %q, want the client to quote its own copy", quoted)
	}
}

