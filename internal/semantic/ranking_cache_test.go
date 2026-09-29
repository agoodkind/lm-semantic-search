package semantic

import (
	"fmt"
	"math"
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
		WriteGeneration: 0,
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

// cacheKey returns baseRankingKey with query name in collection at write
// generation 0.
func cacheKey(name string, collection string) rankingKey {
	key := baseRankingKey()
	key.Query = name
	key.CollectionName = collection
	return key
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
	continued := baseRankingKey()
	continued.CollectionID = 462
	continued.WriteGeneration = 4
	continued.CallerState = "fp-2"
	if continued.requestDigest() != baseRankingKey().requestDigest() {
		t.Fatal("the collection id, write generation, or caller state changed the request digest")
	}
	requested := baseRankingKey()
	requested.MinScore = 0.26
	if requested.requestDigest() == baseRankingKey().requestDigest() {
		t.Fatal("changing the min score kept the request digest")
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
	cache.put(base, rankingOf(40, 40, "base"))

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
	cache.put(key, rankingOf(10, 10, "gen"))
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
	cache.put(cacheKey("written-a", "conv_chunks_written"), rankingOf(10, 10, "a"))
	cache.put(cacheKey("written-b", "conv_chunks_written"), rankingOf(10, 10, "b"))
	cache.put(cacheKey("other", "conv_chunks_other"), rankingOf(10, 10, "c"))
	other := rankingOf(10, 10, "c").estimatedBytes()

	cache.noteWrite("conv_chunks_written")
	if len(cache.entries) != 1 || cache.usedBytes != other {
		t.Fatalf("after a write the cache stores %d rankings and %d bytes, want 1 and %d", len(cache.entries), cache.usedBytes, other)
	}
	if _, found := cache.get(cacheKey("other", "conv_chunks_other").digest(), 10); !found {
		t.Fatal("a write to one collection removed another collection's ranking")
	}
	cache.put(cacheKey("late", "conv_chunks_written"), rankingOf(10, 10, "late"))
	if _, found := cache.get(cacheKey("late", "conv_chunks_written").digest(), 10); found {
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
	cache.put(baseRankingKey(), rankingOf(5, 5, "ttl"))

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
			results[request], failures[request] = cache.rank(baseRankingKey(), 5, compute)
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
	cache.put(cacheKey("first", testRankingCollection), first)
	cache.put(cacheKey("second", testRankingCollection), second)
	if _, found := cache.get(cacheKey("first", testRankingCollection).digest(), 100); !found {
		t.Fatal("first ranking missed before eviction")
	}
	cache.put(cacheKey("third", testRankingCollection), third)

	if _, found := cache.get(cacheKey("second", testRankingCollection).digest(), 100); found {
		t.Fatal("the least recently used ranking survived eviction")
	}
	if _, found := cache.get(cacheKey("first", testRankingCollection).digest(), 100); !found {
		t.Fatal("the recently read ranking was evicted")
	}
	if _, found := cache.get(cacheKey("third", testRankingCollection).digest(), 100); !found {
		t.Fatal("the newest ranking was evicted")
	}
	if cache.usedBytes > bound {
		t.Fatalf("cache counts %d bytes, above the bound %d", cache.usedBytes, bound)
	}

	oversized := newRankingCache(clock.read, first.estimatedBytes()-1)
	oversized.put(cacheKey("first", testRankingCollection), first)
	if _, found := oversized.get(cacheKey("first", testRankingCollection).digest(), 100); found {
		t.Fatal("a ranking larger than the bound was stored")
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

// TestPageWindowSelectsOffsetRows proves an offset page selects offset plus
// limit rows, capped at the int32 maximum, and starts at the offset capped at
// the selection length. Paging a cached ranking by offset returns the whole
// selection once at page sizes 1, 10, and 100.
func TestPageWindowSelectsOffsetRows(t *testing.T) {
	t.Parallel()

	for _, testCase := range []struct {
		offset int32
		limit  int32
		want   int32
	}{
		{offset: 0, limit: 10, want: 10},
		{offset: 30, limit: 10, want: 40},
		{offset: -5, limit: 10, want: 10},
		{offset: math.MaxInt32 - 3, limit: 10, want: math.MaxInt32},
	} {
		if got := PageSelectionLimit(testCase.offset, testCase.limit); got != testCase.want {
			t.Fatalf("PageSelectionLimit(%d, %d) = %d, want %d", testCase.offset, testCase.limit, got, testCase.want)
		}
	}
	if got := PageStart(30, 12); got != 12 {
		t.Fatalf("PageStart(30, 12) = %d, want 12", got)
	}
	if got := PageStart(-1, 12); got != 0 {
		t.Fatalf("PageStart(-1, 12) = %d, want 0", got)
	}

	ranking := rankingOf(137, 137, "offset")
	whole := selectRankedCandidates(ranking.Candidates, 0, 0, int32(len(ranking.Candidates)))
	for _, pageSize := range []int32{1, 10, 100} {
		paged := make([]rankedCandidate, 0, len(whole))
		for offset := int32(0); ; offset += pageSize {
			selected := selectRankedCandidates(ranking.Candidates, 0, 0, PageSelectionLimit(offset, pageSize))
			page := selected[PageStart(offset, len(selected)):]
			paged = append(paged, page...)
			if int32(len(page)) < pageSize {
				break
			}
		}
		if !slices.Equal(paged, whole) {
			t.Fatalf("page size %d: offset pages returned %d rows, want the %d selected rows in order", pageSize, len(paged), len(whole))
		}
	}
}

// TestRankingTokenReadsItsRankingAfterWrites proves a token reads its ranking
// after a write to the collection, while a request without a token misses it,
// that a write removes a ranking no token names, and that a token rejects
// another query and a recreated collection.
func TestRankingTokenReadsItsRankingAfterWrites(t *testing.T) {
	t.Parallel()

	cache := newRankingCache(time.Now, RankingCacheMaxBytes)
	key := baseRankingKey()
	cache.put(key, rankingOf(10, 10, "token"))
	token := cache.issueToken(key.digest())
	if token == "" {
		t.Fatal("issueToken returned no token for a stored ranking")
	}
	plain := cacheKey("plain", key.CollectionName)
	cache.put(plain, rankingOf(10, 10, "plain"))

	cache.noteWrite(key.CollectionName)
	after := key
	after.WriteGeneration = cache.writeGeneration(key.CollectionName)
	after.CallerState = "fp-after-write"
	if _, found := cache.get(after.digest(), 10); found {
		t.Fatal("a request without a token read the ranking computed before the write")
	}
	if issued := cache.issueToken(plain.digest()); issued != "" {
		t.Fatal("a write kept a ranking that no token names")
	}
	ranking, reason := cache.readToken(token, key.CollectionID, after.requestDigest())
	if reason != "" {
		t.Fatalf("readToken after a write returned %q", reason)
	}
	if len(ranking.Candidates) != 10 || ranking.Candidates[0].PrimaryKey != "token-0000" {
		t.Fatalf("readToken returned %d candidates starting %v, want the stored ranking", len(ranking.Candidates), ranking.Candidates)
	}
	otherQuery := key
	otherQuery.Query = "another query"
	if _, reason := cache.readToken(token, key.CollectionID, otherQuery.requestDigest()); reason != tokenReasonMismatch {
		t.Fatalf("readToken for another query returned %q, want %q", reason, tokenReasonMismatch)
	}
	if _, reason := cache.readToken(token, key.CollectionID+1, key.requestDigest()); reason != tokenReasonRecreated {
		t.Fatalf("readToken for a recreated collection returned %q, want %q", reason, tokenReasonRecreated)
	}
}

// TestRankingTokenExpires proves a token reads no ranking once RankingCacheTTL
// passes without a read, and that an unknown or malformed token reads none.
func TestRankingTokenExpires(t *testing.T) {
	t.Parallel()

	clock := &fakeRankingClock{now: time.Unix(1_700_000_000, 0)}
	cache := newRankingCache(clock.read, RankingCacheMaxBytes)
	key := baseRankingKey()
	cache.put(key, rankingOf(5, 5, "expiring"))
	token := cache.issueToken(key.digest())
	clock.now = clock.now.Add(RankingCacheTTL - time.Second)
	if _, reason := cache.readToken(token, key.CollectionID, key.requestDigest()); reason != "" {
		t.Fatalf("readToken one second before expiry returned %q", reason)
	}
	clock.now = clock.now.Add(RankingCacheTTL)
	for candidate, want := range map[string]string{
		token:                          tokenReasonExpired,
		rankingTokenPrefix + "unknown": tokenReasonUnknown,
		"malformed":                    tokenReasonMalformed,
	} {
		if _, reason := cache.readToken(candidate, key.CollectionID, key.requestDigest()); reason != want {
			t.Fatalf("readToken(%q) returned %q, want %q", candidate, reason, want)
		}
	}
}

// get locks the cache and reads the ranking under digest with getLocked.
func (cache *rankingCache) get(digest string, eligible int64) (collectionRanking, bool) {
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	return cache.getLocked(digest, eligible)
}
