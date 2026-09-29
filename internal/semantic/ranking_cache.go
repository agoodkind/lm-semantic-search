package semantic

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"hash"
	"log/slog"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"goodkind.io/lm-semantic-search/internal/model"
)

// rankingTokenPrefix starts every ranking token. A token is the prefix and
// the ranking key digest.
const rankingTokenPrefix = "rk1."

// ErrRankingExpired reports a ranking token that names no cached ranking: the
// ranking expired, was evicted, belongs to a dropped collection, or the daemon
// restarted.
var ErrRankingExpired = errors.New("ranking expired")

// ErrRankingTokenMismatch reports a ranking token sent with a query, filter,
// score floor, or group cap other than the ones of its ranking.
var ErrRankingTokenMismatch = errors.New("ranking token belongs to another query")

// RankingCacheTTL is how long a cached collection ranking serves pages after
// the last request that computed or read it.
const RankingCacheTTL = 10 * time.Minute

// RankingCacheMaxBytes bounds the estimated candidate bytes of every cached
// ranking together. Storing past the bound evicts the least recently used
// rankings.
const RankingCacheMaxBytes int64 = 256 << 20

// rankedCandidateFixedBytes is the in-memory size of one rankedCandidate
// without the bytes its strings point to.
var rankedCandidateFixedBytes = int64(unsafe.Sizeof(rankedCandidate{PrimaryKey: "", RelativePath: "", Group: ScalarCell{Column: "", State: "", Value: ScalarValue{Type: "", String: "", Bool: false, Int64: 0}}, Score: 0}))

// RankingDepth returns the candidate count of one collection ranking search
// over eligible rows that match the filter: every eligible row up to
// CollectionRankingDepth, the Milvus single-search ceiling. Both hybrid legs,
// the fused hybrid limit, the dense search, and the offline store rank at this
// depth. The depth depends only on the eligible count.
func RankingDepth(eligible int64) int {
	if eligible <= 0 {
		return 0
	}
	if eligible >= CollectionRankingDepth {
		return CollectionRankingDepth
	}
	return int(eligible)
}

// RankingTruncated reports whether more rows match the filter than one
// ranking search returns. Pages of such a search stop at the first
// CollectionRankingDepth rows of the ranking.
func RankingTruncated(eligible int64) bool {
	return eligible > CollectionRankingDepth
}

// collectionRanking is one computed ranking of a collection search: the
// sorted candidates after legacy group resolution, the eligible row count the
// ranking was computed over, and the caller state read before it was
// computed. It stores no content or metadata.
type collectionRanking struct {
	Candidates  []rankedCandidate
	Eligible    int64
	Truncated   bool
	CallerState string
}

// estimatedBytes returns the in-memory size of the ranking's candidates,
// counting each candidate struct and the string bytes it references.
func (ranking collectionRanking) estimatedBytes() int64 {
	total := int64(len(ranking.CallerState))
	for _, ranked := range ranking.Candidates {
		total += rankedCandidateFixedBytes
		total += int64(len(ranked.PrimaryKey) + len(ranked.RelativePath))
		total += int64(len(ranked.Group.Column) + len(ranked.Group.Value.String))
	}
	return total
}

// rankingKey lists every input that selects one ranking. Two requests with an
// equal key rank the same candidate list on an unchanged collection.
type rankingKey struct {
	CollectionName  string
	CollectionID    int64
	WriteGeneration uint64
	Query           string
	Hybrid          bool
	Filter          compiledFilter
	MinScore        float64
	GroupColumn     string
	PerGroupLimit   int32
	CallerState     string
}

// digest returns the SHA-256 of the key's canonical encoding. The encoding
// writes every field as its name and a length-prefixed value. Each template
// parameter writes its values in sorted order, and a membership set in any
// order has one encoding.
func (key rankingKey) digest() string {
	encoder := rankingKeyEncoder{hash: sha256.New()}
	encoder.field("collection_id", strconv.FormatInt(key.CollectionID, 10))
	encoder.field("write_generation", strconv.FormatUint(key.WriteGeneration, 10))
	encoder.field("caller_state", key.CallerState)
	key.encodeRequest(encoder)
	return hex.EncodeToString(encoder.hash.Sum(nil))
}

// requestDigest returns the SHA-256 of the fields a request repeats on every
// page: the collection name, query, hybrid mode, filter, score floor, and group
// cap. It leaves out the collection ID, write generation, and caller state,
// which can change between pages of one ranking.
func (key rankingKey) requestDigest() string {
	encoder := rankingKeyEncoder{hash: sha256.New()}
	key.encodeRequest(encoder)
	return hex.EncodeToString(encoder.hash.Sum(nil))
}

func (key rankingKey) encodeRequest(encoder rankingKeyEncoder) {
	encoder.field("collection", key.CollectionName)
	encoder.field("query", key.Query)
	encoder.field("hybrid", strconv.FormatBool(key.Hybrid))
	encoder.field("expression", key.Filter.Expression)
	encoder.field("params", strconv.Itoa(len(key.Filter.Params)))
	for _, param := range key.Filter.Params {
		encoder.templateParam(param)
	}
	encoder.field("min_score", strconv.FormatFloat(key.MinScore, 'g', -1, 64))
	encoder.field("group_column", key.GroupColumn)
	encoder.field("per_group_limit", strconv.FormatInt(int64(key.PerGroupLimit), 10))
}

type rankingKeyEncoder struct {
	hash hash.Hash
}

func (encoder rankingKeyEncoder) field(name string, value string) {
	for _, part := range []string{name, value} {
		_, _ = encoder.hash.Write([]byte(strconv.Itoa(len(part))))
		_, _ = encoder.hash.Write([]byte{':'})
		_, _ = encoder.hash.Write([]byte(part))
	}
}

func (encoder rankingKeyEncoder) templateParam(param filterTemplateParam) {
	encoder.field("param", param.Name)
	encoder.field("type", string(param.Type))
	values := make([]string, 0, len(param.Strings)+len(param.Bools)+len(param.Int64s))
	switch param.Type {
	case model.ScalarTypeBool:
		for _, value := range param.Bools {
			values = append(values, strconv.FormatBool(value))
		}
	case model.ScalarTypeInt64:
		for _, value := range param.Int64s {
			values = append(values, strconv.FormatInt(value, 10))
		}
	case model.ScalarTypeString:
		values = append(values, param.Strings...)
	default:
		values = append(values, param.Strings...)
	}
	slices.Sort(values)
	encoder.field("values", strconv.Itoa(len(values)))
	for _, value := range values {
		encoder.field("value", value)
	}
}

// rankingCacheEntry is one cached ranking and its bookkeeping.
type rankingCacheEntry struct {
	digest        string
	requestDigest string
	collection    string
	collectionID  int64
	ranking       collectionRanking
	lastUsed      time.Time
	bytes         int64
	// tokenIssued is true after a response returned a token for the entry.
	tokenIssued bool
}

// rankingCache stores collection rankings in process, one per ranking key. An
// entry expires RankingCacheTTL after the last request that stored or read it.
// Storing past maxBytes evicts the least recently used entries. The cache also
// counts committed writes per collection name. A ranking key includes that
// count. A write changes the key of every later request for the collection
// and removes the collection's stored rankings that no token names. A token
// reads its ranking until the ranking expires or is evicted. Concurrent
// requests that miss one key and count share one
// ranking computation.
type rankingCache struct {
	mutex       sync.Mutex
	clock       func() time.Time
	maxBytes    int64
	usedBytes   int64
	entries     map[string]*list.Element
	recency     *list.List
	generations map[string]uint64
	flights     map[string]*rankingFlight
}

// rankingFlight is one ranking computation that concurrent requests share.
// done closes after ranking and err are set.
type rankingFlight struct {
	done    chan struct{}
	ranking collectionRanking
	err     error
}

func newRankingCache(clock func() time.Time, maxBytes int64) *rankingCache {
	return &rankingCache{
		mutex:       sync.Mutex{},
		clock:       clock,
		maxBytes:    maxBytes,
		usedBytes:   0,
		entries:     make(map[string]*list.Element),
		recency:     list.New(),
		generations: make(map[string]uint64),
		flights:     make(map[string]*rankingFlight),
	}
}

// writeGeneration returns the committed write count of collectionName. A nil
// cache returns zero.
func (cache *rankingCache) writeGeneration(collectionName string) uint64 {
	if cache == nil {
		return 0
	}
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	return cache.generations[collectionName]
}

// noteWrite counts one committed write to collectionName and removes every
// stored ranking of collectionName that no token names. A nil cache ignores
// it.
func (cache *rankingCache) noteWrite(collectionName string) {
	if cache == nil {
		return
	}
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	cache.generations[collectionName]++
	for element := cache.recency.Front(); element != nil; {
		next := element.Next()
		entry, isEntry := element.Value.(*rankingCacheEntry)
		if isEntry && entry.collection == collectionName && !entry.tokenIssued {
			cache.removeLocked(element)
		}
		element = next
	}
}

// get returns the unexpired ranking stored under digest when it was computed
// over eligible rows. A stored ranking with a different eligible count is a
// miss: a process other than this daemon wrote rows after the ranking.
func (cache *rankingCache) get(digest string, eligible int64) (collectionRanking, bool) {
	var missing collectionRanking
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	element, found := cache.entries[digest]
	if !found {
		return missing, false
	}
	entry, isEntry := element.Value.(*rankingCacheEntry)
	if !isEntry {
		return missing, false
	}
	now := cache.clock()
	if now.Sub(entry.lastUsed) >= RankingCacheTTL {
		cache.removeLocked(element)
		return missing, false
	}
	if entry.ranking.Eligible != eligible {
		return missing, false
	}
	entry.lastUsed = now
	cache.recency.MoveToFront(element)
	return entry.ranking, true
}

// rank returns the cached ranking under digest for eligible rows, or runs
// compute once for every concurrent request that misses the same digest and
// count and stores its result under key.
func (cache *rankingCache) rank(key rankingKey, eligible int64, compute func() (collectionRanking, error)) (collectionRanking, error) {
	digest := key.digest()
	if ranking, cached := cache.get(digest, eligible); cached {
		return ranking, nil
	}
	flightKey := digest + ":" + strconv.FormatInt(eligible, 10)
	cache.mutex.Lock()
	existing, inFlight := cache.flights[flightKey]
	if inFlight {
		cache.mutex.Unlock()
		<-existing.done
		return existing.ranking, existing.err
	}
	flight := &rankingFlight{
		done:    make(chan struct{}),
		ranking: collectionRanking{Candidates: nil, Eligible: 0, Truncated: false, CallerState: ""},
		err:     nil,
	}
	cache.flights[flightKey] = flight
	cache.mutex.Unlock()

	flight.ranking, flight.err = compute()
	if flight.err == nil {
		cache.put(key, flight.ranking)
	}
	cache.mutex.Lock()
	delete(cache.flights, flightKey)
	cache.mutex.Unlock()
	close(flight.done)
	return flight.ranking, flight.err
}

// issueToken marks the ranking stored under digest as named by a token and
// returns the token. It returns an empty token when no ranking is stored under
// digest, for example after a write removed it or when it exceeded the bound.
func (cache *rankingCache) issueToken(digest string) string {
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	element, found := cache.entries[digest]
	if !found {
		return ""
	}
	entry, isEntry := element.Value.(*rankingCacheEntry)
	if !isEntry {
		return ""
	}
	entry.tokenIssued = true
	return rankingTokenPrefix + digest
}

// Reasons readToken gives for a token that reads no ranking.
const (
	tokenReasonMalformed = "malformed token"
	tokenReasonUnknown   = "no cached ranking for the token"
	tokenReasonExpired   = "the ranking expired"
	tokenReasonRecreated = "the collection was recreated"
	tokenReasonMismatch  = "the token belongs to another query"
)

// readToken returns the unexpired ranking that token names, whatever writes
// followed it, or an empty ranking and the reason no ranking was read: a
// malformed or unknown token, an expired ranking, a ranking of another
// collection ID, or a requestDigest that differs from the ranking's request
// digest. A read restarts the expiry period.
func (cache *rankingCache) readToken(token string, collectionID int64, requestDigest string) (collectionRanking, string) {
	var missing collectionRanking
	digest, prefixed := strings.CutPrefix(token, rankingTokenPrefix)
	if !prefixed {
		return missing, tokenReasonMalformed
	}
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	element, found := cache.entries[digest]
	if !found {
		return missing, tokenReasonUnknown
	}
	entry, isEntry := element.Value.(*rankingCacheEntry)
	if !isEntry {
		return missing, tokenReasonUnknown
	}
	now := cache.clock()
	if now.Sub(entry.lastUsed) >= RankingCacheTTL {
		cache.removeLocked(element)
		return missing, tokenReasonExpired
	}
	if entry.collectionID != collectionID {
		return missing, tokenReasonRecreated
	}
	if entry.requestDigest != requestDigest {
		return missing, tokenReasonMismatch
	}
	entry.lastUsed = now
	cache.recency.MoveToFront(element)
	return entry.ranking, ""
}

// put stores ranking under the digest of key, replacing an earlier entry, and
// evicts the least recently used entries until the cache fits maxBytes. A
// ranking larger than maxBytes is not stored. A ranking computed at a write
// generation older than the collection's current generation is not stored: a
// write committed while it was computed.
func (cache *rankingCache) put(key rankingKey, ranking collectionRanking) {
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	if key.WriteGeneration != cache.generations[key.CollectionName] {
		return
	}
	digest := key.digest()
	if element, found := cache.entries[digest]; found {
		cache.removeLocked(element)
	}
	bytes := ranking.estimatedBytes()
	if bytes > cache.maxBytes {
		slog.Warn("collection ranking exceeds the ranking cache bound and is not cached",
			"ranking_bytes", bytes, "cache_bytes", cache.maxBytes, "candidates", len(ranking.Candidates))
		return
	}
	entry := &rankingCacheEntry{
		digest:        digest,
		requestDigest: key.requestDigest(),
		collection:    key.CollectionName,
		collectionID:  key.CollectionID,
		ranking:       ranking,
		lastUsed:      cache.clock(),
		bytes:         bytes,
		tokenIssued:   false,
	}
	cache.entries[digest] = cache.recency.PushFront(entry)
	cache.usedBytes += bytes
	for cache.usedBytes > cache.maxBytes {
		oldest := cache.recency.Back()
		if oldest == nil {
			return
		}
		cache.removeLocked(oldest)
	}
}

func (cache *rankingCache) removeLocked(element *list.Element) {
	cache.recency.Remove(element)
	entry, isEntry := element.Value.(*rankingCacheEntry)
	if !isEntry {
		return
	}
	delete(cache.entries, entry.digest)
	cache.usedBytes -= entry.bytes
}
