package milvus

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"sync"

	"goodkind.io/lm-semantic-search/collection"
)

// rankReuseKey is the context key for a rank reuse scope.
type rankReuseKey struct{}

// vectorKeyEntry stores vector keys from one search in score order. Its complete field is true when the search returned fewer keys than its limit and the list includes every matching vector.
type vectorKeyEntry struct {
	keys     []scoredKey
	complete bool
}

// rankReuseScope stores one request's vector key lists by collection, filter expression, and query vector bytes. It also records whether each collection permits reuse.
type rankReuseScope struct {
	mutex    sync.Mutex
	entries  map[string]vectorKeyEntry
	reusable map[string]bool
}

// prefixStableIndexTypes contains FLAT, IVF_FLAT, and IVF_SQ8. Searches using these indexes select candidates independently of the result limit, and larger results share the smaller result's first keys unless scores tie at the cut.
var prefixStableIndexTypes = map[string]bool{"FLAT": true, "IVF_FLAT": true, "IVF_SQ8": true}

// WithRankReuse returns a context with a scope for one request. SharedStore.Rank can reuse a search within that scope when the collection, filter, and query vector match. Rank searches Milvus when the context has no scope.
func WithRankReuse(ctx context.Context) context.Context {
	return context.WithValue(ctx, rankReuseKey{}, &rankReuseScope{mutex: sync.Mutex{}, entries: map[string]vectorKeyEntry{}, reusable: map[string]bool{}})
}

func vectorKeyCacheKey(vectors string, expression string, vector []float32) string {
	digest := sha256.New()
	digest.Write([]byte(vectors))
	digest.Write([]byte{0})
	digest.Write([]byte(expression))
	digest.Write([]byte{0})
	buffer := make([]byte, 4)
	for _, value := range vector {
		binary.LittleEndian.PutUint32(buffer, math.Float32bits(value))
		digest.Write(buffer)
	}
	return hex.EncodeToString(digest.Sum(nil))
}

// lookup returns a copy of up to fetch keys when a saved list has at least fetch keys or is complete. It also reports whether the next key has the same score as the last returned key.
func (scope *rankReuseScope) lookup(key string, fetch int) ([]scoredKey, bool, bool) {
	scope.mutex.Lock()
	defer scope.mutex.Unlock()
	entry, found := scope.entries[key]
	if !found || (len(entry.keys) < fetch && !entry.complete) {
		return nil, false, false
	}
	return append([]scoredKey(nil), entry.keys[:min(fetch, len(entry.keys))]...), true, tiedAtCut(entry.keys, fetch)
}

// tiedAtCut reports whether the list has more than fetch keys and the key at position fetch has the same score as the preceding key.
func tiedAtCut(keys []scoredKey, fetch int) bool {
	return fetch > 0 && len(keys) > fetch && keys[fetch-1].score == keys[fetch].score
}

// store saves a list of vector keys under its key.
func (scope *rankReuseScope) store(key string, keys []scoredKey, complete bool) {
	scope.mutex.Lock()
	defer scope.mutex.Unlock()
	scope.entries[key] = vectorKeyEntry{keys: keys, complete: complete}
}

// reusableIndex reports whether a collection's dense index type appears in prefixStableIndexTypes. It reads the index definition from Milvus once per scope and saves the answer.
func (store *SharedStore) reusableIndex(ctx context.Context, scope *rankReuseScope, vectors string) (bool, error) {
	scope.mutex.Lock()
	reusable, known := scope.reusable[vectors]
	scope.mutex.Unlock()
	if known {
		return reusable, nil
	}
	definition, err := store.denseIndexDefinition(ctx, vectors)
	if err != nil {
		return false, err
	}
	reusable = prefixStableIndexTypes[definition.indexType]
	scope.mutex.Lock()
	scope.reusable[vectors] = reusable
	scope.mutex.Unlock()
	return reusable, nil
}

// rankedVectorKeys returns up to fetch keys in score order. With a scope and a reusable index, it reuses saved keys or saves a search for fetch times denseFetchGrowth keys, capped at collection.RankingDepth. Otherwise, or when scores tie at the cut, it searches for fetch keys.
func (store *SharedStore) rankedVectorKeys(ctx context.Context, vectors string, vector []float32, expression string, fetch int) ([]scoredKey, error) {
	scope, scoped := ctx.Value(rankReuseKey{}).(*rankReuseScope)
	if !scoped {
		return store.searchVectors(ctx, vectors, vector, expression, fetch)
	}
	reusable, err := store.reusableIndex(ctx, scope, vectors)
	if err != nil {
		return nil, err
	}
	if !reusable {
		return store.searchVectors(ctx, vectors, vector, expression, fetch)
	}
	key := vectorKeyCacheKey(vectors, expression, vector)
	cached, found, tied := scope.lookup(key, fetch)
	if found && !tied {
		return cached, nil
	}
	if found {
		return store.searchVectors(ctx, vectors, vector, expression, fetch)
	}
	limit := min(fetch*denseFetchGrowth, collection.RankingDepth)
	keys, err := store.searchVectors(ctx, vectors, vector, expression, limit)
	if err != nil {
		return nil, err
	}
	scope.store(key, keys, len(keys) < limit)
	if tiedAtCut(keys, fetch) {
		return store.searchVectors(ctx, vectors, vector, expression, fetch)
	}
	return append([]scoredKey(nil), keys[:min(fetch, len(keys))]...), nil
}
