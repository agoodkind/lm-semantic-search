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

// rankReuseScope stores one request's vector key lists by vector collection, filter expression, and query vector bytes.
type rankReuseScope struct {
	mutex   sync.Mutex
	entries map[string]vectorKeyEntry
}

// WithRankReuse returns a context with a scope for one request. SharedStore.Rank can reuse a search within that scope when the collection, filter, and query vector match. Rank searches Milvus when the context has no scope.
func WithRankReuse(ctx context.Context) context.Context {
	return context.WithValue(ctx, rankReuseKey{}, &rankReuseScope{mutex: sync.Mutex{}, entries: map[string]vectorKeyEntry{}})
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

// lookup returns a copy of the first fetch keys from a saved list when the list has at least fetch keys or is complete.
func (scope *rankReuseScope) lookup(key string, fetch int) ([]scoredKey, bool) {
	scope.mutex.Lock()
	defer scope.mutex.Unlock()
	entry, found := scope.entries[key]
	if !found || (len(entry.keys) < fetch && !entry.complete) {
		return nil, false
	}
	return append([]scoredKey(nil), entry.keys[:min(fetch, len(entry.keys))]...), true
}

// store saves a list of vector keys under its key.
func (scope *rankReuseScope) store(key string, keys []scoredKey, complete bool) {
	scope.mutex.Lock()
	defer scope.mutex.Unlock()
	scope.entries[key] = vectorKeyEntry{keys: keys, complete: complete}
}

// rankedVectorKeys returns the first fetch keys in score order for a query vector and filter. Without a scope it searches for fetch keys. With a scope it reuses a qualifying list or saves a search for fetch times denseFetchGrowth keys, capped at collection.RankingDepth.
func (store *SharedStore) rankedVectorKeys(ctx context.Context, vectors string, vector []float32, expression string, fetch int) ([]scoredKey, error) {
	scope, scoped := ctx.Value(rankReuseKey{}).(*rankReuseScope)
	if !scoped {
		return store.searchVectors(ctx, vectors, vector, expression, fetch)
	}
	key := vectorKeyCacheKey(vectors, expression, vector)
	if cached, found := scope.lookup(key, fetch); found {
		return cached, nil
	}
	limit := min(fetch*denseFetchGrowth, collection.RankingDepth)
	keys, err := store.searchVectors(ctx, vectors, vector, expression, limit)
	if err != nil {
		return nil, err
	}
	scope.store(key, keys, len(keys) < limit)
	return append([]scoredKey(nil), keys[:min(fetch, len(keys))]...), nil
}
