package milvus

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"slices"
	"sync"

	"goodkind.io/lm-semantic-search/collection"
)

// rankReuseKey is the context key for a rank reuse scope.
type rankReuseKey struct{}

// rankReuseEntryKey identifies a scoped vector search by SharedStore, a key built from the collection, filter expression and query vector bytes, and the result limit. Different stores or limits use separate entries.
type rankReuseEntryKey struct {
	store *SharedStore
	key   string
	limit int
}

// vectorKeyEntry stores the result of one scoped vector search. Its done channel closes after the search sets keys and err.
type vectorKeyEntry struct {
	done chan struct{}
	keys []scoredKey
	err  error
}

// rankReuseScope tracks vector searches for one request. Background searches use ctx, a child of the request context that cancel ends. background counts running background searches, and speculated records queries that started one.
type rankReuseScope struct {
	ctx        context.Context
	cancel     context.CancelFunc
	background sync.WaitGroup
	mutex      sync.Mutex
	entries    map[rankReuseEntryKey]*vectorKeyEntry
	speculated map[rankReuseEntryKey]bool
}

// WithRankReuse returns a context and an end function for one request. SharedStore.Rank reuses searches with the same store, collection, filter, query vector, and limit. Call end when the request finishes to cancel and await background searches. Without a scope, Rank searches Milvus on every call.
func WithRankReuse(ctx context.Context) (context.Context, func()) {
	scopeCtx, cancel := context.WithCancel(ctx)
	scope := &rankReuseScope{
		ctx:        scopeCtx,
		cancel:     cancel,
		background: sync.WaitGroup{},
		mutex:      sync.Mutex{},
		entries:    map[rankReuseEntryKey]*vectorKeyEntry{},
		speculated: map[rankReuseEntryKey]bool{},
	}
	end := func() {
		cancel()
		scope.background.Wait()
	}
	return context.WithValue(ctx, rankReuseKey{}, scope), end
}

// firstSearch reports whether the scope remains open and key has not started a background search. It records key when both conditions hold.
func (scope *rankReuseScope) firstSearch(key rankReuseEntryKey) bool {
	scope.mutex.Lock()
	defer scope.mutex.Unlock()
	if scope.speculated[key] || scope.ctx.Err() != nil {
		return false
	}
	scope.speculated[key] = true
	return true
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

// claim returns the entry for key and reports whether the caller created it and must run its search.
func (scope *rankReuseScope) claim(key rankReuseEntryKey) (*vectorKeyEntry, bool) {
	scope.mutex.Lock()
	defer scope.mutex.Unlock()
	if entry, found := scope.entries[key]; found {
		return entry, false
	}
	entry := &vectorKeyEntry{done: make(chan struct{}), keys: nil, err: nil}
	scope.entries[key] = entry
	return entry, true
}

// searchInto runs one vector search, stores its keys and error in entry, and closes entry.done.
func (store *SharedStore) searchInto(ctx context.Context, entry *vectorKeyEntry, vectors string, vector []float32, expression string, limit int) {
	entry.keys, entry.err = store.searchVectors(ctx, vectors, vector, expression, limit)
	close(entry.done)
}

// rankedVectorKeys returns the best fetch vector keys in score order. Without a scope, it searches Milvus. With a scope, it starts one background search per query for fetch times denseFetchGrowth keys, capped at collection.RankingDepth. It reuses exact limits and searches directly if reuse fails.
func (store *SharedStore) rankedVectorKeys(ctx context.Context, vectors string, vector []float32, expression string, fetch int) ([]scoredKey, error) {
	scope, scoped := ctx.Value(rankReuseKey{}).(*rankReuseScope)
	if !scoped {
		return store.searchVectors(ctx, vectors, vector, expression, fetch)
	}
	base := vectorKeyCacheKey(vectors, expression, vector)
	next := min(fetch*denseFetchGrowth, collection.RankingDepth)
	if next > fetch && scope.firstSearch(rankReuseEntryKey{store: store, key: base, limit: 0}) {
		if speculative, owner := scope.claim(rankReuseEntryKey{store: store, key: base, limit: next}); owner {
			// The background search reads a vector copy made before the goroutine starts because it can continue after Rank returns and the caller reuses the buffer.
			owned := slices.Clone(vector)
			scope.background.Add(1)
			go func() {
				defer scope.background.Done()
				defer func() {
					if recovered := recover(); recovered != nil {
						speculative.err = fmt.Errorf("background vector search of %s panicked: %v", vectors, recovered)
						close(speculative.done)
					}
				}()
				store.searchInto(scope.ctx, speculative, vectors, owned, expression, next)
			}()
		}
	}
	entry, owner := scope.claim(rankReuseEntryKey{store: store, key: base, limit: fetch})
	if owner {
		store.searchInto(ctx, entry, vectors, vector, expression, fetch)
	} else {
		select {
		case <-entry.done:
		case <-ctx.Done():
			return nil, SearchError(ctx, "dense vector search", vectors, ctx.Err())
		}
	}
	if entry.err != nil {
		if owner {
			return nil, entry.err
		}
		return store.searchVectors(ctx, vectors, vector, expression, fetch)
	}
	return append([]scoredKey(nil), entry.keys...), nil
}
