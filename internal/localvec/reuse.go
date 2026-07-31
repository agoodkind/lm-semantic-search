package localvec

import (
	"context"
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// LoadReuseVectorsForContents resolves candidate contents across one complete
// collection without changing any row.
func (store *Store) LoadReuseVectorsForContents(
	ctx context.Context,
	collectionName string,
	chunks []model.StoredChunk,
) (map[string][]float32, error) {
	wanted := make(map[string]struct{}, len(chunks))
	for _, chunk := range chunks {
		wanted[semantic.ContentVectorKey(chunk.Content)] = struct{}{}
	}
	reuse := make(map[string][]float32, len(wanted))
	if err := operationContextError(ctx, "load local vector reuse contents"); err != nil {
		return nil, err
	}
	err := store.loadReuseWhere(
		collectionName,
		func(candidate row) bool {
			key := candidate.ContentVectorKey
			if key == "" {
				key = semantic.ContentVectorKey(candidate.Content)
			}
			_, found := wanted[key]
			return found
		},
		reuse,
	)
	return reuse, err
}

// LoadReuseVectors loads reusable vectors from collections.
func (store *Store) LoadReuseVectors(
	ctx context.Context,
	collectionNames []string,
) (map[string][]float32, error) {
	reuse := make(map[string][]float32)
	for _, collectionName := range collectionNames {
		if err := operationContextError(ctx, "load local vector reuse map"); err != nil {
			return nil, err
		}
		if err := store.loadReuseWhere(
			collectionName,
			func(row) bool { return true },
			reuse,
		); err != nil {
			return nil, err
		}
	}
	return reuse, nil
}

// LoadReuseVectorsForPrefix loads reusable vectors matching a relative path prefix.
func (store *Store) LoadReuseVectorsForPrefix(
	ctx context.Context,
	collectionName string,
	relativePathPrefix string,
) (map[string][]float32, error) {
	reuse := make(map[string][]float32)
	if collectionName == "" || relativePathPrefix == "" {
		return reuse, nil
	}
	if err := operationContextError(ctx, "load local vector reuse prefix"); err != nil {
		return nil, err
	}
	err := store.loadReuseWhere(
		collectionName,
		func(stored row) bool {
			return strings.HasPrefix(stored.RelativePath, relativePathPrefix)
		},
		reuse,
	)
	return reuse, err
}

// LoadReuseVectorsForPath loads reusable vectors matching a relative path.
func (store *Store) LoadReuseVectorsForPath(
	ctx context.Context,
	collectionName string,
	relativePath string,
) (map[string][]float32, error) {
	reuse := make(map[string][]float32)
	if collectionName == "" || relativePath == "" {
		return reuse, nil
	}
	if err := operationContextError(ctx, "load local vector reuse path"); err != nil {
		return nil, err
	}
	err := store.loadReuseWhere(
		collectionName,
		func(stored row) bool {
			return stored.RelativePath == relativePath
		},
		reuse,
	)
	return reuse, err
}

func (store *Store) loadReuseWhere(
	collectionName string,
	matches func(row) bool,
	reuse map[string][]float32,
) error {
	stored, err := store.collectionForName(collectionName, false)
	if err != nil {
		return err
	}
	rows, exists, err := stored.snapshot()
	if err != nil {
		return err
	}
	if !exists {
		return nil
	}
	for _, candidate := range rows {
		if !matches(candidate) {
			continue
		}
		key := candidate.ContentVectorKey
		if key == "" {
			key = semantic.ContentVectorKey(candidate.Content)
		}
		reuse[key] = append([]float32(nil), candidate.Vector...)
	}
	return nil
}
