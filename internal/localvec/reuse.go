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
) (map[string]model.ReuseVector, error) {
	wanted := make(map[string]struct{}, len(chunks))
	for _, chunk := range chunks {
		wanted[semantic.ContentVectorKey(chunk.Content)] = struct{}{}
	}
	if err := operationContextError(ctx, "load local vector reuse contents"); err != nil {
		return nil, err
	}
	stored, err := store.collectionForName(collectionName, false)
	if err != nil {
		return nil, err
	}
	return stored.reuseVectorsForKeys(wanted)
}

// LoadReuseVectorsForContentsFrom reads only the listed collections in order.
// The first collection with a matching vector supplies the vector.
func (store *Store) LoadReuseVectorsForContentsFrom(
	ctx context.Context,
	collectionNames []string,
	chunks []model.StoredChunk,
) (map[string]model.ReuseVector, error) {
	wanted := make(map[string]struct{}, len(chunks))
	for _, chunk := range chunks {
		wanted[semantic.ContentVectorKey(chunk.Content)] = struct{}{}
	}
	reuse := make(map[string]model.ReuseVector, len(wanted))
	for _, collectionName := range collectionNames {
		if collectionName == "" || len(wanted) == 0 {
			continue
		}
		if err := operationContextError(ctx, "load local vector reuse contents"); err != nil {
			return nil, err
		}
		stored, err := store.collectionForName(collectionName, false)
		if err != nil {
			return nil, err
		}
		found, err := stored.reuseVectorsForKeys(wanted)
		if err != nil {
			return nil, err
		}
		for key, vector := range found {
			reuse[key] = vector
			delete(wanted, key)
		}
	}
	return reuse, nil
}

// LoadReuseVectors loads reusable vectors from collections.
func (store *Store) LoadReuseVectors(
	ctx context.Context,
	collectionNames []string,
) (map[string]model.ReuseVector, error) {
	reuse := make(map[string]model.ReuseVector)
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
) (map[string]model.ReuseVector, error) {
	reuse := make(map[string]model.ReuseVector)
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
) (map[string]model.ReuseVector, error) {
	reuse := make(map[string]model.ReuseVector)
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
	reuse map[string]model.ReuseVector,
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
		reuse[key] = model.ReuseVector{
			Vector:           append([]float32(nil), candidate.Vector...),
			SourceModelKnown: false,
		}
	}
	return nil
}
