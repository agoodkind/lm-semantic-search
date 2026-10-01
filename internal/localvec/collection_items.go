package localvec

import (
	"context"
	"slices"
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// RecordCollectionDeclaration stores scalar definitions for local collection operations.
func (store *Store) RecordCollectionDeclaration(collectionName string, declaration model.CollectionDeclaration) {
	store.declaredScalars.Store(collectionName, slices.Clone(declaration.Scalars))
}

func (store *Store) recordedScalars(collectionName string) ([]model.ScalarColumn, bool) {
	loaded, found := store.declaredScalars.Load(collectionName)
	if !found {
		return nil, false
	}
	columns, isColumns := loaded.([]model.ScalarColumn)
	if !isColumns {
		return nil, false
	}
	return slices.Clone(columns), true
}

// LoadCollectionItemBatch reads the stored rows of itemIDs from a generic
// document collection. It selects rows by the value of the declared item id
// column itemColumn.
func (store *Store) LoadCollectionItemBatch(
	ctx context.Context,
	collectionName string,
	itemColumn string,
	itemIDs []string,
) (semantic.CollectionItemBatchState, error) {
	state := semantic.CollectionItemBatchState{
		Rows:  map[string]semantic.CollectionItemRows{},
		Reuse: map[string][]float32{},
	}
	requested := make(map[string]struct{}, len(itemIDs))
	for _, itemID := range itemIDs {
		trimmed := strings.TrimSpace(itemID)
		if trimmed != "" {
			requested[trimmed] = struct{}{}
		}
	}
	if collectionName == "" || itemColumn == "" || len(requested) == 0 {
		return state, nil
	}
	if err := operationContextError(ctx, "load local collection items"); err != nil {
		return semantic.CollectionItemBatchState{}, err
	}
	stored, err := store.collectionForName(collectionName, false)
	if err != nil {
		return semantic.CollectionItemBatchState{}, err
	}
	rows, exists, err := stored.snapshot()
	if err != nil {
		return semantic.CollectionItemBatchState{}, err
	}
	if !exists {
		return state, nil
	}
	for _, candidate := range rows {
		itemID, present := candidate.itemID(itemColumn)
		if !present {
			continue
		}
		if _, wanted := requested[itemID]; !wanted {
			continue
		}
		contentKey := candidate.ContentVectorKey
		if contentKey == "" {
			contentKey = semantic.ContentVectorKey(candidate.Content)
		}
		state.Reuse[contentKey] = append([]float32(nil), candidate.Vector...)
		itemRows, found := state.Rows[itemID]
		if !found {
			itemRows = semantic.CollectionItemRows{UsablePaths: map[string]struct{}{}}
			state.Rows[itemID] = itemRows
		}
		if strings.TrimSpace(candidate.Content) != "" {
			itemRows.UsablePaths[candidate.RelativePath] = struct{}{}
		}
	}
	return state, nil
}
