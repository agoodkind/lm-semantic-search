package milvus

import (
	"context"
	"errors"
	"io"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

func (store *SharedStore) occurrenceKeysByID(
	ctx context.Context,
	occurrences string,
	ids []string,
) ([]string, error) {
	keyByID, err := store.occurrenceKeyMap(ctx, occurrences, ids)
	if err != nil {
		return nil, err
	}

	keySet := map[string]struct{}{}

	for _, key := range keyByID {
		keySet[key] = struct{}{}
	}

	return sortedKeys(keySet), nil
}

func (store *SharedStore) occurrenceKeyMap(
	ctx context.Context,
	occurrences string,
	ids []string,
) (map[string]string, error) {
	keyByID := make(map[string]string, len(ids))
	for start := 0; start < len(ids); start += vectorKeyBatchSize {
		batch := ids[start:min(start+vectorKeyBatchSize, len(ids))]
		resultSet, err := store.inner.client.Get(
			ctx,
			milvusclient.NewQueryOption(occurrences).WithIDs(primaryKeyColumn(IDField, batch)).
				WithOutputFields(
					IDField,
					VectorKeyField,
				).
				WithConsistencyLevel(
					entity.ClStrong,
				),
		)
		if err != nil {
			return nil, WrapError(ctx, err, "read vector keys of "+occurrences)
		}

		if resultSet.ResultCount == 0 {
			continue
		}

		idColumn := resultSet.GetColumn(IDField)
		keyColumn := resultSet.GetColumn(VectorKeyField)
		if idColumn == nil || keyColumn == nil {
			return nil, collection.ErrSearchResultIncomplete
		}

		for rowIndex := range resultSet.ResultCount {
			rowID, err := idColumn.GetAsString(rowIndex)
			if err != nil {
				return nil, rowReadError(IDField, rowIndex, err)
			}

			key, err := keyColumn.GetAsString(rowIndex)
			if err != nil {
				return nil, rowReadError(VectorKeyField, rowIndex, err)
			}

			keyByID[rowID] = key
		}
	}

	return keyByID, nil
}

// storedSupersets reads the supersets of the stored vector rows among keys.
// The result omits a key without a stored row.
func (store *SharedStore) storedSupersets(
	ctx context.Context,
	vectors string,
	scalars []collection.ScalarColumn,
	keys []string,
) (map[string]*vectorRecord, error) {
	stored := make(map[string]*vectorRecord, len(keys))
	outputFields := append([]string{VectorKeyField}, supersetOutputFields(scalars)...)

	for start := 0; start < len(keys); start += vectorKeyBatchSize {
		batch := keys[start:min(start+vectorKeyBatchSize, len(keys))]
		resultSet, err := store.inner.client.Get(
			ctx,
			milvusclient.NewQueryOption(vectors).WithIDs(primaryKeyColumn(VectorKeyField, batch)).
				WithOutputFields(
					outputFields...,
				).
				WithConsistencyLevel(
					entity.ClStrong,
				),
		)
		if err != nil {
			return nil, WrapError(ctx, err, "read vector supersets of "+vectors)
		}

		if resultSet.ResultCount == 0 {
			continue
		}

		keyColumn := resultSet.GetColumn(VectorKeyField)
		if keyColumn == nil {
			return nil, collection.ErrSearchResultIncomplete
		}

		for rowIndex := range resultSet.ResultCount {
			key, err := keyColumn.GetAsString(rowIndex)
			if err != nil {
				return nil, rowReadError(VectorKeyField, rowIndex, err)
			}

			record := newVectorRecord(key, nil, scalars)
			if err := supersetsFromResult(resultSet, rowIndex, scalars, record); err != nil {
				return nil, err
			}

			stored[key] = record
		}
	}

	return stored, nil
}

func (store *SharedStore) vectorsByKey(
	ctx context.Context,
	vectors string,
	keys []string,
) (map[string][]float32, error) {
	vectorByKey := make(map[string][]float32, len(keys))
	for start := 0; start < len(keys); start += vectorKeyBatchSize {
		batch := keys[start:min(start+vectorKeyBatchSize, len(keys))]
		resultSet, err := store.inner.client.Get(
			ctx,
			milvusclient.NewQueryOption(vectors).WithIDs(primaryKeyColumn(VectorKeyField, batch)).
				WithOutputFields(
					VectorKeyField,
					DenseVectorField,
				).
				WithConsistencyLevel(
					entity.ClStrong,
				),
		)
		if err != nil {
			return nil, WrapError(ctx, err, "read vectors of "+vectors)
		}

		if resultSet.ResultCount == 0 {
			continue
		}

		keyColumn := resultSet.GetColumn(VectorKeyField)
		vectorColumn := resultSet.GetColumn(DenseVectorField)
		if keyColumn == nil || vectorColumn == nil {
			return nil, collection.ErrSearchResultIncomplete
		}

		for rowIndex := range resultSet.ResultCount {
			key, err := keyColumn.GetAsString(rowIndex)
			if err != nil {
				return nil, rowReadError(VectorKeyField, rowIndex, err)
			}

			vector, err := VectorAt(vectorColumn, rowIndex)
			if err != nil {
				return nil, err
			}

			vectorByKey[key] = vector
		}
	}

	return vectorByKey, nil
}

// iterateVectorKeys returns the distinct vector keys of the occurrences an
// expression matches.
func (store *SharedStore) iterateVectorKeys(
	ctx context.Context,
	occurrences string,
	expression string,
) ([]string, error) {
	keys := map[string]struct{}{}
	err := store.iterate(
		ctx,
		occurrences,
		expression,
		[]string{VectorKeyField},
		func(resultSet milvusclient.ResultSet) error {
			keyColumn := resultSet.GetColumn(VectorKeyField)
			if keyColumn == nil {
				return collection.ErrSearchResultIncomplete
			}

			for rowIndex := range resultSet.ResultCount {
				key, err := keyColumn.GetAsString(rowIndex)
				if err != nil {
					return rowReadError(VectorKeyField, rowIndex, err)
				}

				keys[key] = struct{}{}
			}

			return nil
		},
	)
	if err != nil {
		return nil, err
	}

	return sortedKeys(keys), nil
}

// iterate pages through every row an expression matches with strong
// consistency.
func (store *SharedStore) iterate(
	ctx context.Context,
	collectionName string,
	expression string,
	outputFields []string,
	page func(milvusclient.ResultSet) error,
) error {
	return store.iterateWithBatch(
		ctx,
		collectionName,
		expression,
		outputFields,
		sharedIteratorBatch,
		page,
	)
}

func (store *SharedStore) iterateWithBatch(
	ctx context.Context,
	collectionName string,
	expression string,
	outputFields []string,
	batchSize int,
	page func(milvusclient.ResultSet) error,
) error {
	iterator, err := store.inner.client.QueryIterator(
		ctx,
		milvusclient.NewQueryIteratorOption(collectionName).WithBatchSize(batchSize).
			WithFilter(
				expression,
			).
			WithOutputFields(
				outputFields...,
			).
			WithConsistencyLevel(
				entity.ClStrong,
			),
	)
	if err != nil {
		return SearchError(ctx, "open iterator for", collectionName, err)
	}

	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			return nil
		}

		if nextErr != nil {
			return SearchError(ctx, "iterate", collectionName, nextErr)
		}

		if resultSet.ResultCount == 0 {
			continue
		}

		if err := page(resultSet); err != nil {
			return err
		}
	}
}
