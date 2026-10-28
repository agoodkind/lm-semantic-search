package milvus

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"sort"
	"strings"
	"sync"

	"github.com/milvus-io/milvus/client/v3/column"
	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

const (
	// vectorKeyBatchSize bounds the vector keys one expression lists.
	vectorKeyBatchSize = 512
	// sharedWriteBatchSize bounds the rows one shared upsert request writes.
	sharedWriteBatchSize = 256
	sharedIteratorBatch  = 1000
)

// SharedStore implements [collection.Store] on two Milvus collections. The
// occurrence collection stores every row with every column except the dense
// vector, plus the row's vector key. The vector collection stores each distinct
// dense vector once under its vector key, with the superset of each declared
// scalar column across the vector's occurrences. One SharedStore serializes its
// writes. Two processes must not write the same collection pair.
type SharedStore struct {
	inner *Store
	write sync.Mutex
}

var _ collection.Store = (*SharedStore)(nil)

// NewShared returns a SharedStore over client. The options must set Hybrid,
// because the occurrence collection keeps the BM25 sparse vector as its only
// vector field.
func NewShared(client *milvusclient.Client, options Options) (*SharedStore, error) {
	if !options.Hybrid {
		return nil, errors.New("a shared vector store requires the hybrid BM25 option")
	}
	return &SharedStore{inner: New(client, options), write: sync.Mutex{}}, nil
}

// Inner returns the single-collection store that runs occurrence reads.
func (store *SharedStore) Inner() *Store {
	return store.inner
}

// EnsureCollection creates each absent collection of the occurrence and vector
// pair with mmap enabled on every field and index. It fails when the
// declaration adds a scalar column the stored occurrence collection lacks.
func (store *SharedStore) EnsureCollection(ctx context.Context, request collection.EnsureRequest) error {
	occurrences := strings.TrimSpace(request.Collection)
	if occurrences == "" {
		return errors.New("collection name is required")
	}
	vectors := VectorCollectionName(occurrences)
	client := store.inner.client
	occurrencesExist, err := client.HasCollection(ctx, milvusclient.NewHasCollectionOption(occurrences))
	if err != nil {
		return WrapError(ctx, err, "check Milvus collection "+occurrences)
	}
	vectorsExist, err := client.HasCollection(ctx, milvusclient.NewHasCollectionOption(vectors))
	if err != nil {
		return WrapError(ctx, err, "check Milvus collection "+vectors)
	}
	if !vectorsExist {
		if request.Dimension <= 0 {
			return fmt.Errorf("create shared collection %s: dimension %d is not positive", occurrences, request.Dimension)
		}
		if err := client.CreateCollection(ctx, store.inner.vectorCollectionOption(vectors, request.Dimension, request.Declaration.Scalars)); err != nil {
			return WrapError(ctx, err, "create Milvus collection "+vectors)
		}
		if err := store.enableMmap(ctx, vectors); err != nil {
			return err
		}
	}
	if !occurrencesExist {
		if err := client.CreateCollection(ctx, store.inner.occurrenceCollectionOption(occurrences, request.Declaration.Scalars)); err != nil {
			return WrapError(ctx, err, "create Milvus collection "+occurrences)
		}
		return store.enableMmap(ctx, occurrences)
	}
	described, err := client.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(occurrences))
	if err != nil {
		return WrapError(ctx, err, "describe Milvus collection "+occurrences)
	}
	stored := map[string]struct{}{}
	if described.Schema != nil {
		for _, field := range described.Schema.Fields {
			stored[field.Name] = struct{}{}
		}
	}
	for _, declared := range request.Declaration.Scalars {
		if _, found := stored[declared.Name]; !found {
			return fmt.Errorf("shared collection %s lacks declared column %s; a shared collection cannot add a column after creation", occurrences, declared.Name)
		}
	}
	return nil
}

// Upsert writes rows. It merges each row's scalar values into the superset of
// its vector, writes the vector rows before the occurrence rows, and then
// deletes every vector the replaced rows referenced and no row references now.
func (store *SharedStore) Upsert(ctx context.Context, collectionName string, declaration collection.Declaration, rows []collection.Row) error {
	if len(rows) == 0 {
		return nil
	}
	store.write.Lock()
	defer store.write.Unlock()
	return store.writeRows(ctx, strings.TrimSpace(collectionName), declaration, rows, computedIdentity(rows, store.inner.options.EmbeddingModel))
}

// writeRows writes rows with the given reuse identities. The caller must lock
// store.write first.
func (store *SharedStore) writeRows(ctx context.Context, occurrences string, declaration collection.Declaration, rows []collection.Row, identity rowIdentity) error {
	vectors := VectorCollectionName(occurrences)
	dimension := len(rows[0].Vector)
	if dimension == 0 {
		return fmt.Errorf("upsert into %s: row %s has no vector", occurrences, rows[0].ID)
	}
	ids := make([]string, 0, len(rows))
	keys := make([]string, 0, len(rows))
	records := map[string]*vectorRecord{}
	for _, row := range rows {
		if len(row.Vector) != dimension {
			return fmt.Errorf("upsert into %s: row %s has %d dimensions, want %d", occurrences, row.ID, len(row.Vector), dimension)
		}
		key := VectorKey(row.Vector)
		ids = append(ids, row.ID)
		keys = append(keys, key)
		record, found := records[key]
		if !found {
			record = newVectorRecord(key, row.Vector, declaration.Scalars)
			records[key] = record
		}
		record.addRow(declaration.Scalars, row.Scalars)
	}
	replacedKeys, err := store.occurrenceKeysByID(ctx, occurrences, ids)
	if err != nil {
		return err
	}
	unchanged, err := store.mergeStoredSupersets(ctx, vectors, declaration.Scalars, records)
	if err != nil {
		return err
	}
	ordered := make([]*vectorRecord, 0, len(records))
	for _, key := range sortedRecordKeys(records) {
		if _, skip := unchanged[key]; skip {
			continue
		}
		ordered = append(ordered, records[key])
	}
	for start := 0; start < len(ordered); start += sharedWriteBatchSize {
		batch := ordered[start:min(start+sharedWriteBatchSize, len(ordered))]
		option, err := vectorUpsertOption(vectors, dimension, declaration.Scalars, batch)
		if err != nil {
			return err
		}
		if _, err := store.inner.client.Upsert(ctx, option); err != nil {
			return WrapError(ctx, err, "upsert Milvus vectors into "+vectors)
		}
	}
	for start := 0; start < len(rows); start += sharedWriteBatchSize {
		end := min(start+sharedWriteBatchSize, len(rows))
		option, err := buildOccurrenceWriteOption(ctx, occurrences, rows[start:end], keys[start:end], declaration.Scalars, identity.slice(start, end))
		if err != nil {
			return err
		}
		if _, err := store.inner.client.Upsert(ctx, option); err != nil {
			return WrapError(ctx, err, "upsert Milvus occurrences into "+occurrences)
		}
	}
	stale := make([]string, 0, len(replacedKeys))
	for _, key := range replacedKeys {
		if _, written := records[key]; !written {
			stale = append(stale, key)
		}
	}
	return store.deleteUnreferencedVectors(ctx, occurrences, vectors, stale)
}

// Delete removes the occurrences a filter matches and every vector that no
// remaining occurrence references.
func (store *SharedStore) Delete(ctx context.Context, collectionName string, filter collection.Filter) (int64, error) {
	expression, err := collection.CompileInline(&filter)
	if err != nil {
		slog.ErrorContext(ctx, "compile collection filter failed", "collection", collectionName, "err", err)
		return 0, fmt.Errorf("compile filter for %s: %w", collectionName, err)
	}
	store.write.Lock()
	defer store.write.Unlock()
	return store.deleteOccurrences(ctx, strings.TrimSpace(collectionName), []string{expression})
}

// DeleteItems removes the occurrences of the requested items and every vector
// that no remaining occurrence references.
func (store *SharedStore) DeleteItems(ctx context.Context, request collection.DeleteItemsRequest) (int64, error) {
	occurrences := strings.TrimSpace(request.Collection)
	if occurrences == "" {
		return 0, errors.New("collection name is required")
	}
	if len(request.ItemIDs) == 0 && len(request.PathPrefixes) == 0 {
		return 0, errors.New("delete items selects no item and no path prefix")
	}
	if len(request.ItemIDs) > 0 && request.Declaration.ItemIDColumn == "" {
		return 0, fmt.Errorf("delete items from %s: declaration has no item ID column", occurrences)
	}
	expressions := make([]string, 0)
	for start := 0; start < len(request.ItemIDs); start += deleteItemIDBatchSize {
		end := min(start+deleteItemIDBatchSize, len(request.ItemIDs))
		expressions = append(expressions, itemIDsExpression(request.Declaration.ItemIDColumn, request.ItemIDs[start:end]))
	}
	for _, prefix := range request.PathPrefixes {
		if prefix == "" {
			continue
		}
		expressions = append(expressions, fmt.Sprintf(`%s like "%s%%"`, RelativePathField, escapeLikePattern(prefix)))
	}
	store.write.Lock()
	defer store.write.Unlock()
	return store.deleteOccurrences(ctx, occurrences, expressions)
}

func (store *SharedStore) deleteOccurrences(ctx context.Context, occurrences string, expressions []string) (int64, error) {
	vectors := VectorCollectionName(occurrences)
	affectedKeys := map[string]struct{}{}
	var removed int64
	for _, expression := range expressions {
		keys, err := store.iterateVectorKeys(ctx, occurrences, expression)
		if err != nil {
			return removed, err
		}
		for _, key := range keys {
			affectedKeys[key] = struct{}{}
		}
		result, err := store.inner.client.Delete(ctx, milvusclient.NewDeleteOption(occurrences).WithExpr(expression))
		if err != nil {
			return removed, WrapError(ctx, err, "delete from "+occurrences)
		}
		removed += result.DeleteCount
	}
	if err := store.deleteUnreferencedVectors(ctx, occurrences, vectors, sortedKeys(affectedKeys)); err != nil {
		return removed, err
	}
	return removed, nil
}

// Query returns occurrence rows a filter matches without ranking.
func (store *SharedStore) Query(ctx context.Context, request collection.QueryRequest) ([]collection.Hit, error) {
	return store.inner.Query(ctx, request)
}

// QueryRows returns the stored occurrence rows of the requested items. With
// IncludeVector, each row reads its dense vector from the vector collection.
func (store *SharedStore) QueryRows(ctx context.Context, request collection.RowsRequest) ([]collection.StoredRow, error) {
	withoutVector := request
	withoutVector.IncludeVector = false
	rows, err := store.inner.QueryRows(ctx, withoutVector)
	if err != nil || !request.IncludeVector || len(rows) == 0 {
		return rows, err
	}
	occurrences := strings.TrimSpace(request.Collection)
	ids := make([]string, 0, len(rows))
	for _, row := range rows {
		ids = append(ids, row.ID)
	}
	keyByID, err := store.occurrenceKeyMap(ctx, occurrences, ids)
	if err != nil {
		return nil, err
	}
	keySet := map[string]struct{}{}
	for _, key := range keyByID {
		keySet[key] = struct{}{}
	}
	vectorByKey, err := store.vectorsByKey(ctx, VectorCollectionName(occurrences), sortedKeys(keySet))
	if err != nil {
		return nil, err
	}
	for index := range rows {
		key, found := keyByID[rows[index].ID]
		if !found {
			return nil, fmt.Errorf("query rows of %s: row %s has no vector key", occurrences, rows[index].ID)
		}
		vector, found := vectorByKey[key]
		if !found {
			return nil, fmt.Errorf("query rows of %s: row %s references missing vector %s", occurrences, rows[index].ID, key)
		}
		rows[index].Vector = slices.Clone(vector)
	}
	return rows, nil
}

// BackfillScalars fills null or empty declared columns of occurrence rows and
// then rebuilds the supersets of every vector the streamed items reference.
func (store *SharedStore) BackfillScalars(ctx context.Context, collectionName string, backfill collection.ScalarBackfill) (int, int, error) {
	store.write.Lock()
	defer store.write.Unlock()
	occurrences := strings.TrimSpace(collectionName)
	changed, orphan, err := store.inner.BackfillScalars(ctx, occurrences, backfill)
	if err != nil || backfill.DryRun || changed == 0 {
		return changed, orphan, err
	}
	itemIDs := make([]string, 0, len(backfill.Values))
	for itemID := range backfill.Values {
		itemIDs = append(itemIDs, itemID)
	}
	sort.Strings(itemIDs)
	expressions := make([]string, 0)
	for start := 0; start < len(itemIDs); start += deleteItemIDBatchSize {
		end := min(start+deleteItemIDBatchSize, len(itemIDs))
		expressions = append(expressions, itemIDsExpression(backfill.ItemColumn, itemIDs[start:end]))
	}
	for _, family := range backfill.LegacyPathFamilies {
		for _, itemID := range itemIDs {
			expressions = append(expressions, fmt.Sprintf(`%s like "%s%%"`, RelativePathField, escapeLikePattern(family+itemID+"/")))
		}
	}
	affectedKeys := map[string]struct{}{}
	for _, expression := range expressions {
		keys, keyErr := store.iterateVectorKeys(ctx, occurrences, expression)
		if keyErr != nil {
			return changed, orphan, keyErr
		}
		for _, key := range keys {
			affectedKeys[key] = struct{}{}
		}
	}
	declaration, err := store.storedDeclaration(ctx, occurrences, backfill.ItemColumn)
	if err != nil {
		return changed, orphan, err
	}
	return changed, orphan, store.rebuildSupersets(ctx, occurrences, declaration.Scalars, sortedKeys(affectedKeys))
}

// storedDeclaration reads the declared scalar columns of the occurrence
// collection from its schema. Every field outside the built-in columns and the
// vector key is a declared scalar.
func (store *SharedStore) storedDeclaration(ctx context.Context, occurrences string, itemColumn string) (collection.Declaration, error) {
	described, err := store.inner.client.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(occurrences))
	if err != nil {
		return collection.Declaration{ItemIDColumn: itemColumn, Scalars: nil}, WrapError(ctx, err, "describe Milvus collection "+occurrences)
	}
	builtin := map[string]struct{}{VectorKeyField: {}}
	for _, name := range BuiltinColumnNames() {
		builtin[name] = struct{}{}
	}
	declaration := collection.Declaration{ItemIDColumn: itemColumn, Scalars: nil}
	if described.Schema == nil {
		return declaration, nil
	}
	scalarTypes := map[entity.FieldType]collection.ScalarType{
		entity.FieldTypeVarChar: collection.ScalarTypeString,
		entity.FieldTypeBool:    collection.ScalarTypeBool,
		entity.FieldTypeInt64:   collection.ScalarTypeInt64,
	}
	for _, field := range described.Schema.Fields {
		if _, isBuiltin := builtin[field.Name]; isBuiltin {
			continue
		}
		scalarType, supported := scalarTypes[field.DataType]
		if !supported {
			return declaration, fmt.Errorf("occurrence collection %s field %s has unsupported type %s", occurrences, field.Name, field.DataType.Name())
		}
		declaration.Scalars = append(declaration.Scalars, collection.ScalarColumn{Name: field.Name, Type: scalarType, Nullable: field.Nullable, MaxLength: 0})
	}
	return declaration, nil
}

// rebuildSupersets recomputes the supersets of keys from every occurrence that
// references them and rewrites those vector rows with their stored vectors.
func (store *SharedStore) rebuildSupersets(ctx context.Context, occurrences string, scalars []collection.ScalarColumn, keys []string) error {
	vectors := VectorCollectionName(occurrences)
	for start := 0; start < len(keys); start += vectorKeyBatchSize {
		batch := keys[start:min(start+vectorKeyBatchSize, len(keys))]
		vectorByKey, err := store.vectorsByKey(ctx, vectors, batch)
		if err != nil {
			return err
		}
		records := make(map[string]*vectorRecord, len(batch))
		dimension := 0
		for _, key := range batch {
			vector, found := vectorByKey[key]
			if !found {
				return fmt.Errorf("rebuild supersets of %s: vector %s is missing", vectors, key)
			}
			dimension = len(vector)
			records[key] = newVectorRecord(key, vector, scalars)
		}
		outputFields := []string{VectorKeyField}
		for _, declared := range scalars {
			outputFields = append(outputFields, declared.Name)
		}
		err = store.iterate(ctx, occurrences, vectorKeysExpression(batch), outputFields, func(resultSet milvusclient.ResultSet) error {
			return addOccurrenceCells(resultSet, scalars, records)
		})
		if err != nil {
			return err
		}
		ordered := make([]*vectorRecord, 0, len(records))
		for _, key := range sortedRecordKeys(records) {
			ordered = append(ordered, records[key])
		}
		option, err := vectorUpsertOption(vectors, dimension, scalars, ordered)
		if err != nil {
			return err
		}
		if _, err := store.inner.client.Upsert(ctx, option); err != nil {
			return WrapError(ctx, err, "upsert Milvus vectors into "+vectors)
		}
	}
	return nil
}

// Search runs a typed search and returns at most Limit hits with the same
// selection rules as [Store.Search].
func (store *SharedStore) Search(ctx context.Context, request collection.SearchRequest) ([]collection.Hit, error) {
	candidates, err := store.Rank(ctx, request)
	if err != nil {
		return nil, err
	}
	_, grouped := GroupColumnFor(request)
	perGroupLimit := int32(0)
	if grouped {
		perGroupLimit = request.PerGroupLimit
	}
	limit := request.Limit
	if limit <= 0 {
		limit = defaultSearchLimit
	}
	selected := SelectCandidates(candidates, perGroupLimit, request.MinScore, limit)
	return store.inner.Load(ctx, request.Collection, selected, request.Declaration.Scalars)
}

// Load reads the selected occurrence rows in selection order.
func (store *SharedStore) Load(ctx context.Context, collectionName string, selected []Candidate, scalarColumns []collection.ScalarColumn) ([]collection.Hit, error) {
	return store.inner.Load(ctx, collectionName, selected, scalarColumns)
}

func (store *SharedStore) occurrenceKeysByID(ctx context.Context, occurrences string, ids []string) ([]string, error) {
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

func (store *SharedStore) occurrenceKeyMap(ctx context.Context, occurrences string, ids []string) (map[string]string, error) {
	keyByID := make(map[string]string, len(ids))
	for start := 0; start < len(ids); start += vectorKeyBatchSize {
		batch := ids[start:min(start+vectorKeyBatchSize, len(ids))]
		resultSet, err := store.inner.client.Get(ctx, milvusclient.NewQueryOption(occurrences).
			WithIDs(primaryKeyColumn(IDField, batch)).
			WithOutputFields(IDField, VectorKeyField).
			WithConsistencyLevel(entity.ClStrong))
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
			id, err := idColumn.GetAsString(rowIndex)
			if err != nil {
				return nil, rowReadError(IDField, rowIndex, err)
			}
			key, err := keyColumn.GetAsString(rowIndex)
			if err != nil {
				return nil, rowReadError(VectorKeyField, rowIndex, err)
			}
			keyByID[id] = key
		}
	}
	return keyByID, nil
}

// storedSupersets reads the supersets of the stored vector rows among keys.
// The result omits a key without a stored row.
func (store *SharedStore) storedSupersets(ctx context.Context, vectors string, scalars []collection.ScalarColumn, keys []string) (map[string]*vectorRecord, error) {
	stored := make(map[string]*vectorRecord, len(keys))
	outputFields := append([]string{VectorKeyField}, supersetOutputFields(scalars)...)
	for start := 0; start < len(keys); start += vectorKeyBatchSize {
		batch := keys[start:min(start+vectorKeyBatchSize, len(keys))]
		resultSet, err := store.inner.client.Get(ctx, milvusclient.NewQueryOption(vectors).
			WithIDs(primaryKeyColumn(VectorKeyField, batch)).
			WithOutputFields(outputFields...).
			WithConsistencyLevel(entity.ClStrong))
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

// mergeStoredSupersets adds the stored supersets of existing vector rows into
// records. It returns each key with a stored superset that already contained
// the record superset. Those vector rows do not need a write.
func (store *SharedStore) mergeStoredSupersets(ctx context.Context, vectors string, scalars []collection.ScalarColumn, records map[string]*vectorRecord) (map[string]struct{}, error) {
	stored, err := store.storedSupersets(ctx, vectors, scalars, sortedRecordKeys(records))
	if err != nil {
		return nil, err
	}
	unchanged := make(map[string]struct{}, len(stored))
	for key, storedRecord := range stored {
		record, requested := records[key]
		if !requested {
			return nil, fmt.Errorf("read vector supersets of %s: unrequested vector key %q", vectors, key)
		}
		if storedRecord.containsRecord(record) {
			unchanged[key] = struct{}{}
		}
		for name, superset := range storedRecord.supersets {
			record.supersets[name].merge(superset)
		}
	}
	return unchanged, nil
}

func (store *SharedStore) vectorsByKey(ctx context.Context, vectors string, keys []string) (map[string][]float32, error) {
	vectorByKey := make(map[string][]float32, len(keys))
	for start := 0; start < len(keys); start += vectorKeyBatchSize {
		batch := keys[start:min(start+vectorKeyBatchSize, len(keys))]
		resultSet, err := store.inner.client.Get(ctx, milvusclient.NewQueryOption(vectors).
			WithIDs(primaryKeyColumn(VectorKeyField, batch)).
			WithOutputFields(VectorKeyField, DenseVectorField).
			WithConsistencyLevel(entity.ClStrong))
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

// deleteUnreferencedVectors deletes the vector rows among keys that no
// occurrence references.
func (store *SharedStore) deleteUnreferencedVectors(ctx context.Context, occurrences string, vectors string, keys []string) error {
	_, err := store.removeUnreferencedKeys(ctx, occurrences, vectors, keys)
	return err
}

// removeUnreferencedKeys deletes the vectors among keys that no occurrence
// references and returns the number of deleted vector rows.
func (store *SharedStore) removeUnreferencedKeys(ctx context.Context, occurrences string, vectors string, keys []string) (int64, error) {
	var deleted int64
	for start := 0; start < len(keys); start += vectorKeyBatchSize {
		batch := keys[start:min(start+vectorKeyBatchSize, len(keys))]
		referenced, err := store.iterateVectorKeys(ctx, occurrences, vectorKeysExpression(batch))
		if err != nil {
			return deleted, err
		}
		live := map[string]struct{}{}
		for _, key := range referenced {
			live[key] = struct{}{}
		}
		unreferenced := make([]string, 0, len(batch))
		for _, key := range batch {
			if _, found := live[key]; !found {
				unreferenced = append(unreferenced, key)
			}
		}
		if len(unreferenced) == 0 {
			continue
		}
		result, err := store.inner.client.Delete(ctx, milvusclient.NewDeleteOption(vectors).WithStringIDs(VectorKeyField, slices.Clone(unreferenced)))
		if err != nil {
			return deleted, WrapError(ctx, err, "delete unreferenced vectors from "+vectors)
		}
		deleted += result.DeleteCount
	}
	return deleted, nil
}

// iterateVectorKeys returns the distinct vector keys of the occurrences an
// expression matches.
func (store *SharedStore) iterateVectorKeys(ctx context.Context, occurrences string, expression string) ([]string, error) {
	keys := map[string]struct{}{}
	err := store.iterate(ctx, occurrences, expression, []string{VectorKeyField}, func(resultSet milvusclient.ResultSet) error {
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
	})
	if err != nil {
		return nil, err
	}
	return sortedKeys(keys), nil
}

// iterate pages through every row an expression matches with strong
// consistency.
func (store *SharedStore) iterate(ctx context.Context, collectionName string, expression string, outputFields []string, page func(milvusclient.ResultSet) error) error {
	return store.iterateWithBatch(ctx, collectionName, expression, outputFields, sharedIteratorBatch, page)
}

func (store *SharedStore) iterateWithBatch(ctx context.Context, collectionName string, expression string, outputFields []string, batchSize int, page func(milvusclient.ResultSet) error) error {
	iterator, err := store.inner.client.QueryIterator(ctx, milvusclient.NewQueryIteratorOption(collectionName).
		WithBatchSize(batchSize).
		WithFilter(expression).
		WithOutputFields(outputFields...).
		WithConsistencyLevel(entity.ClStrong))
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

// addOccurrenceCells adds the declared scalar cells of every occurrence row in
// one result page to the record of the row's vector.
func addOccurrenceCells(resultSet milvusclient.ResultSet, scalars []collection.ScalarColumn, records map[string]*vectorRecord) error {
	keyColumn := resultSet.GetColumn(VectorKeyField)
	if keyColumn == nil {
		return collection.ErrSearchResultIncomplete
	}
	for rowIndex := range resultSet.ResultCount {
		key, err := keyColumn.GetAsString(rowIndex)
		if err != nil {
			return rowReadError(VectorKeyField, rowIndex, err)
		}
		record, found := records[key]
		if !found {
			return rowReadError(VectorKeyField, rowIndex, fmt.Errorf("unrequested vector key %q", key))
		}
		cells := make(map[string]collection.ScalarCell, len(scalars))
		for _, declared := range scalars {
			cell, cellErr := ScalarCellAt(resultSet.GetColumn(declared.Name), declared, rowIndex)
			if cellErr != nil {
				return cellErr
			}
			cells[declared.Name] = cell
		}
		record.addCells(scalars, cells)
	}
	return nil
}

// primaryKeyColumn returns a VarChar ID column over a copy of values. The
// Milvus client's WithIDs quotes the strings of the column it receives in
// place, which would change the caller's slice.
func primaryKeyColumn(field string, values []string) *column.ColumnVarChar {
	return column.NewColumnVarChar(field, slices.Clone(values))
}

func vectorKeysExpression(keys []string) string {
	quoted := make([]string, 0, len(keys))
	for _, key := range keys {
		quoted = append(quoted, `"`+collection.EscapeString(key)+`"`)
	}
	return VectorKeyField + " in [" + strings.Join(quoted, ", ") + "]"
}

func sortedKeys(values map[string]struct{}) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func sortedRecordKeys(records map[string]*vectorRecord) []string {
	keys := make([]string, 0, len(records))
	for key := range records {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// rowIdentity is the reuse identity an occurrence write stores for each row:
// the content hash and the embedding model, each with its null state.
type rowIdentity struct {
	contentHashes     []string
	contentHashSet    []bool
	embeddingModels   []string
	embeddingModelSet []bool
}

// computedIdentity is the identity [SharedStore.Upsert] writes: the hash of each
// row's content and one embedding model for every row. An empty model writes
// null.
func computedIdentity(rows []collection.Row, embeddingModel string) rowIdentity {
	normalizedModel, _ := SanitizeUTF8(embeddingModel)
	identity := rowIdentity{
		contentHashes:     make([]string, 0, len(rows)),
		contentHashSet:    make([]bool, 0, len(rows)),
		embeddingModels:   make([]string, 0, len(rows)),
		embeddingModelSet: make([]bool, 0, len(rows)),
	}
	for _, row := range rows {
		identity.contentHashes = append(identity.contentHashes, ContentHash(row.Content))
		identity.contentHashSet = append(identity.contentHashSet, true)
		identity.embeddingModels = append(identity.embeddingModels, normalizedModel)
		identity.embeddingModelSet = append(identity.embeddingModelSet, normalizedModel != "")
	}
	return identity
}

func (identity rowIdentity) slice(start int, end int) rowIdentity {
	return rowIdentity{
		contentHashes:     identity.contentHashes[start:end],
		contentHashSet:    identity.contentHashSet[start:end],
		embeddingModels:   identity.embeddingModels[start:end],
		embeddingModelSet: identity.embeddingModelSet[start:end],
	}
}

// buildOccurrenceWriteOption builds the column-based upsert of occurrence rows:
// every built-in column except the dense vector, plus each row's vector key.
func buildOccurrenceWriteOption(ctx context.Context, collectionName string, rows []collection.Row, keys []string, scalars []collection.ScalarColumn, identity rowIdentity) (WriteOption, error) {
	columns := buildInsertColumns(ctx, rows)
	splitPartColumn, err := NewSplitPartColumn(collectionName, columns.splitParts, columns.splitRecorded)
	if err != nil {
		return nil, err
	}
	embeddingModelColumn, err := column.NewNullableColumnVarChar(EmbeddingModelField, identity.embeddingModels, identity.embeddingModelSet, column.WithSparseNullableMode[string](true))
	if err != nil {
		slog.ErrorContext(ctx, "build embedding model column failed", "collection", collectionName, "err", err)
		return nil, fmt.Errorf("build embedding model column for %s: %w", collectionName, err)
	}
	contentHashColumn, err := column.NewNullableColumnVarChar(ContentHashField, identity.contentHashes, identity.contentHashSet, column.WithSparseNullableMode[string](true))
	if err != nil {
		slog.ErrorContext(ctx, "build content hash column failed", "collection", collectionName, "err", err)
		return nil, fmt.Errorf("build content hash column for %s: %w", collectionName, err)
	}
	declaredColumns, err := DeclaredScalarInsertColumns(collectionName, scalars, rows)
	if err != nil {
		return nil, err
	}
	option := milvusclient.NewColumnBasedInsertOption(collectionName).
		WithVarcharColumn(IDField, columns.ids).
		WithVarcharColumn(ContentField, columns.contents).
		WithColumns(contentHashColumn).
		WithVarcharColumn(RelativePathField, columns.relativePaths).
		WithInt64Column(StartLineField, columns.startLines).
		WithInt64Column(EndLineField, columns.endLines).
		WithVarcharColumn(FileExtensionField, columns.fileExtensions).
		WithVarcharColumn(MetadataField, columns.metadataValues).
		WithVarcharColumn(VectorKeyField, keys).
		WithColumns(splitPartColumn, embeddingModelColumn)
	if len(declaredColumns) > 0 {
		option = option.WithColumns(declaredColumns...)
	}
	return option, nil
}
