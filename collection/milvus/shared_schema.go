package milvus

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"slices"
	"sort"

	"github.com/milvus-io/milvus/client/v3/column"
	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/index"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

const (
	// VectorKeyField is the occurrence column and vector primary key that
	// stores the hex SHA-256 of a dense vector's little-endian float32 bytes.
	VectorKeyField = "vectorKey"
	// VectorCollectionSuffix is appended to an occurrence collection name to
	// form its vector collection name.
	VectorCollectionSuffix = "_vectors"

	vectorKeyFieldMaxLength = 64
	supersetValuesPrefix    = "values_"
	supersetOverflowPrefix  = "overflow_"
	supersetMinimumPrefix   = "minimum_"
	supersetMaximumPrefix   = "maximum_"
	// supersetArrayCapacity is the Milvus maximum array capacity.
	supersetArrayCapacity = 4096
	boolArrayCapacity     = 2
	float32Bytes          = 4
)

// VectorKey returns the hex SHA-256 of the little-endian float32 bytes of
// vector.
func VectorKey(vector []float32) string {
	body := make([]byte, len(vector)*float32Bytes)
	for position, value := range vector {
		binary.LittleEndian.PutUint32(body[position*float32Bytes:], math.Float32bits(value))
	}
	digest := sha256.Sum256(body)
	return hex.EncodeToString(digest[:])
}

// VectorCollectionName returns the vector collection of an occurrence
// collection.
func VectorCollectionName(occurrenceCollection string) string {
	return occurrenceCollection + VectorCollectionSuffix
}

func supersetValuesField(columnName string) string {
	return supersetValuesPrefix + columnName
}

func supersetOverflowField(columnName string) string {
	return supersetOverflowPrefix + columnName
}

func supersetMinimumField(columnName string) string {
	return supersetMinimumPrefix + columnName
}

func supersetMaximumField(columnName string) string {
	return supersetMaximumPrefix + columnName
}

// occurrenceCollectionOption builds the occurrence collection: every built-in
// column except the dense vector, the vector key, the declared scalars, and the
// BM25 sparse vector.
func (store *Store) occurrenceCollectionOption(collectionName string, scalars []collection.ScalarColumn) milvusclient.CreateCollectionOption {
	schema := entity.NewSchema().
		WithField(entity.NewField().WithName(IDField).WithDataType(entity.FieldTypeVarChar).WithMaxLength(idFieldMaxLength).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName(ContentField).WithDataType(entity.FieldTypeVarChar).WithMaxLength(contentFieldMaxLength).WithEnableAnalyzer(true).WithEnableMatch(true)).
		WithField(entity.NewField().WithName(RelativePathField).WithDataType(entity.FieldTypeVarChar).WithMaxLength(relativePathFieldMaxLength)).
		WithField(entity.NewField().WithName(StartLineField).WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName(EndLineField).WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName(FileExtensionField).WithDataType(entity.FieldTypeVarChar).WithMaxLength(fileExtensionFieldMaxLength)).
		WithField(entity.NewField().WithName(MetadataField).WithDataType(entity.FieldTypeVarChar).WithMaxLength(metadataFieldMaxLength)).
		WithField(ContentHashFieldSchema()).
		WithField(EmbeddingModelFieldSchema()).
		WithField(SplitPartFieldSchema()).
		WithField(entity.NewField().WithName(VectorKeyField).WithDataType(entity.FieldTypeVarChar).WithMaxLength(vectorKeyFieldMaxLength))
	for _, field := range ScalarFieldSchemas(scalars) {
		schema = schema.WithField(field)
	}
	schema = schema.
		WithField(entity.NewField().WithName(SparseVectorField).WithDataType(entity.FieldTypeSparseVector)).
		WithFunction(entity.NewFunction().WithName(bm25FunctionName).WithType(entity.FunctionTypeBM25).WithInputFields(ContentField).WithOutputFields(SparseVectorField))
	indexOptions := []milvusclient.CreateIndexOption{
		milvusclient.NewCreateIndexOption(collectionName, SparseVectorField, index.NewSparseInvertedIndex(entity.BM25, bm25DropRatioBuild)),
		milvusclient.NewCreateIndexOption(collectionName, ContentHashField, index.NewInvertedIndex()),
		milvusclient.NewCreateIndexOption(collectionName, VectorKeyField, index.NewInvertedIndex()),
	}
	return milvusclient.NewCreateCollectionOption(collectionName, schema).WithIndexOptions(indexOptions...)
}

// vectorCollectionOption builds the vector collection: the vector key, the
// dense vector, and the superset columns of every declared scalar.
func (store *Store) vectorCollectionOption(collectionName string, dimension int, scalars []collection.ScalarColumn) milvusclient.CreateCollectionOption {
	schema := entity.NewSchema().
		WithField(entity.NewField().WithName(VectorKeyField).WithDataType(entity.FieldTypeVarChar).WithMaxLength(vectorKeyFieldMaxLength).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName(DenseVectorField).WithDataType(entity.FieldTypeFloatVector).WithDim(int64(dimension)))
	for _, field := range supersetFieldSchemas(scalars) {
		schema = schema.WithField(field)
	}
	return milvusclient.NewCreateCollectionOption(collectionName, schema).
		WithIndexOptions(milvusclient.NewCreateIndexOption(collectionName, DenseVectorField, store.denseIndex()))
}

func supersetFieldSchemas(scalars []collection.ScalarColumn) []*entity.Field {
	fields := make([]*entity.Field, 0, len(scalars)*2)
	for _, declared := range scalars {
		switch declared.Type {
		case collection.ScalarTypeString:
			fields = append(fields,
				entity.NewField().WithName(supersetValuesField(declared.Name)).WithDataType(entity.FieldTypeArray).
					WithElementType(entity.FieldTypeVarChar).WithMaxLength(int64(declared.MaxLength)).WithMaxCapacity(supersetArrayCapacity),
				entity.NewField().WithName(supersetOverflowField(declared.Name)).WithDataType(entity.FieldTypeBool))
		case collection.ScalarTypeBool:
			fields = append(fields,
				entity.NewField().WithName(supersetValuesField(declared.Name)).WithDataType(entity.FieldTypeArray).
					WithElementType(entity.FieldTypeBool).WithMaxCapacity(boolArrayCapacity))
		case collection.ScalarTypeInt64:
			fields = append(fields,
				entity.NewField().WithName(supersetMinimumField(declared.Name)).WithDataType(entity.FieldTypeInt64),
				entity.NewField().WithName(supersetMaximumField(declared.Name)).WithDataType(entity.FieldTypeInt64))
		}
	}
	return fields
}

func supersetOutputFields(scalars []collection.ScalarColumn) []string {
	names := make([]string, 0, len(scalars)*2)
	for _, field := range supersetFieldSchemas(scalars) {
		names = append(names, field.Name)
	}
	return names
}

// columnSuperset is the set of non-null values one declared column takes across
// the occurrences of one vector. A string set past the array capacity keeps
// overflow set and stops listing values.
type columnSuperset struct {
	strings  map[string]struct{}
	bools    map[bool]struct{}
	minimum  int64
	maximum  int64
	hasRange bool
	overflow bool
}

func newColumnSuperset() *columnSuperset {
	return &columnSuperset{
		strings:  map[string]struct{}{},
		bools:    map[bool]struct{}{},
		minimum:  math.MaxInt64,
		maximum:  math.MinInt64,
		hasRange: false,
		overflow: false,
	}
}

func (superset *columnSuperset) add(value collection.ScalarValue) {
	if value.Null || value.Type == "" {
		return
	}
	switch value.Type {
	case collection.ScalarTypeString:
		if superset.overflow {
			return
		}
		normalized, _ := SanitizeUTF8(value.String)
		superset.strings[normalized] = struct{}{}
		if len(superset.strings) > supersetArrayCapacity {
			superset.overflow = true
			superset.strings = map[string]struct{}{}
		}
	case collection.ScalarTypeBool:
		superset.bools[value.Bool] = struct{}{}
	case collection.ScalarTypeInt64:
		superset.minimum = min(superset.minimum, value.Int64)
		superset.maximum = max(superset.maximum, value.Int64)
		superset.hasRange = true
	}
}

func (superset *columnSuperset) merge(other *columnSuperset) {
	if other.overflow {
		superset.overflow = true
		superset.strings = map[string]struct{}{}
	}
	for value := range other.strings {
		superset.add(collection.StringScalar(value))
	}
	for value := range other.bools {
		superset.bools[value] = struct{}{}
	}
	if other.hasRange {
		superset.add(collection.Int64Scalar(other.minimum))
		superset.add(collection.Int64Scalar(other.maximum))
	}
}

func (superset *columnSuperset) sortedStrings() []string {
	values := make([]string, 0, len(superset.strings))
	for value := range superset.strings {
		values = append(values, value)
	}
	sort.Strings(values)
	return values
}

func (superset *columnSuperset) sortedBools() []bool {
	values := make([]bool, 0, len(superset.bools))
	for _, candidate := range []bool{false, true} {
		if _, found := superset.bools[candidate]; found {
			values = append(values, candidate)
		}
	}
	return values
}

// vectorRecord is one vector collection row: the vector and the superset of
// each declared column across the vector's occurrences.
type vectorRecord struct {
	key       string
	vector    []float32
	supersets map[string]*columnSuperset
}

func newVectorRecord(key string, vector []float32, scalars []collection.ScalarColumn) *vectorRecord {
	record := &vectorRecord{key: key, vector: slices.Clone(vector), supersets: make(map[string]*columnSuperset, len(scalars))}
	for _, declared := range scalars {
		record.supersets[declared.Name] = newColumnSuperset()
	}
	return record
}

func (record *vectorRecord) addRow(scalars []collection.ScalarColumn, values map[string]collection.ScalarValue) {
	for _, declared := range scalars {
		record.supersets[declared.Name].add(values[declared.Name])
	}
}

func (record *vectorRecord) addCells(scalars []collection.ScalarColumn, cells map[string]collection.ScalarCell) {
	for _, declared := range scalars {
		cell := cells[declared.Name]
		if cell.State == collection.ScalarCellValue {
			record.supersets[declared.Name].add(cell.Value)
		}
	}
}

// vectorUpsertOption builds the column-based upsert of vector records.
func vectorUpsertOption(collectionName string, dimension int, scalars []collection.ScalarColumn, records []*vectorRecord) (milvusclient.UpsertOption, error) {
	keys := make([]string, 0, len(records))
	vectors := make([][]float32, 0, len(records))
	for _, record := range records {
		if len(record.vector) != dimension {
			return nil, fmt.Errorf("upsert vectors into %s: vector %s has %d dimensions, want %d", collectionName, record.key, len(record.vector), dimension)
		}
		keys = append(keys, record.key)
		vectors = append(vectors, record.vector)
	}
	option := milvusclient.NewColumnBasedInsertOption(collectionName).
		WithVarcharColumn(VectorKeyField, keys).
		WithFloatVectorColumn(DenseVectorField, dimension, vectors)
	for _, declared := range scalars {
		switch declared.Type {
		case collection.ScalarTypeString:
			values := make([][]string, 0, len(records))
			overflows := make([]bool, 0, len(records))
			for _, record := range records {
				superset := record.supersets[declared.Name]
				values = append(values, superset.sortedStrings())
				overflows = append(overflows, superset.overflow)
			}
			option = option.WithColumns(
				column.NewColumnVarCharArray(supersetValuesField(declared.Name), values),
				column.NewColumnBool(supersetOverflowField(declared.Name), overflows))
		case collection.ScalarTypeBool:
			values := make([][]bool, 0, len(records))
			for _, record := range records {
				values = append(values, record.supersets[declared.Name].sortedBools())
			}
			option = option.WithColumns(column.NewColumnBoolArray(supersetValuesField(declared.Name), values))
		case collection.ScalarTypeInt64:
			minimums := make([]int64, 0, len(records))
			maximums := make([]int64, 0, len(records))
			for _, record := range records {
				superset := record.supersets[declared.Name]
				minimums = append(minimums, superset.minimum)
				maximums = append(maximums, superset.maximum)
			}
			option = option.WithColumns(
				column.NewColumnInt64(supersetMinimumField(declared.Name), minimums),
				column.NewColumnInt64(supersetMaximumField(declared.Name), maximums))
		default:
			return nil, fmt.Errorf("upsert vectors into %s: unsupported declared column type %q", collectionName, declared.Type)
		}
	}
	return option, nil
}

// supersetsFromResult reads the superset columns of one vector collection row.
func supersetsFromResult(resultSet milvusclient.ResultSet, rowIndex int, scalars []collection.ScalarColumn, record *vectorRecord) error {
	for _, declared := range scalars {
		superset := record.supersets[declared.Name]
		var err error
		switch declared.Type {
		case collection.ScalarTypeString:
			err = readStringSuperset(resultSet, rowIndex, declared.Name, superset)
		case collection.ScalarTypeBool:
			err = readBoolSuperset(resultSet, rowIndex, declared.Name, superset)
		case collection.ScalarTypeInt64:
			err = readRangeSuperset(resultSet, rowIndex, declared.Name, superset)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func readStringSuperset(resultSet milvusclient.ResultSet, rowIndex int, columnName string, superset *columnSuperset) error {
	values, err := arrayStringsAt(resultSet.GetColumn(supersetValuesField(columnName)), rowIndex)
	if err != nil {
		return err
	}
	for _, value := range values {
		superset.add(collection.StringScalar(value))
	}
	overflowColumn := resultSet.GetColumn(supersetOverflowField(columnName))
	if overflowColumn == nil {
		return collection.ErrSearchResultIncomplete
	}
	overflow, err := overflowColumn.GetAsBool(rowIndex)
	if err != nil {
		return rowReadError(supersetOverflowField(columnName), rowIndex, err)
	}
	if overflow {
		superset.overflow = true
		superset.strings = map[string]struct{}{}
	}
	return nil
}

func readBoolSuperset(resultSet milvusclient.ResultSet, rowIndex int, columnName string, superset *columnSuperset) error {
	values, err := arrayBoolsAt(resultSet.GetColumn(supersetValuesField(columnName)), rowIndex)
	if err != nil {
		return err
	}
	for _, value := range values {
		superset.add(collection.BoolScalar(value))
	}
	return nil
}

func readRangeSuperset(resultSet milvusclient.ResultSet, rowIndex int, columnName string, superset *columnSuperset) error {
	minimumColumn := resultSet.GetColumn(supersetMinimumField(columnName))
	maximumColumn := resultSet.GetColumn(supersetMaximumField(columnName))
	if minimumColumn == nil || maximumColumn == nil {
		return collection.ErrSearchResultIncomplete
	}
	minimum, err := minimumColumn.GetAsInt64(rowIndex)
	if err != nil {
		return rowReadError(supersetMinimumField(columnName), rowIndex, err)
	}
	maximum, err := maximumColumn.GetAsInt64(rowIndex)
	if err != nil {
		return rowReadError(supersetMaximumField(columnName), rowIndex, err)
	}
	if minimum <= maximum {
		superset.add(collection.Int64Scalar(minimum))
		superset.add(collection.Int64Scalar(maximum))
	}
	return nil
}

func arrayStringsAt(valueColumn column.Column, rowIndex int) ([]string, error) {
	if valueColumn == nil {
		return nil, collection.ErrSearchResultIncomplete
	}
	raw, err := valueColumn.Get(rowIndex)
	if err != nil {
		return nil, rowReadError(valueColumn.Name(), rowIndex, err)
	}
	values, ok := raw.([]string)
	if !ok {
		return nil, rowReadError(valueColumn.Name(), rowIndex, fmt.Errorf("unexpected type %T", raw))
	}
	return values, nil
}

func arrayBoolsAt(valueColumn column.Column, rowIndex int) ([]bool, error) {
	if valueColumn == nil {
		return nil, collection.ErrSearchResultIncomplete
	}
	raw, err := valueColumn.Get(rowIndex)
	if err != nil {
		return nil, rowReadError(valueColumn.Name(), rowIndex, err)
	}
	values, ok := raw.([]bool)
	if !ok {
		return nil, rowReadError(valueColumn.Name(), rowIndex, fmt.Errorf("unexpected type %T", raw))
	}
	return values, nil
}
