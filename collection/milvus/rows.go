package milvus

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

// ContentHash returns the hex SHA-256 of content after UTF-8 repair. It is the
// value of the contentHash column.
func ContentHash(content string) string {
	normalized, _ := SanitizeUTF8(content)
	sum := sha256.Sum256([]byte(normalized))
	return hex.EncodeToString(sum[:])
}

// insertColumns accumulates the built-in column values of one insert batch.
type insertColumns struct {
	ids            []string
	contents       []string
	contentHashes  []string
	relativePaths  []string
	startLines     []int64
	endLines       []int64
	fileExtensions []string
	metadataValues []string
	splitParts     []int64
	splitRecorded  []bool
	vectors        [][]float32
	sanitizedCount int
}

// WriteOption is a column-based write request that both Insert and Upsert accept.
type WriteOption interface {
	milvusclient.InsertOption
	milvusclient.UpsertOption
}

// BuildInsertOption builds the column-based insert request for rows. Every row
// must carry a vector of the same width. scalars lists the declared scalar
// columns the request writes from each row's Scalars map. embeddingModel fills
// the embeddingModel column of every row, and an empty value writes null.
func BuildInsertOption(ctx context.Context, collectionName string, rows []collection.Row, scalars []collection.ScalarColumn, embeddingModel string) (WriteOption, error) {
	columns := buildInsertColumns(ctx, rows)
	if columns.sanitizedCount > 0 {
		slog.WarnContext(
			ctx,
			"semantic.insertBatch sanitized chunks before Milvus marshal",
			"collection",
			collectionName,
			"sanitized",
			columns.sanitizedCount,
			"batch_size",
			len(rows),
		)
	}
	splitPartColumn, err := NewSplitPartColumn(collectionName, columns.splitParts, columns.splitRecorded)
	if err != nil {
		return nil, err
	}
	embeddingModelColumn, err := NewEmbeddingModelColumn(collectionName, embeddingModel, len(rows))
	if err != nil {
		return nil, err
	}
	declaredColumns, err := DeclaredScalarInsertColumns(collectionName, scalars, rows)
	if err != nil {
		return nil, err
	}
	insertOption := milvusclient.NewColumnBasedInsertOption(collectionName).
		WithVarcharColumn(IDField, columns.ids).
		WithVarcharColumn(ContentField, columns.contents).
		WithVarcharColumn(ContentHashField, columns.contentHashes).
		WithVarcharColumn(RelativePathField, columns.relativePaths).
		WithInt64Column(StartLineField, columns.startLines).
		WithInt64Column(EndLineField, columns.endLines).
		WithVarcharColumn(FileExtensionField, columns.fileExtensions).
		WithVarcharColumn(MetadataField, columns.metadataValues).
		WithColumns(splitPartColumn, embeddingModelColumn).
		WithFloatVectorColumn(DenseVectorField, len(columns.vectors[0]), columns.vectors)
	if len(declaredColumns) > 0 {
		insertOption = insertOption.WithColumns(declaredColumns...)
	}
	return insertOption, nil
}

func buildInsertColumns(ctx context.Context, rows []collection.Row) insertColumns {
	columns := insertColumns{
		ids:            make([]string, 0, len(rows)),
		contents:       make([]string, 0, len(rows)),
		contentHashes:  make([]string, 0, len(rows)),
		relativePaths:  make([]string, 0, len(rows)),
		startLines:     make([]int64, 0, len(rows)),
		endLines:       make([]int64, 0, len(rows)),
		fileExtensions: make([]string, 0, len(rows)),
		metadataValues: make([]string, 0, len(rows)),
		splitParts:     make([]int64, 0, len(rows)),
		splitRecorded:  make([]bool, 0, len(rows)),
		vectors:        make([][]float32, 0, len(rows)),
		sanitizedCount: 0,
	}
	for _, row := range rows {
		content, contentChanged := SanitizeUTF8(row.Content)
		relativePath, pathChanged := SanitizeUTF8(row.RelativePath)
		fileExtension, extensionChanged := SanitizeUTF8(row.FileExtension)
		metadataValue, metadataChanged := SanitizeUTF8(row.Metadata)
		if contentChanged || pathChanged || extensionChanged || metadataChanged {
			columns.sanitizedCount++
			slog.WarnContext(ctx, "semantic.sanitized_invalid_utf8", "relative_path", row.RelativePath, "start_line", row.StartLine, "end_line", row.EndLine, "content_changed", contentChanged, "path_changed", pathChanged, "extension_changed", extensionChanged, "metadata_changed", metadataChanged)
		}
		columns.ids = append(columns.ids, row.ID)
		columns.contents = append(columns.contents, content)
		columns.contentHashes = append(columns.contentHashes, ContentHash(row.Content))
		columns.relativePaths = append(columns.relativePaths, relativePath)
		columns.startLines = append(columns.startLines, int64(row.StartLine))
		columns.endLines = append(columns.endLines, int64(row.EndLine))
		columns.fileExtensions = append(columns.fileExtensions, fileExtension)
		columns.metadataValues = append(columns.metadataValues, metadataValue)
		columns.splitParts = append(columns.splitParts, int64(row.SplitPart))
		columns.splitRecorded = append(columns.splitRecorded, row.SplitPartRecorded)
		columns.vectors = append(columns.vectors, row.Vector)
	}
	return columns
}

// NewSplitPartColumn builds the nullable splitPart insert column. The recorded
// flags mark which rows store a value.
func NewSplitPartColumn(collectionName string, splitParts []int64, splitPartsRecorded []bool) (column.Column, error) {
	splitPartColumn, err := column.NewNullableColumnInt64(
		SplitPartField,
		splitParts,
		splitPartsRecorded,
		column.WithSparseNullableMode[int64](true),
	)
	if err != nil {
		slog.Error("build split part insert column failed", "collection", collectionName, "err", err)
		return nil, fmt.Errorf("build split part column for %s: %w", collectionName, err)
	}
	return splitPartColumn, nil
}

// NewEmbeddingModelColumn builds the nullable embeddingModel insert column with
// one value for every row. An empty model writes null.
func NewEmbeddingModelColumn(collectionName string, embeddingModel string, rowCount int) (column.Column, error) {
	normalizedModel, _ := SanitizeUTF8(embeddingModel)
	values := make([]string, rowCount)
	validData := make([]bool, rowCount)
	for index := range rowCount {
		values[index] = normalizedModel
		validData[index] = normalizedModel != ""
	}
	embeddingModelColumn, err := column.NewNullableColumnVarChar(
		EmbeddingModelField,
		values,
		validData,
		column.WithSparseNullableMode[string](true),
	)
	if err != nil {
		wrappedErr := fmt.Errorf("build embedding model column for %s: %w", collectionName, err)
		slog.Error("build embedding model column failed", "collection", collectionName, "err", wrappedErr)
		return nil, wrappedErr
	}
	return embeddingModelColumn, nil
}

// Insert appends rows to a collection and returns the count Milvus
// acknowledged.
func (store *Store) Insert(ctx context.Context, collectionName string, rows []collection.Row, scalars []collection.ScalarColumn) (int64, error) {
	if len(rows) == 0 {
		return 0, nil
	}
	option, err := BuildInsertOption(ctx, collectionName, rows, scalars, store.options.EmbeddingModel)
	if err != nil {
		return 0, err
	}
	result, err := store.client.Insert(ctx, option)
	if err != nil {
		return 0, WrapError(ctx, err, "insert Milvus batch into "+collectionName)
	}
	return result.InsertCount, nil
}

// Upsert writes rows to a collection. A row replaces the stored row with the
// same ID.
func (store *Store) Upsert(ctx context.Context, collectionName string, declaration collection.Declaration, rows []collection.Row) error {
	if len(rows) == 0 {
		return nil
	}
	option, err := BuildInsertOption(ctx, collectionName, rows, declaration.Scalars, store.options.EmbeddingModel)
	if err != nil {
		return err
	}
	if _, err := store.client.Upsert(ctx, option); err != nil {
		return WrapError(ctx, err, "upsert Milvus batch into "+collectionName)
	}
	return nil
}

// DeclaredScalarInsertColumns builds one Milvus insert column per declared
// scalar column from the Scalars map of each row. A nullable column marks a
// missing or null value invalid. A column that is not nullable rejects a
// missing or null value, and every column rejects a value of another type.
func DeclaredScalarInsertColumns(
	collectionName string,
	declared []collection.ScalarColumn,
	rows []collection.Row,
) ([]column.Column, error) {
	columns := make([]column.Column, 0, len(declared))
	for _, declaration := range declared {
		built, err := declaredScalarInsertColumn(collectionName, declaration, rows)
		if err != nil {
			return nil, err
		}
		columns = append(columns, built)
	}
	return columns, nil
}

func declaredScalarInsertColumn(
	collectionName string,
	declaration collection.ScalarColumn,
	rows []collection.Row,
) (column.Column, error) {
	validData := make([]bool, 0, len(rows))
	stringValues := make([]string, 0, len(rows))
	boolValues := make([]bool, 0, len(rows))
	int64Values := make([]int64, 0, len(rows))
	for _, row := range rows {
		value, present := row.Scalars[declaration.Name]
		valid := present && !value.Null
		if !valid && !declaration.Nullable {
			return nil, fmt.Errorf("insert into %s: row %s has no value for column %s, which is not nullable", collectionName, row.RelativePath, declaration.Name)
		}
		if valid && value.Type != declaration.Type {
			return nil, fmt.Errorf("insert into %s: row %s has a %s value for %s column %s", collectionName, row.RelativePath, value.Type, declaration.Type, declaration.Name)
		}
		validData = append(validData, valid)
		stringValue, _ := SanitizeUTF8(value.String)
		stringValues = append(stringValues, stringValue)
		boolValues = append(boolValues, value.Bool)
		int64Values = append(int64Values, value.Int64)
	}
	return newDeclaredColumn(collectionName, declaration, validData, stringValues, boolValues, int64Values)
}

func newDeclaredColumn(
	collectionName string,
	declaration collection.ScalarColumn,
	validData []bool,
	stringValues []string,
	boolValues []bool,
	int64Values []int64,
) (column.Column, error) {
	var built column.Column
	var err error
	switch declaration.Type {
	case collection.ScalarTypeString:
		if !declaration.Nullable {
			return column.NewColumnVarChar(declaration.Name, stringValues), nil
		}
		built, err = column.NewNullableColumnVarChar(declaration.Name, stringValues, validData, column.WithSparseNullableMode[string](true))
	case collection.ScalarTypeBool:
		if !declaration.Nullable {
			return column.NewColumnBool(declaration.Name, boolValues), nil
		}
		built, err = column.NewNullableColumnBool(declaration.Name, boolValues, validData, column.WithSparseNullableMode[bool](true))
	case collection.ScalarTypeInt64:
		if !declaration.Nullable {
			return column.NewColumnInt64(declaration.Name, int64Values), nil
		}
		built, err = column.NewNullableColumnInt64(declaration.Name, int64Values, validData, column.WithSparseNullableMode[int64](true))
	default:
		return nil, fmt.Errorf("unsupported declared column type %q", declaration.Type)
	}
	if err != nil {
		slog.Error("build declared scalar insert column failed", "collection", collectionName, "column", declaration.Name, "err", err)
		return nil, fmt.Errorf("build declared column %s for %s: %w", declaration.Name, collectionName, err)
	}
	return built, nil
}

// ScalarRowBytes estimates the raw bytes the declared scalar values of one row
// add to an insert request, plus a small per-column framing allowance.
func ScalarRowBytes(declared []collection.ScalarColumn, scalars map[string]collection.ScalarValue) int {
	const perColumnFraming = 6
	total := 0
	for _, declaration := range declared {
		total += perColumnFraming
		value := scalars[declaration.Name]
		switch declaration.Type {
		case collection.ScalarTypeString:
			total += len(value.String)
		case collection.ScalarTypeBool:
			total += boolBytes
		case collection.ScalarTypeInt64:
			total += int64Bytes
		default:
		}
	}
	return total
}
