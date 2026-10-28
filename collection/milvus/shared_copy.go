package milvus

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"github.com/milvus-io/milvus/client/v3/column"
	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

const (
	// copyBatchSize bounds the source rows one copy page reads and writes.
	copyBatchSize = 256
	// copyReportSampleLimit bounds the row IDs a copy report lists per class.
	copyReportSampleLimit = 100
)

// CopyReport is the result of [SharedStore.VerifyCopy]. Each count covers every
// row. Each ID list keeps at most copyReportSampleLimit IDs.
type CopyReport struct {
	SourceRows          int64
	TargetRows          int64
	TargetVectors       int64
	MissingCount        int64
	Missing             []string
	MismatchedCount     int64
	Mismatched          []string
	ExtraCount          int64
	Extra               []string
	DanglingCount       int64
	UnreferencedVectors int64
}

// Equal reports whether the target stores every source row with every field
// and the exact vector bytes, stores no other row, and stores no vector that no
// row references.
func (report CopyReport) Equal() bool {
	return report.MissingCount == 0 && report.MismatchedCount == 0 && report.ExtraCount == 0 &&
		report.DanglingCount == 0 && report.UnreferencedVectors == 0 && report.SourceRows == report.TargetRows
}

func appendSample(sample []string, id string) []string {
	if len(sample) >= copyReportSampleLimit {
		return sample
	}
	return append(sample, id)
}

// storedRow is one stored row with every column a copy preserves: the row
// columns, the content hash with its null state, the embedding model, and the
// vector key of a shared occurrence row.
type storedRow struct {
	row            collection.Row
	contentHash    string
	contentHashSet bool
	embeddingModel string
	vectorKey      string
}

func copyOutputFields(declaration collection.Declaration, withVector bool, withKey bool) []string {
	fields := []string{IDField, ContentField, RelativePathField, StartLineField, EndLineField, FileExtensionField, MetadataField, SplitPartField, ContentHashField, EmbeddingModelField}
	if withVector {
		fields = append(fields, DenseVectorField)
	}
	if withKey {
		fields = append(fields, VectorKeyField)
	}
	for _, declared := range declaration.Scalars {
		fields = append(fields, declared.Name)
	}
	return fields
}

// nullableStringAt reads a nullable string column and its null state. A missing
// column reads as null.
func nullableStringAt(valueColumn column.Column, rowIndex int) (string, bool, error) {
	if valueColumn == nil {
		return "", false, nil
	}
	isNull, err := valueColumn.IsNull(rowIndex)
	if err != nil {
		return "", false, rowReadError(valueColumn.Name(), rowIndex, err)
	}
	if isNull {
		return "", false, nil
	}
	value, err := valueColumn.GetAsString(rowIndex)
	if err != nil {
		return "", false, rowReadError(valueColumn.Name(), rowIndex, err)
	}
	return value, true, nil
}

func requiredString(resultSet milvusclient.ResultSet, field string, rowIndex int) (string, error) {
	valueColumn := resultSet.GetColumn(field)
	if valueColumn == nil {
		return "", collection.ErrSearchResultIncomplete
	}
	value, err := valueColumn.GetAsString(rowIndex)
	if err != nil {
		return "", rowReadError(field, rowIndex, err)
	}
	return value, nil
}

func requiredInt64(resultSet milvusclient.ResultSet, field string, rowIndex int) (int64, error) {
	valueColumn := resultSet.GetColumn(field)
	if valueColumn == nil {
		return 0, collection.ErrSearchResultIncomplete
	}
	value, err := valueColumn.GetAsInt64(rowIndex)
	if err != nil {
		return 0, rowReadError(field, rowIndex, err)
	}
	return value, nil
}

// readStoredRows decodes every row of one result page.
func readStoredRows(resultSet milvusclient.ResultSet, declaration collection.Declaration, withVector bool, withKey bool) ([]storedRow, error) {
	rows := make([]storedRow, 0, resultSet.ResultCount)
	for rowIndex := range resultSet.ResultCount {
		row, err := readStoredRow(resultSet, declaration, withVector, withKey, rowIndex)
		if err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	return rows, nil
}

func readStoredRow(resultSet milvusclient.ResultSet, declaration collection.Declaration, withVector bool, withKey bool, rowIndex int) (storedRow, error) {
	stored := storedRow{row: collection.Row{Scalars: make(map[string]collection.ScalarValue, len(declaration.Scalars))}}
	var err error
	for field, target := range map[string]*string{IDField: &stored.row.ID, ContentField: &stored.row.Content, RelativePathField: &stored.row.RelativePath, FileExtensionField: &stored.row.FileExtension, MetadataField: &stored.row.Metadata} {
		if *target, err = requiredString(resultSet, field, rowIndex); err != nil {
			return stored, err
		}
	}
	startLine, err := requiredInt64(resultSet, StartLineField, rowIndex)
	if err != nil {
		return stored, err
	}
	endLine, err := requiredInt64(resultSet, EndLineField, rowIndex)
	if err != nil {
		return stored, err
	}
	stored.row.StartLine = SafeInt32(startLine)
	stored.row.EndLine = SafeInt32(endLine)
	if stored.row.SplitPart, stored.row.SplitPartRecorded, err = SplitPartAt(resultSet.GetColumn(SplitPartField), rowIndex); err != nil {
		return stored, err
	}
	if stored.contentHash, stored.contentHashSet, err = nullableStringAt(resultSet.GetColumn(ContentHashField), rowIndex); err != nil {
		return stored, err
	}
	if stored.embeddingModel, _, err = nullableStringAt(resultSet.GetColumn(EmbeddingModelField), rowIndex); err != nil {
		return stored, err
	}
	if withVector {
		vectorColumn := resultSet.GetColumn(DenseVectorField)
		if vectorColumn == nil {
			return stored, collection.ErrSearchResultIncomplete
		}
		if stored.row.Vector, err = VectorAt(vectorColumn, rowIndex); err != nil {
			return stored, err
		}
	}
	if withKey {
		if stored.vectorKey, err = requiredString(resultSet, VectorKeyField, rowIndex); err != nil {
			return stored, err
		}
	}
	for _, declared := range declaration.Scalars {
		value, valueErr := ScalarValueAt(resultSet.GetColumn(declared.Name), declared, rowIndex)
		if valueErr != nil {
			return stored, valueErr
		}
		stored.row.Scalars[declared.Name] = value
	}
	return stored, nil
}

// writeStoredRows writes copied rows with the identity each source row stores.
func (store *SharedStore) writeStoredRows(ctx context.Context, target string, declaration collection.Declaration, rows []storedRow) error {
	if len(rows) == 0 {
		return nil
	}
	plain := make([]collection.Row, 0, len(rows))
	identity := rowIdentity{
		contentHashes:   make([]string, 0, len(rows)),
		contentHashSet:  make([]bool, 0, len(rows)),
		embeddingModels: make([]string, 0, len(rows)),
	}
	for _, row := range rows {
		plain = append(plain, row.row)
		identity.contentHashes = append(identity.contentHashes, row.contentHash)
		identity.contentHashSet = append(identity.contentHashSet, row.contentHashSet)
		identity.embeddingModels = append(identity.embeddingModels, row.embeddingModel)
	}
	store.write.Lock()
	defer store.write.Unlock()
	return store.writeRows(ctx, target, declaration, plain, identity)
}

// CopyFrom copies every row of the single-collection source into the shared
// target. It keeps each row's ID, every column, the content hash with its null
// state, the embedding model, and the exact vector bytes. It never embeds. A
// repeated copy rewrites the same rows. progress receives the copied row count
// after each page.
func (store *SharedStore) CopyFrom(ctx context.Context, source string, target string, declaration collection.Declaration, progress func(int64)) (int64, error) {
	var copied int64
	err := store.iterateBatch(ctx, source, IDField+` != ""`, copyOutputFields(declaration, true, false), copyBatchSize, func(resultSet milvusclient.ResultSet) error {
		rows, err := readStoredRows(resultSet, declaration, true, false)
		if err != nil {
			return err
		}
		if err := store.writeStoredRows(ctx, target, declaration, rows); err != nil {
			return err
		}
		copied += int64(len(rows))
		if progress != nil {
			progress(copied)
		}
		return nil
	})
	return copied, err
}

// CopyIDs copies the source rows with the given IDs into the shared target, the
// same way [SharedStore.CopyFrom] copies every row.
func (store *SharedStore) CopyIDs(ctx context.Context, source string, target string, declaration collection.Declaration, ids []string) error {
	for start := 0; start < len(ids); start += copyBatchSize {
		batch := ids[start:min(start+copyBatchSize, len(ids))]
		resultSet, err := store.inner.client.Get(ctx, milvusclient.NewQueryOption(source).
			WithIDs(primaryKeyColumn(IDField, batch)).
			WithOutputFields(copyOutputFields(declaration, true, false)...).
			WithConsistencyLevel(entity.ClStrong))
		if err != nil {
			return WrapError(ctx, err, "read copy rows of "+source)
		}
		rows, err := readStoredRows(resultSet, declaration, true, false)
		if err != nil {
			return err
		}
		if err := store.writeStoredRows(ctx, target, declaration, rows); err != nil {
			return err
		}
	}
	return nil
}

// RemoveOccurrences deletes the shared occurrence rows with the given IDs and
// every vector that no remaining row references.
func (store *SharedStore) RemoveOccurrences(ctx context.Context, target string, ids []string) (int64, error) {
	expressions := make([]string, 0)
	for start := 0; start < len(ids); start += vectorKeyBatchSize {
		batch := ids[start:min(start+vectorKeyBatchSize, len(ids))]
		quoted := make([]string, 0, len(batch))
		for _, id := range batch {
			quoted = append(quoted, `"`+collection.EscapeString(id)+`"`)
		}
		expressions = append(expressions, IDField+" in ["+strings.Join(quoted, ", ")+"]")
	}
	store.write.Lock()
	defer store.write.Unlock()
	return store.deleteOccurrences(ctx, strings.TrimSpace(target), expressions)
}

// VerifyCopy compares the shared target with the single-collection source row
// by row. It checks every column, the content hash null state, the embedding
// model, the vector key, and the exact vector bytes, then lists target rows the
// source lacks and target vectors that no row references.
func (store *SharedStore) VerifyCopy(ctx context.Context, source string, target string, declaration collection.Declaration) (CopyReport, error) {
	report := CopyReport{Missing: nil, Mismatched: nil, Extra: nil}
	err := store.iterateBatch(ctx, source, IDField+` != ""`, copyOutputFields(declaration, true, false), copyBatchSize, func(resultSet milvusclient.ResultSet) error {
		sourceRows, err := readStoredRows(resultSet, declaration, true, false)
		if err != nil {
			return err
		}
		report.SourceRows += int64(len(sourceRows))
		return store.compareCopyPage(ctx, target, declaration, sourceRows, &report)
	})
	if err != nil {
		return report, err
	}
	if err := store.countExtraRows(ctx, source, target, &report); err != nil {
		return report, err
	}
	if err := store.countUnreferencedVectors(ctx, target, &report); err != nil {
		return report, err
	}
	slog.InfoContext(ctx, "collection.shared_copy_verified", "source", source, "target", target,
		"source_rows", report.SourceRows, "target_rows", report.TargetRows, "target_vectors", report.TargetVectors,
		"missing", report.MissingCount, "mismatched", report.MismatchedCount, "extra", report.ExtraCount,
		"dangling", report.DanglingCount, "unreferenced_vectors", report.UnreferencedVectors)
	return report, nil
}

func (store *SharedStore) compareCopyPage(ctx context.Context, target string, declaration collection.Declaration, sourceRows []storedRow, report *CopyReport) error {
	ids := make([]string, 0, len(sourceRows))
	for _, row := range sourceRows {
		ids = append(ids, row.row.ID)
	}
	resultSet, err := store.inner.client.Get(ctx, milvusclient.NewQueryOption(target).
		WithIDs(primaryKeyColumn(IDField, ids)).
		WithOutputFields(copyOutputFields(declaration, false, true)...).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		return WrapError(ctx, err, "read copied rows of "+target)
	}
	targetRows, err := readStoredRows(resultSet, declaration, false, true)
	if err != nil {
		return err
	}
	byID := make(map[string]storedRow, len(targetRows))
	keySet := map[string]struct{}{}
	for _, row := range targetRows {
		byID[row.row.ID] = row
		keySet[row.vectorKey] = struct{}{}
	}
	vectorByKey, err := store.vectorsByKey(ctx, VectorCollectionName(target), sortedKeys(keySet))
	if err != nil {
		return err
	}
	for _, sourceRow := range sourceRows {
		targetRow, found := byID[sourceRow.row.ID]
		if !found {
			report.MissingCount++
			report.Missing = appendSample(report.Missing, sourceRow.row.ID)
			continue
		}
		targetVector, found := vectorByKey[targetRow.vectorKey]
		if !found {
			report.DanglingCount++
		}
		if !found || !sameStoredRow(sourceRow, targetRow, targetVector, declaration) {
			report.MismatchedCount++
			report.Mismatched = appendSample(report.Mismatched, sourceRow.row.ID)
		}
	}
	return nil
}

// sameStoredRow reports whether a copied occurrence row and its vector store
// exactly the values of the source row.
func sameStoredRow(source storedRow, target storedRow, targetVector []float32, declaration collection.Declaration) bool {
	sourceKey := VectorKey(source.row.Vector)
	if target.vectorKey != sourceKey || VectorKey(targetVector) != sourceKey {
		return false
	}
	left := source.row
	right := target.row
	if left.Content != right.Content || left.RelativePath != right.RelativePath || left.StartLine != right.StartLine ||
		left.EndLine != right.EndLine || left.FileExtension != right.FileExtension || left.Metadata != right.Metadata ||
		left.SplitPart != right.SplitPart || left.SplitPartRecorded != right.SplitPartRecorded {
		return false
	}
	if source.contentHash != target.contentHash || source.contentHashSet != target.contentHashSet || source.embeddingModel != target.embeddingModel {
		return false
	}
	for _, declared := range declaration.Scalars {
		if left.Scalars[declared.Name] != right.Scalars[declared.Name] {
			return false
		}
	}
	return true
}

// countExtraRows counts the target rows and the target row IDs the source
// lacks.
func (store *SharedStore) countExtraRows(ctx context.Context, source string, target string, report *CopyReport) error {
	return store.iterateBatch(ctx, target, IDField+` != ""`, []string{IDField}, copyBatchSize, func(resultSet milvusclient.ResultSet) error {
		ids := make([]string, 0, resultSet.ResultCount)
		for rowIndex := range resultSet.ResultCount {
			id, err := requiredString(resultSet, IDField, rowIndex)
			if err != nil {
				return err
			}
			ids = append(ids, id)
		}
		report.TargetRows += int64(len(ids))
		present, err := store.inner.client.Get(ctx, milvusclient.NewQueryOption(source).
			WithIDs(primaryKeyColumn(IDField, ids)).
			WithOutputFields(IDField).
			WithConsistencyLevel(entity.ClStrong))
		if err != nil {
			return WrapError(ctx, err, "read source IDs of "+source)
		}
		found := make(map[string]struct{}, present.ResultCount)
		for rowIndex := range present.ResultCount {
			id, idErr := requiredString(present, IDField, rowIndex)
			if idErr != nil {
				return idErr
			}
			found[id] = struct{}{}
		}
		for _, id := range ids {
			if _, exists := found[id]; !exists {
				report.ExtraCount++
				report.Extra = appendSample(report.Extra, id)
			}
		}
		return nil
	})
}

// countUnreferencedVectors counts the target vectors and the vectors that no
// target row references.
func (store *SharedStore) countUnreferencedVectors(ctx context.Context, target string, report *CopyReport) error {
	vectors := VectorCollectionName(target)
	return store.iterateBatch(ctx, vectors, VectorKeyField+` != ""`, []string{VectorKeyField}, vectorKeyBatchSize, func(resultSet milvusclient.ResultSet) error {
		keys := make([]string, 0, resultSet.ResultCount)
		for rowIndex := range resultSet.ResultCount {
			key, err := requiredString(resultSet, VectorKeyField, rowIndex)
			if err != nil {
				return err
			}
			keys = append(keys, key)
		}
		report.TargetVectors += int64(len(keys))
		referenced, err := store.iterateVectorKeys(ctx, target, vectorKeysExpression(keys))
		if err != nil {
			return err
		}
		for _, key := range keys {
			if !slices.Contains(referenced, key) {
				report.UnreferencedVectors++
			}
		}
		return nil
	})
}

// iterateBatch pages through every row an expression matches with strong
// consistency and the given page size.
func (store *SharedStore) iterateBatch(ctx context.Context, collectionName string, expression string, outputFields []string, batchSize int, page func(milvusclient.ResultSet) error) error {
	if batchSize <= 0 {
		return fmt.Errorf("iterate %s: batch size %d is not positive", collectionName, batchSize)
	}
	return store.iterateWithBatch(ctx, collectionName, expression, outputFields, batchSize, page)
}
