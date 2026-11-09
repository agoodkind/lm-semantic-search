package milvus

import (
	"context"

	"github.com/milvus-io/milvus/client/v3/column"
	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

func copyOutputFields(declaration collection.Declaration, withVector bool, withKey bool) []string {
	fields := []string{
		IDField,
		ContentField,
		RelativePathField,
		StartLineField,
		EndLineField,
		FileExtensionField,
		MetadataField,
		SplitPartField,
		ContentHashField,
		EmbeddingModelField,
	}
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
func readStoredRows(
	resultSet milvusclient.ResultSet,
	declaration collection.Declaration,
	withVector bool,
	withKey bool,
) ([]storedRow, error) {
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

func readStoredRow(
	resultSet milvusclient.ResultSet,
	declaration collection.Declaration,
	withVector bool,
	withKey bool,
	rowIndex int,
) (storedRow, error) {
	stored := storedRow{
		row: collection.Row{
			Scalars: make(map[string]collection.ScalarValue, len(declaration.Scalars)),
		},
	}
	var err error

	for field, target := range map[string]*string{
		IDField:            &stored.row.ID,
		ContentField:       &stored.row.Content,
		RelativePathField:  &stored.row.RelativePath,
		FileExtensionField: &stored.row.FileExtension,
		MetadataField:      &stored.row.Metadata,
	} {
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
	if stored.row.SplitPart, stored.row.SplitPartRecorded, err = SplitPartAt(
		resultSet.GetColumn(SplitPartField),
		rowIndex,
	); err != nil {
		return stored, err
	}

	if stored.contentHash, stored.contentHashSet, err = nullableStringAt(
		resultSet.GetColumn(ContentHashField),
		rowIndex,
	); err != nil {
		return stored, err
	}

	if stored.embeddingModel, stored.embeddingModelSet, err = nullableStringAt(
		resultSet.GetColumn(EmbeddingModelField),
		rowIndex,
	); err != nil {
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

func (store *SharedStore) compareCopyPage(
	ctx context.Context,
	target string,
	declaration collection.Declaration,
	sourceRows []storedRow,
	report *CopyReport,
) error {
	differences, err := store.copyPageDifferences(ctx, target, declaration, sourceRows)
	if err != nil {
		return err
	}

	for _, row := range differences.missing {
		report.MissingCount++
		report.Missing = appendSample(report.Missing, row.row.ID)
	}

	for _, row := range differences.mismatched {
		report.MismatchedCount++
		report.Mismatched = appendSample(report.Mismatched, row.row.ID)
	}

	report.DanglingCount += differences.dangling

	return nil
}

// copyPageDifferences reads the target rows with the source page's IDs and
// their vectors, then compares them with the source rows.
func (store *SharedStore) copyPageDifferences(
	ctx context.Context,
	target string,
	declaration collection.Declaration,
	sourceRows []storedRow,
) (pageDifferences, error) {
	differences := pageDifferences{missing: nil, mismatched: nil, dangling: 0}

	ids := make([]string, 0, len(sourceRows))
	for _, row := range sourceRows {
		ids = append(ids, row.row.ID)
	}

	resultSet, err := store.inner.client.Get(
		ctx,
		milvusclient.NewQueryOption(target).WithIDs(primaryKeyColumn(IDField, ids)).
			WithOutputFields(
				copyOutputFields(declaration, false, true)...,
			).
			WithConsistencyLevel(
				entity.ClStrong,
			),
	)
	if err != nil {
		return differences, WrapError(ctx, err, "read copied rows of "+target)
	}

	targetRows, err := readStoredRows(resultSet, declaration, false, true)
	if err != nil {
		return differences, err
	}

	byID := make(map[string]storedRow, len(targetRows))
	keySet := map[string]struct{}{}

	for _, row := range targetRows {
		byID[row.row.ID] = row
		keySet[row.vectorKey] = struct{}{}
	}

	vectorByKey, err := store.vectorsByKey(ctx, VectorCollectionName(target), sortedKeys(keySet))
	if err != nil {
		return differences, err
	}

	for _, sourceRow := range sourceRows {
		targetRow, found := byID[sourceRow.row.ID]
		if !found {
			differences.missing = append(differences.missing, sourceRow)

			continue
		}

		targetVector, found := vectorByKey[targetRow.vectorKey]
		if !found {
			differences.dangling++
		}

		if !found || !sameStoredRow(sourceRow, targetRow, targetVector, declaration) {
			differences.mismatched = append(differences.mismatched, sourceRow)
		}
	}

	return differences, nil
}

// pageIDs reads the primary key of each row in one result page.
func pageIDs(resultSet milvusclient.ResultSet) ([]string, error) {
	ids := make([]string, 0, resultSet.ResultCount)
	for rowIndex := range resultSet.ResultCount {
		id, err := requiredString(resultSet, IDField, rowIndex)
		if err != nil {
			return nil, err
		}

		ids = append(ids, id)
	}

	return ids, nil
}

// sameStoredRow reports whether a copied occurrence row and its vector store
// exactly the values of the source row.
func sameStoredRow(
	source storedRow,
	target storedRow,
	targetVector []float32,
	declaration collection.Declaration,
) bool {
	sourceKey := VectorKey(source.row.Vector)
	if target.vectorKey != sourceKey || VectorKey(targetVector) != sourceKey {
		return false
	}

	left := source.row
	right := target.row
	if left.Content != right.Content || left.RelativePath != right.RelativePath ||
		left.StartLine != right.StartLine ||
		left.EndLine != right.EndLine ||
		left.FileExtension != right.FileExtension ||
		left.Metadata != right.Metadata ||
		left.SplitPart != right.SplitPart ||
		left.SplitPartRecorded != right.SplitPartRecorded {
		return false
	}

	if source.contentHash != target.contentHash || source.contentHashSet != target.contentHashSet ||
		source.embeddingModel != target.embeddingModel || source.embeddingModelSet != target.embeddingModelSet {
		return false
	}

	for _, declared := range declaration.Scalars {
		if left.Scalars[declared.Name] != right.Scalars[declared.Name] {
			return false
		}
	}

	return true
}
