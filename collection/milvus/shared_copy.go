package milvus

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
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
	// copyMaxIDBatchSize bounds the IDs one page of MaxOccurrenceID reads.
	copyMaxIDBatchSize = 16_384
)

// CopyReport contains the result of [SharedStore.VerifyCopy]. Each count covers
// every row, and each ID list keeps at most copyReportSampleLimit IDs as a
// diagnostic sample. [SharedStore.ReconcileCopy] repairs every difference
// without using those lists.
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
// columns, the content hash and the embedding model with their null states,
// and the vector key of a shared occurrence row.
type storedRow struct {
	row               collection.Row
	contentHash       string
	contentHashSet    bool
	embeddingModel    string
	embeddingModelSet bool
	vectorKey         string
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
	if stored.embeddingModel, stored.embeddingModelSet, err = nullableStringAt(resultSet.GetColumn(EmbeddingModelField), rowIndex); err != nil {
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
		contentHashes:     make([]string, 0, len(rows)),
		contentHashSet:    make([]bool, 0, len(rows)),
		embeddingModels:   make([]string, 0, len(rows)),
		embeddingModelSet: make([]bool, 0, len(rows)),
	}
	for _, row := range rows {
		plain = append(plain, row.row)
		identity.contentHashes = append(identity.contentHashes, row.contentHash)
		identity.contentHashSet = append(identity.contentHashSet, row.contentHashSet)
		identity.embeddingModels = append(identity.embeddingModels, row.embeddingModel)
		identity.embeddingModelSet = append(identity.embeddingModelSet, row.embeddingModelSet)
	}
	store.write.Lock()
	defer store.write.Unlock()
	return store.writeRows(ctx, target, declaration, plain, identity)
}

// requireDeclarationCovers fails when the source stores a scalar column that
// the declaration does not list. A copy reads and compares only declared
// columns.
func (store *SharedStore) requireDeclarationCovers(ctx context.Context, source string, declaration collection.Declaration) error {
	stored, err := store.storedDeclaration(ctx, source, declaration.ItemIDColumn)
	if err != nil {
		return err
	}
	declared := make(map[string]struct{}, len(declaration.Scalars))
	for _, scalar := range declaration.Scalars {
		declared[scalar.Name] = struct{}{}
	}
	for _, scalar := range stored.Scalars {
		if _, found := declared[scalar.Name]; !found {
			err := fmt.Errorf("copy from %s: the declaration does not declare source column %s", source, scalar.Name)
			slog.ErrorContext(ctx, "collection.shared_copy_declaration_incomplete", "source", source, "err", err)
			return err
		}
	}
	return nil
}

// CopyFrom copies every row of the single-collection source into the shared
// target. It keeps each row's ID, every column, the content hash with its null
// state, the embedding model, and the exact vector bytes. It never embeds. A
// repeated copy rewrites the same rows. progress receives the rows each phase
// has processed after each page.
func (store *SharedStore) CopyFrom(ctx context.Context, source string, target string, declaration collection.Declaration, progress ProgressFunc) (CopyResult, error) {
	return store.CopyFromAfter(ctx, source, target, declaration, "", progress)
}

// CopyResult counts the rows one copy wrote: every copied occurrence row, and
// the vector rows the copy wrote because the target did not store them with
// their exact superset.
type CopyResult struct {
	Occurrences    int64
	VectorsWritten int64
}

// CopyFromAfter copies source rows with primary keys after afterID in VarChar
// string order. An empty afterID copies every row, like [SharedStore.CopyFrom].
//
// The first pass writes each occurrence row after afterID. The second pass
// reads every source row's vector, including vectors that an earlier stopped
// copy omitted for rows at or below afterID. It writes each distinct vector
// once with every occurrence that references it, unless the vector row
// already stores that complete set.
//
// A source row added after an earlier pass can have an ID at or below
// afterID. CopyFromAfter does not copy that row. [SharedStore.ReconcileCopy]
// copies it.
func (store *SharedStore) CopyFromAfter(ctx context.Context, source string, target string, declaration collection.Declaration, afterID string, progress ProgressFunc) (CopyResult, error) {
	result := CopyResult{Occurrences: 0, VectorsWritten: 0}
	if err := store.requireDeclarationCovers(ctx, source, declaration); err != nil {
		return result, err
	}
	expression := IDField + ` != ""`
	if afterID != "" {
		expression = IDField + ` > "` + collection.EscapeString(afterID) + `"`
	}
	occurrences := strings.TrimSpace(target)
	copyTotal, err := store.countRows(ctx, source, expression)
	if err != nil {
		return result, err
	}
	progress.report(CopyPhaseOccurrences, 0, copyTotal)
	err = store.iterateBatch(ctx, source, expression, copyOutputFields(declaration, true, false), copyBatchSize, func(resultSet milvusclient.ResultSet) error {
		rows, err := readStoredRows(resultSet, declaration, true, false)
		if err != nil {
			return err
		}
		if err := store.writeStoredOccurrences(ctx, occurrences, declaration, rows); err != nil {
			return err
		}
		result.Occurrences += int64(len(rows))
		progress.report(CopyPhaseOccurrences, result.Occurrences, copyTotal)
		return nil
	})
	if err != nil {
		return result, err
	}
	result.VectorsWritten, err = store.writeCopiedVectors(ctx, source, occurrences, declaration.Scalars, IDField+` != ""`, progress)
	slog.InfoContext(ctx, "collection.shared_copy_vectors_written", "target", occurrences, "occurrences", result.Occurrences, "vectors_written", result.VectorsWritten)
	return result, err
}

// writeStoredOccurrences writes the occurrence rows of copied rows with the
// identity each source row stores. It does not write vector rows.
func (store *SharedStore) writeStoredOccurrences(ctx context.Context, occurrences string, declaration collection.Declaration, rows []storedRow) error {
	if len(rows) == 0 {
		return nil
	}
	plain := make([]collection.Row, 0, len(rows))
	keys := make([]string, 0, len(rows))
	identity := rowIdentity{
		contentHashes:     make([]string, 0, len(rows)),
		contentHashSet:    make([]bool, 0, len(rows)),
		embeddingModels:   make([]string, 0, len(rows)),
		embeddingModelSet: make([]bool, 0, len(rows)),
	}
	for _, row := range rows {
		plain = append(plain, row.row)
		keys = append(keys, VectorKey(row.row.Vector))
		identity.contentHashes = append(identity.contentHashes, row.contentHash)
		identity.contentHashSet = append(identity.contentHashSet, row.contentHashSet)
		identity.embeddingModels = append(identity.embeddingModels, row.embeddingModel)
		identity.embeddingModelSet = append(identity.embeddingModelSet, row.embeddingModelSet)
	}
	store.write.Lock()
	defer store.write.Unlock()
	for start := 0; start < len(plain); start += sharedWriteBatchSize {
		end := min(start+sharedWriteBatchSize, len(plain))
		option, err := buildOccurrenceWriteOption(ctx, occurrences, plain[start:end], keys[start:end], declaration.Scalars, identity.slice(start, end))
		if err != nil {
			return err
		}
		if _, err := store.inner.client.Upsert(ctx, option); err != nil {
			return WrapError(ctx, err, "upsert Milvus occurrences into "+occurrences)
		}
	}
	return nil
}

// writeCopiedVectors reads the vectors of the source rows an expression
// matches and writes each distinct vector once into the vector collection of
// occurrences, with the superset of every stored occurrence that references
// it. It skips a vector row that already stores that superset and returns the
// number of written vector rows.
func (store *SharedStore) writeCopiedVectors(ctx context.Context, source string, occurrences string, scalars []collection.ScalarColumn, expression string, progress ProgressFunc) (int64, error) {
	vectors := VectorCollectionName(occurrences)
	total, err := store.countRows(ctx, source, expression)
	if err != nil {
		return 0, err
	}
	progress.report(CopyPhaseVectors, 0, total)
	var scanned int64
	// done stores the SHA-256 digest of each finished vector key instead of
	// its 64-character hex string.
	done := map[[sha256.Size]byte]struct{}{}
	var written int64
	err = store.iterateBatch(ctx, source, expression, []string{IDField, DenseVectorField}, copyBatchSize, func(resultSet milvusclient.ResultSet) error {
		vectorColumn := resultSet.GetColumn(DenseVectorField)
		if vectorColumn == nil {
			return collection.ErrSearchResultIncomplete
		}
		scanned += int64(resultSet.ResultCount)
		records := map[string]*vectorRecord{}
		dimension := 0
		for rowIndex := range resultSet.ResultCount {
			vector, err := VectorAt(vectorColumn, rowIndex)
			if err != nil {
				return err
			}
			key := VectorKey(vector)
			digest, err := vectorKeyDigest(key)
			if err != nil {
				return err
			}
			if _, finished := done[digest]; finished {
				continue
			}
			done[digest] = struct{}{}
			dimension = len(vector)
			records[key] = newVectorRecord(key, vector, scalars)
		}
		if len(records) > 0 {
			pageWritten, err := store.writeExactVectors(ctx, occurrences, vectors, dimension, scalars, records)
			written += pageWritten
			if err != nil {
				return err
			}
		}
		progress.report(CopyPhaseVectors, scanned, total)
		return nil
	})
	return written, err
}

// vectorKeyDigest decodes a hex vector key into its SHA-256 digest bytes.
func vectorKeyDigest(key string) ([sha256.Size]byte, error) {
	var digest [sha256.Size]byte
	decoded, err := hex.DecodeString(key)
	if err != nil || len(decoded) != sha256.Size {
		return digest, fmt.Errorf("vector key %q is not a hex SHA-256 digest", key)
	}
	copy(digest[:], decoded)
	return digest, nil
}

// writeExactVectors fills records with the superset of every stored occurrence
// that references each record and writes the records that the vector
// collection does not already store with that superset.
func (store *SharedStore) writeExactVectors(ctx context.Context, occurrences string, vectors string, dimension int, scalars []collection.ScalarColumn, records map[string]*vectorRecord) (int64, error) {
	store.write.Lock()
	defer store.write.Unlock()
	keys := sortedRecordKeys(records)
	outputFields := []string{VectorKeyField}
	for _, declared := range scalars {
		outputFields = append(outputFields, declared.Name)
	}
	err := store.iterate(ctx, occurrences, vectorKeysExpression(keys), outputFields, func(resultSet milvusclient.ResultSet) error {
		return addOccurrenceCells(resultSet, scalars, records)
	})
	if err != nil {
		return 0, err
	}
	stored, err := store.storedSupersets(ctx, vectors, scalars, keys)
	if err != nil {
		return 0, err
	}
	ordered := make([]*vectorRecord, 0, len(records))
	for _, key := range keys {
		if storedRecord, found := stored[key]; found && storedRecord.equalRecord(records[key]) {
			continue
		}
		ordered = append(ordered, records[key])
	}
	for start := 0; start < len(ordered); start += sharedWriteBatchSize {
		batch := ordered[start:min(start+sharedWriteBatchSize, len(ordered))]
		option, err := vectorUpsertOption(vectors, dimension, scalars, batch)
		if err != nil {
			return 0, err
		}
		if _, err := store.inner.client.Upsert(ctx, option); err != nil {
			return 0, WrapError(ctx, err, "upsert Milvus vectors into "+vectors)
		}
	}
	return int64(len(ordered)), nil
}

// MaxOccurrenceID scans every primary key of the shared occurrence collection
// and returns the largest one in string order, the order
// [SharedStore.CopyFromAfter] uses. It returns an empty string for an empty
// collection, and CopyFromAfter copies every row after an empty string.
func (store *SharedStore) MaxOccurrenceID(ctx context.Context, target string) (string, error) {
	largest := ""
	err := store.iterateBatch(ctx, strings.TrimSpace(target), IDField+` != ""`, []string{IDField}, copyMaxIDBatchSize, func(resultSet milvusclient.ResultSet) error {
		for rowIndex := range resultSet.ResultCount {
			id, err := requiredString(resultSet, IDField, rowIndex)
			if err != nil {
				return err
			}
			if id > largest {
				largest = id
			}
		}
		return nil
	})
	return largest, err
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
// progress receives the rows each verification phase has processed after each
// page.
func (store *SharedStore) VerifyCopy(ctx context.Context, source string, target string, declaration collection.Declaration, progress ProgressFunc) (CopyReport, error) {
	report := CopyReport{Missing: nil, Mismatched: nil, Extra: nil}
	if err := store.requireDeclarationCovers(ctx, source, declaration); err != nil {
		return report, err
	}
	sourceTotal, err := store.countRows(ctx, source, IDField+` != ""`)
	if err != nil {
		return report, err
	}
	progress.report(VerifyPhaseRows, 0, sourceTotal)
	err = store.iterateBatch(ctx, source, IDField+` != ""`, copyOutputFields(declaration, true, false), copyBatchSize, func(resultSet milvusclient.ResultSet) error {
		sourceRows, err := readStoredRows(resultSet, declaration, true, false)
		if err != nil {
			return err
		}
		report.SourceRows += int64(len(sourceRows))
		if err := store.compareCopyPage(ctx, target, declaration, sourceRows, &report); err != nil {
			return err
		}
		progress.report(VerifyPhaseRows, report.SourceRows, sourceTotal)
		return nil
	})
	if err != nil {
		return report, err
	}
	if err := store.countExtraRows(ctx, source, target, &report, progress); err != nil {
		return report, err
	}
	if err := store.countUnreferencedVectors(ctx, target, &report, progress); err != nil {
		return report, err
	}
	slog.InfoContext(ctx, "collection.shared_copy_verified", "source", source, "target", target,
		"source_rows", report.SourceRows, "target_rows", report.TargetRows, "target_vectors", report.TargetVectors,
		"missing", report.MissingCount, "mismatched", report.MismatchedCount, "extra", report.ExtraCount,
		"dangling", report.DanglingCount, "unreferenced_vectors", report.UnreferencedVectors)
	return report, nil
}

func (store *SharedStore) compareCopyPage(ctx context.Context, target string, declaration collection.Declaration, sourceRows []storedRow, report *CopyReport) error {
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

// pageDifferences stores source rows from one page that the target lacks or
// stores with a different value. It also counts target rows on that page that
// reference a vector the vector collection lacks.
type pageDifferences struct {
	missing    []storedRow
	mismatched []storedRow
	dangling   int64
}

// copyPageDifferences reads the target rows with the source page's IDs and
// their vectors, then compares them with the source rows.
func (store *SharedStore) copyPageDifferences(ctx context.Context, target string, declaration collection.Declaration, sourceRows []storedRow) (pageDifferences, error) {
	differences := pageDifferences{missing: nil, mismatched: nil, dangling: 0}
	ids := make([]string, 0, len(sourceRows))
	for _, row := range sourceRows {
		ids = append(ids, row.row.ID)
	}
	resultSet, err := store.inner.client.Get(ctx, milvusclient.NewQueryOption(target).
		WithIDs(primaryKeyColumn(IDField, ids)).
		WithOutputFields(copyOutputFields(declaration, false, true)...).
		WithConsistencyLevel(entity.ClStrong))
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

// ReconcileResult counts the rows one [SharedStore.ReconcileCopy] read and the
// differences it repaired. The counts cover every row without a sample limit.
type ReconcileResult struct {
	SourceRows          int64
	TargetRows          int64
	Missing             int64
	Mismatched          int64
	Dangling            int64
	Extra               int64
	UnreferencedVectors int64
}

// Repaired returns the sum of Missing, Mismatched, Extra, and
// UnreferencedVectors.
func (result ReconcileResult) Repaired() int64 {
	return result.Missing + result.Mismatched + result.Extra + result.UnreferencedVectors
}

// ReconcileCopy makes the shared target equal to the single-collection source
// in three passes. The first pass compares every source page with the target
// as [SharedStore.VerifyCopy] does, then writes each missing or different row
// with its exact vector bytes. The second pass removes target rows that the
// source lacks. The third pass removes vectors that target rows do not
// reference. ReconcileCopy never changes the source. After each page, progress
// receives the number of rows that the current pass has processed.
// Run [SharedStore.VerifyCopy] afterwards to confirm equality.
func (store *SharedStore) ReconcileCopy(ctx context.Context, source string, target string, declaration collection.Declaration, progress ProgressFunc) (ReconcileResult, error) {
	result := ReconcileResult{}
	if err := store.requireDeclarationCovers(ctx, source, declaration); err != nil {
		return result, err
	}
	occurrences := strings.TrimSpace(target)
	sourceTotal, err := store.countRows(ctx, source, IDField+` != ""`)
	if err != nil {
		return result, err
	}
	progress.report(ReconcilePhaseRows, 0, sourceTotal)
	err = store.iterateBatch(ctx, source, IDField+` != ""`, copyOutputFields(declaration, true, false), copyBatchSize, func(resultSet milvusclient.ResultSet) error {
		sourceRows, err := readStoredRows(resultSet, declaration, true, false)
		if err != nil {
			return err
		}
		differences, err := store.copyPageDifferences(ctx, occurrences, declaration, sourceRows)
		if err != nil {
			return err
		}
		rewrite := make([]storedRow, 0, len(differences.missing)+len(differences.mismatched))
		rewrite = append(rewrite, differences.missing...)
		rewrite = append(rewrite, differences.mismatched...)
		if err := store.writeStoredRows(ctx, occurrences, declaration, rewrite); err != nil {
			return err
		}
		result.SourceRows += int64(len(sourceRows))
		result.Missing += int64(len(differences.missing))
		result.Mismatched += int64(len(differences.mismatched))
		result.Dangling += differences.dangling
		progress.report(ReconcilePhaseRows, min(result.SourceRows, sourceTotal), sourceTotal)
		return nil
	})
	if err != nil {
		return result, err
	}
	if err := store.removeExtraRows(ctx, source, occurrences, &result, progress); err != nil {
		return result, err
	}
	result.UnreferencedVectors, err = store.removeUnreferencedVectors(ctx, occurrences, func(done int64, total int64) {
		progress.report(ReconcilePhaseVectors, done, total)
	})
	if err != nil {
		return result, err
	}
	slog.InfoContext(ctx, "collection.shared_copy_reconciled", "source", source, "target", occurrences,
		"source_rows", result.SourceRows, "target_rows", result.TargetRows,
		"missing", result.Missing, "mismatched", result.Mismatched, "dangling", result.Dangling,
		"extra", result.Extra, "unreferenced_vectors", result.UnreferencedVectors)
	return result, nil
}

// removeExtraRows reads every target page and removes each row on that page
// with an ID the source does not store.
func (store *SharedStore) removeExtraRows(ctx context.Context, source string, target string, result *ReconcileResult, progress ProgressFunc) error {
	total, err := store.countRows(ctx, target, IDField+` != ""`)
	if err != nil {
		return err
	}
	progress.report(ReconcilePhaseExtraRows, 0, total)
	return store.iterateBatch(ctx, target, IDField+` != ""`, []string{IDField}, copyBatchSize, func(resultSet milvusclient.ResultSet) error {
		ids, err := pageIDs(resultSet)
		if err != nil {
			return err
		}
		extra, err := store.idsAbsentFrom(ctx, source, ids)
		if err != nil {
			return err
		}
		if len(extra) > 0 {
			if _, err := store.RemoveOccurrences(ctx, target, extra); err != nil {
				return err
			}
		}
		result.TargetRows += int64(len(ids))
		result.Extra += int64(len(extra))
		progress.report(ReconcilePhaseExtraRows, min(result.TargetRows, total), total)
		return nil
	})
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

// idsAbsentFrom returns the input IDs that the collection does not store, in
// input order.
func (store *SharedStore) idsAbsentFrom(ctx context.Context, collectionName string, ids []string) ([]string, error) {
	present, err := store.inner.client.Get(ctx, milvusclient.NewQueryOption(collectionName).
		WithIDs(primaryKeyColumn(IDField, ids)).
		WithOutputFields(IDField).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		return nil, WrapError(ctx, err, "read source IDs of "+collectionName)
	}
	found := make(map[string]struct{}, present.ResultCount)
	for rowIndex := range present.ResultCount {
		id, idErr := requiredString(present, IDField, rowIndex)
		if idErr != nil {
			return nil, idErr
		}
		found[id] = struct{}{}
	}
	absent := make([]string, 0)
	for _, id := range ids {
		if _, exists := found[id]; !exists {
			absent = append(absent, id)
		}
	}
	return absent, nil
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

// countExtraRows counts the target rows and the target row IDs the source
// lacks.
func (store *SharedStore) countExtraRows(ctx context.Context, source string, target string, report *CopyReport, progress ProgressFunc) error {
	total, err := store.countRows(ctx, target, IDField+` != ""`)
	if err != nil {
		return err
	}
	progress.report(VerifyPhaseExtraRows, 0, total)
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
		progress.report(VerifyPhaseExtraRows, report.TargetRows, total)
		return nil
	})
}

// countUnreferencedVectors counts the target vectors and the vectors that no
// target row references. It reads the vector key of every occurrence row once
// into a set of SHA-256 digests, then reads every vector key once and checks it
// against that set.
func (store *SharedStore) countUnreferencedVectors(ctx context.Context, target string, report *CopyReport, progress ProgressFunc) error {
	vectors := VectorCollectionName(target)
	referenceTotal, err := store.countRows(ctx, target, IDField+` != ""`)
	if err != nil {
		return err
	}
	vectorTotal, err := store.countRows(ctx, vectors, VectorKeyField+` != ""`)
	if err != nil {
		return err
	}
	referenced := map[[sha256.Size]byte]struct{}{}
	var referenceRows int64
	progress.report(VerifyPhaseReferences, 0, referenceTotal)
	err = store.iterateBatch(ctx, target, IDField+` != ""`, []string{VectorKeyField}, copyMaxIDBatchSize, func(resultSet milvusclient.ResultSet) error {
		referenceRows += int64(resultSet.ResultCount)
		for rowIndex := range resultSet.ResultCount {
			key, keyErr := requiredString(resultSet, VectorKeyField, rowIndex)
			if keyErr != nil {
				return keyErr
			}
			digest, digestErr := vectorKeyDigest(key)
			if digestErr != nil {
				return digestErr
			}
			referenced[digest] = struct{}{}
		}
		progress.report(VerifyPhaseReferences, referenceRows, referenceTotal)
		return nil
	})
	if err != nil {
		return err
	}
	progress.report(VerifyPhaseVectors, 0, vectorTotal)
	return store.iterateBatch(ctx, vectors, VectorKeyField+` != ""`, []string{VectorKeyField}, copyMaxIDBatchSize, func(resultSet milvusclient.ResultSet) error {
		for rowIndex := range resultSet.ResultCount {
			key, keyErr := requiredString(resultSet, VectorKeyField, rowIndex)
			if keyErr != nil {
				return keyErr
			}
			digest, digestErr := vectorKeyDigest(key)
			if digestErr != nil {
				return digestErr
			}
			report.TargetVectors++
			if _, found := referenced[digest]; !found {
				report.UnreferencedVectors++
			}
		}
		progress.report(VerifyPhaseVectors, report.TargetVectors, vectorTotal)
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
