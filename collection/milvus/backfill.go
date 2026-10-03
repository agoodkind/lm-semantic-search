package milvus

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

// scalarBackfillBatchSize bounds the rows one backfill page reads and one
// partial update writes. A backfill page reads no vector.
const scalarBackfillBatchSize = 1000

// scalarBackfillPage is one page of a scalar backfill. ids and rows list the
// rows that change, and the Scalars of each row store its filled backfill
// values. changed and orphan count the page rows that need the backfill.
type scalarBackfillPage struct {
	ids     []string
	rows    []collection.Row
	changed int
	orphan  int
}

// BackfillScalars writes backfill values into the declared columns that are null
// or an empty string on the rows of streamed items. It reads only the rows that
// need the backfill. A Milvus partial update writes the backfill columns alone
// and keeps every other column, the vector, and the row key as stored. It
// returns the rows that need the backfill: changed counts the rows of streamed
// items, and orphan counts the rest, which it leaves unchanged. A dry run counts
// and writes nothing. The caller loads the collection first, because Milvus
// serves the filtered read on a loaded collection only.
func (store *Store) BackfillScalars(ctx context.Context, collectionName string, backfill collection.ScalarBackfill) (int, int, error) {
	trimmedName := strings.TrimSpace(collectionName)
	if trimmedName == "" {
		return 0, 0, errors.New("collection name is required")
	}
	if len(backfill.Columns) == 0 {
		return 0, 0, errors.New("scalar backfill lists no column")
	}
	// Strong consistency makes the read include rows an ingest wrote just before
	// the backfill.
	iterator, err := store.client.QueryIterator(ctx, milvusclient.NewQueryIteratorOption(trimmedName).
		WithBatchSize(scalarBackfillBatchSize).
		WithFilter(scalarBackfillFilter(backfill.Columns)).
		WithOutputFields(scalarBackfillOutputFields(backfill)...).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		return 0, 0, SearchError(ctx, "open scalar backfill iterator for", trimmedName, err)
	}
	changed := 0
	orphan := 0
	written := 0
	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			return changed, orphan, SearchError(ctx, "iterate scalar backfill of", trimmedName, nextErr)
		}
		page, pageErr := readScalarBackfillPage(resultSet, backfill)
		if pageErr != nil {
			return changed, orphan, pageErr
		}
		changed += page.changed
		orphan += page.orphan
		if backfill.DryRun || len(page.ids) == 0 {
			continue
		}
		if err := store.writeScalarBackfill(ctx, trimmedName, backfill.Columns, page); err != nil {
			return changed, orphan, err
		}
		written += len(page.ids)
	}
	slog.InfoContext(ctx, "collection.scalar_backfill_completed",
		"collection", trimmedName,
		"changed", changed,
		"orphan", orphan,
		"written", written,
		"dry_run", backfill.DryRun,
	)
	return changed, orphan, nil
}

// scalarBackfillFilter renders the Milvus filter that matches every row with a
// backfill column that is null or an empty string. A column that is neither
// nullable nor a string column is never missing and adds no clause.
func scalarBackfillFilter(columns []collection.ScalarColumn) string {
	clauses := make([]string, 0, len(columns))
	for _, column := range columns {
		missing := make([]string, 0, 2)
		if column.Nullable {
			missing = append(missing, column.Name+" is null")
		}
		if column.Type == collection.ScalarTypeString {
			missing = append(missing, column.Name+` == ""`)
		}
		if len(missing) == 0 {
			continue
		}
		clauses = append(clauses, "("+strings.Join(missing, " or ")+")")
	}
	return strings.Join(clauses, " or ")
}

// scalarBackfillOutputFields lists the fields one backfill page reads: the
// primary key, the row key, the item ID column, and the backfill columns.
func scalarBackfillOutputFields(backfill collection.ScalarBackfill) []string {
	fields := []string{IDField, RelativePathField, backfill.ItemColumn}
	for _, column := range backfill.Columns {
		fields = append(fields, column.Name)
	}
	return fields
}

// readScalarBackfillPage counts the rows of one iterator page that need the
// backfill and collects the filled values of the rows that change.
func readScalarBackfillPage(resultSet milvusclient.ResultSet, backfill collection.ScalarBackfill) (scalarBackfillPage, error) {
	page := scalarBackfillPage{ids: nil, rows: nil, changed: 0, orphan: 0}
	idColumn := resultSet.GetColumn(IDField)
	relativePathColumn := resultSet.GetColumn(RelativePathField)
	itemIDColumn := resultSet.GetColumn(backfill.ItemColumn)
	if idColumn == nil || relativePathColumn == nil || itemIDColumn == nil {
		return page, collection.ErrSearchResultIncomplete
	}
	for rowIndex := range resultSet.ResultCount {
		stored, err := storedBackfillValues(resultSet, backfill.Columns, rowIndex)
		if err != nil {
			return page, err
		}
		if !backfill.Needs(stored) {
			continue
		}
		relativePath, err := relativePathColumn.GetAsString(rowIndex)
		if err != nil {
			return page, rowReadError("relative path", rowIndex, err)
		}
		itemID, err := optionalStringAt(itemIDColumn, rowIndex)
		if err != nil {
			return page, rowReadError("item id", rowIndex, err)
		}
		values, streamed := backfill.ItemValues(itemID, relativePath)
		if !streamed {
			page.orphan++
			continue
		}
		page.changed++
		filled, changed := backfill.Filled(stored, values)
		if !changed {
			continue
		}
		id, err := idColumn.GetAsString(rowIndex)
		if err != nil {
			return page, rowReadError("id", rowIndex, err)
		}
		page.ids = append(page.ids, id)
		page.rows = append(page.rows, scalarBackfillRow(relativePath, filled))
	}
	return page, nil
}

// storedBackfillValues reads the stored value of every backfill column at one
// row.
func storedBackfillValues(resultSet milvusclient.ResultSet, columns []collection.ScalarColumn, rowIndex int) (map[string]collection.ScalarValue, error) {
	stored := make(map[string]collection.ScalarValue, len(columns))
	for _, column := range columns {
		value, err := ScalarValueAt(resultSet.GetColumn(column.Name), column, rowIndex)
		if err != nil {
			slog.Error("read stored backfill column failed", "column", column.Name, "index", rowIndex, "err", err)
			return nil, fmt.Errorf("read stored backfill column %s: %w", column.Name, err)
		}
		stored[column.Name] = value
	}
	return stored, nil
}

// scalarBackfillRow builds the row that DeclaredScalarInsertColumns reads for
// one row. Scalars stores the row's filled backfill values, and a build error
// identifies the row by relativePath.
func scalarBackfillRow(relativePath string, scalars map[string]collection.ScalarValue) collection.Row {
	return collection.Row{
		ID:                "",
		Content:           "",
		RelativePath:      relativePath,
		StartLine:         0,
		EndLine:           0,
		FileExtension:     "",
		Metadata:          "",
		SplitPart:         0,
		SplitPartRecorded: false,
		Vector:            nil,
		Scalars:           scalars,
	}
}

// writeScalarBackfill writes the backfill columns of one page through a Milvus
// partial update. The update sends the primary key and the backfill columns, and
// Milvus keeps every other stored field of each row.
func (store *Store) writeScalarBackfill(ctx context.Context, collectionName string, columns []collection.ScalarColumn, page scalarBackfillPage) error {
	declaredColumns, err := DeclaredScalarInsertColumns(collectionName, columns, page.rows)
	if err != nil {
		slog.ErrorContext(ctx, "build scalar backfill columns failed", "collection", collectionName, "err", err)
		return fmt.Errorf("build backfill columns for %s: %w", collectionName, err)
	}
	option := milvusclient.NewColumnBasedInsertOption(collectionName).
		WithVarcharColumn(IDField, page.ids).
		WithColumns(declaredColumns...).
		WithPartialUpdate(true)
	if _, err := store.client.Upsert(ctx, option); err != nil {
		return WrapError(ctx, err, "backfill scalars in "+collectionName)
	}
	return nil
}
