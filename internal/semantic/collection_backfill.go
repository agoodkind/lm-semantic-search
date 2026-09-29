package semantic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/grpc/peer"
)

// scalarBackfillBatchSize bounds the rows one backfill page reads and one
// partial update writes. A backfill page reads no vector.
const scalarBackfillBatchSize = 1000

// legacyConversationFamilies are the relativePath family prefixes of
// conversation rows. A conversation row written before the conversationId
// column existed stores its conversation id only in this path.
var legacyConversationFamilies = []string{"conv/", "convtool/", "convthink/"}

// ScalarBackfill is one backfill of declared scalar columns in a document
// collection. A row needs the backfill when one of Columns is null, or an
// empty string, on the row. ItemColumn is the declared item id column, and
// Values maps an item id to the value of every column in Columns. Conversation
// marks the conversation declaration. A conversation row without an item id
// then belongs to the item id that follows its conv/, convtool/, or convthink/
// path prefix, and the local store keeps the conversation columns in its
// conversation row fields. DryRun counts rows and writes nothing.
type ScalarBackfill struct {
	ItemColumn   string
	Columns      []model.ScalarColumn
	Values       map[string]map[string]model.ScalarValue
	Conversation bool
	DryRun       bool
}

// ScalarValueMissing reports whether a backfill fills a stored value: a null
// value or an empty string.
func ScalarValueMissing(value model.ScalarValue) bool {
	return value.Null || (value.Type == model.ScalarTypeString && value.String == "")
}

// Needs reports whether a row with the stored backfill column values needs the
// backfill, because one of them is null or an empty string.
func (backfill ScalarBackfill) Needs(stored map[string]model.ScalarValue) bool {
	for _, column := range backfill.Columns {
		if ScalarValueMissing(stored[column.Name]) {
			return true
		}
	}
	return false
}

// ItemValues returns the backfill values of the item that owns a row. itemID
// is the row's item id column value, and it is empty when the column is null.
// A conversation row without an item id belongs to the longest streamed item
// id that follows its conv/, convtool/, or convthink/ path prefix.
func (backfill ScalarBackfill) ItemValues(itemID string, relativePath string) (map[string]model.ScalarValue, bool) {
	if itemID == "" && backfill.Conversation {
		itemID = backfill.legacyConversationItem(relativePath)
	}
	if itemID == "" {
		return nil, false
	}
	values, streamed := backfill.Values[itemID]
	return values, streamed
}

// legacyConversationItem returns the longest streamed item id that follows the
// family prefix of a conversation row path, or an empty string.
func (backfill ScalarBackfill) legacyConversationItem(relativePath string) string {
	for _, family := range legacyConversationFamilies {
		remainder, found := strings.CutPrefix(relativePath, family)
		if !found {
			continue
		}
		for end := strings.LastIndexByte(remainder, '/'); end > 0; end = strings.LastIndexByte(remainder[:end], '/') {
			if _, streamed := backfill.Values[remainder[:end]]; streamed {
				return remainder[:end]
			}
		}
		return ""
	}
	return ""
}

// Filled returns the backfill column values a row stores after the backfill.
// A missing stored value takes the item's value, and every other stored value
// stays. changed reports whether any value differs from the stored one.
func (backfill ScalarBackfill) Filled(stored map[string]model.ScalarValue, values map[string]model.ScalarValue) (map[string]model.ScalarValue, bool) {
	filled := make(map[string]model.ScalarValue, len(backfill.Columns))
	changed := false
	for _, column := range backfill.Columns {
		value := stored[column.Name]
		if ScalarValueMissing(value) && values[column.Name] != value {
			value = values[column.Name]
			changed = true
		}
		filled[column.Name] = value
	}
	return filled, changed
}

// scalarBackfillPage is one page of a Milvus scalar backfill. ids and rows list
// the rows that change, and the Scalars of each row store its filled backfill
// values. changed and orphan count the page rows that need the backfill.
type scalarBackfillPage struct {
	ids     []string
	rows    []model.StoredChunk
	changed int
	orphan  int
}

// BackfillCollectionScalars writes backfill values into the declared columns
// that are null or an empty string on the rows of streamed items. It reads only
// the rows that need the backfill. A Milvus partial update writes the backfill
// columns alone and keeps every other column, the vector, and the row key as
// stored. BackfillCollectionScalars returns the rows that need the backfill:
// changed counts the rows of streamed items, and orphan counts the rest, which
// it leaves unchanged. A dry run counts and writes nothing. A missing
// collection returns ErrCollectionMissing.
func (service *Service) BackfillCollectionScalars(ctx context.Context, collectionName string, backfill ScalarBackfill) (int, int, error) {
	peerInfo, _ := peer.FromContext(ctx)
	if !service.Available() {
		return 0, 0, ErrUnavailable
	}
	hasCollection, err := service.hasCollection(ctx, collectionName, "check Milvus collection "+collectionName)
	if err != nil {
		return 0, 0, err
	}
	if !hasCollection {
		return 0, 0, ErrCollectionMissing
	}
	if err := service.PrepareCollection(ctx, collectionName); err != nil {
		return 0, 0, err
	}
	lease, err := service.AcquireCollection(ctx, collectionName)
	if err != nil {
		return 0, 0, err
	}
	defer lease.Release()
	// Strong consistency makes the read include rows an ingest job wrote just
	// before the backfill.
	iterator, err := service.milvus.QueryIterator(ctx, milvusclient.NewQueryIteratorOption(collectionName).
		WithBatchSize(scalarBackfillBatchSize).
		WithFilter(scalarBackfillFilter(backfill.Columns)).
		WithOutputFields(scalarBackfillOutputFields(backfill)...).
		WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		slog.ErrorContext(ctx, "open scalar backfill iterator failed", "collection", collectionName, "peer", peerInfo.String(), "err", err)
		return 0, 0, fmt.Errorf("open scalar backfill iterator for %s: %w", collectionName, err)
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
			slog.ErrorContext(ctx, "scalar backfill iterator next failed", "collection", collectionName, "changed", changed, "peer", peerInfo.String(), "err", nextErr)
			return changed, orphan, fmt.Errorf("iterate %s for scalar backfill: %w", collectionName, nextErr)
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
		if err := service.writeScalarBackfill(ctx, collectionName, backfill.Columns, page); err != nil {
			return changed, orphan, err
		}
		written += len(page.ids)
	}
	slog.InfoContext(ctx, "semantic.collection_scalar_backfill_complete", "collection", collectionName, "changed", changed, "orphan", orphan, "written", written, "dry_run", backfill.DryRun, "peer", peerInfo.String())
	return changed, orphan, nil
}

// scalarBackfillFilter renders the Milvus filter that matches every row with a
// backfill column that is null or an empty string. A column that is neither
// nullable nor a string column is never missing and adds no clause.
func scalarBackfillFilter(columns []model.ScalarColumn) string {
	clauses := make([]string, 0, len(columns))
	for _, column := range columns {
		missing := make([]string, 0, 2)
		if column.Nullable {
			missing = append(missing, column.Name+" is null")
		}
		if column.Type == model.ScalarTypeString {
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
// primary key, the row key, the item id column, and the backfill columns.
func scalarBackfillOutputFields(backfill ScalarBackfill) []string {
	fields := []string{idFieldName, relativePathFieldName, backfill.ItemColumn}
	for _, column := range backfill.Columns {
		fields = append(fields, column.Name)
	}
	return fields
}

// readScalarBackfillPage counts the rows of one iterator page that need the
// backfill and collects the filled values of the rows that change.
func readScalarBackfillPage(resultSet milvusclient.ResultSet, backfill ScalarBackfill) (scalarBackfillPage, error) {
	page := scalarBackfillPage{ids: nil, rows: nil, changed: 0, orphan: 0}
	idColumn := resultSet.GetColumn(idFieldName)
	relativePathColumn := resultSet.GetColumn(relativePathFieldName)
	itemIDColumn := resultSet.GetColumn(backfill.ItemColumn)
	if idColumn == nil || relativePathColumn == nil || itemIDColumn == nil {
		return page, ErrSearchResultIncomplete
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
			slog.Error("read scalar backfill relative path failed", "index", rowIndex, "err", err)
			return page, fmt.Errorf("read relative path column at %d: %w", rowIndex, err)
		}
		itemID, _, err := readOptionalStringAt(itemIDColumn, rowIndex)
		if err != nil {
			return page, err
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
			slog.Error("read scalar backfill id failed", "index", rowIndex, "err", err)
			return page, fmt.Errorf("read id column at %d: %w", rowIndex, err)
		}
		page.ids = append(page.ids, id)
		page.rows = append(page.rows, scalarBackfillRow(relativePath, filled))
	}
	return page, nil
}

// storedBackfillValues reads the stored value of every backfill column at one
// row.
func storedBackfillValues(resultSet milvusclient.ResultSet, columns []model.ScalarColumn, rowIndex int) (map[string]model.ScalarValue, error) {
	stored := make(map[string]model.ScalarValue, len(columns))
	for _, column := range columns {
		value, err := declaredScalarValueAt(resultSet.GetColumn(column.Name), column, rowIndex)
		if err != nil {
			return nil, err
		}
		stored[column.Name] = value
	}
	return stored, nil
}

// scalarBackfillRow builds the stored chunk that declaredScalarInsertColumns
// reads for one row. Scalars stores the row's filled backfill values, and a
// build error identifies the row by relativePath.
func scalarBackfillRow(relativePath string, scalars map[string]model.ScalarValue) model.StoredChunk {
	return model.StoredChunk{
		Content:              "",
		RelativePath:         relativePath,
		StartLine:            0,
		EndLine:              0,
		Language:             "",
		FileExtension:        "",
		ConversationID:       "",
		ParentConversationID: "",
		MessageIndex:         0,
		Role:                 "",
		TimestampUnix:        0,
		WorkspaceRoot:        "",
		Archived:             false,
		SplitPart:            0,
		SplitPartRecorded:    false,
		LoadRules:            "",
		Scalars:              scalars,
		Score:                0,
	}
}

// writeScalarBackfill writes the backfill columns of one page through a Milvus
// partial update. The update sends the primary key and the backfill columns,
// and Milvus keeps every other stored field of each row.
func (service *Service) writeScalarBackfill(ctx context.Context, collectionName string, columns []model.ScalarColumn, page scalarBackfillPage) error {
	declaredColumns, err := declaredScalarInsertColumns(collectionName, columns, page.rows)
	if err != nil {
		return err
	}
	option := milvusclient.NewColumnBasedInsertOption(collectionName).
		WithVarcharColumn(idFieldName, page.ids).
		WithColumns(declaredColumns...).
		WithPartialUpdate(true)
	_, err = service.milvus.Upsert(ctx, option)
	service.rankings.noteWrite(collectionName)
	if err != nil {
		slog.ErrorContext(ctx, "scalar backfill partial update failed", "collection", collectionName, "rows", len(page.ids), "err", err)
		return wrapStoreError(ctx, err, "backfill scalars in "+collectionName)
	}
	return nil
}
