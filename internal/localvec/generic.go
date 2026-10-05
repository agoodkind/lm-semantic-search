package localvec

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	lmcollection "goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const invalidUTF8Replacement = "�"

// GenericStore implements [lmcollection.Store] on the local vector files. The
// caller supplies every vector. A write of new IDs appends to the row file. A
// write that replaces a stored ID rewrites the row file. Every write replaces
// the vector index file.
type GenericStore struct {
	store          *Store
	embeddingModel string
}

var _ lmcollection.Store = (*GenericStore)(nil)

// OpenGeneric writes embeddingModel to each row that the returned store writes.
func OpenGeneric(root string, embeddingModel string) (*GenericStore, error) {
	if strings.TrimSpace(root) == "" {
		return nil, errors.New("local vector store root is required")
	}
	store, err := newStoreAtRoot(config.Config{}, root, nil)
	if err != nil {
		return nil, err
	}
	return &GenericStore{store: store, embeddingModel: embeddingModel}, nil
}

// Close releases the loaded vector indexes.
func (generic *GenericStore) Close() {
	generic.store.mutex.Lock()
	defer generic.store.mutex.Unlock()
	for key, stored := range generic.store.collections {
		stored.mutex.Lock()
		stored.discardLoadedLocked()
		stored.mutex.Unlock()
		delete(generic.store.collections, key)
	}
}

func (generic *GenericStore) collection(collectionName string) (*collection, string, error) {
	name := strings.TrimSpace(collectionName)
	if name == "" {
		return nil, "", errors.New("collection name is required")
	}
	stored, err := generic.store.collectionForName(name, false)
	return stored, name, err
}

// EnsureCollection returns an error for a collection with rows of another
// vector width.
func (generic *GenericStore) EnsureCollection(ctx context.Context, request lmcollection.EnsureRequest) error {
	if err := operationContextError(ctx, "ensure local collection"); err != nil {
		return err
	}
	stored, name, err := generic.collection(request.Collection)
	if err != nil {
		return err
	}
	if err := stored.ensure(request.Dimension); err != nil {
		slog.ErrorContext(ctx, "ensure local collection failed", "collection", name, "err", err)
		return fmt.Errorf("ensure local collection %s: %w", name, err)
	}
	return nil
}

// Upsert writes rows to a collection. A row replaces the stored row with the
// same ID. The store normalizes each vector to unit length. Upsert validates
// every row before it writes.
func (generic *GenericStore) Upsert(ctx context.Context, collectionName string, declaration lmcollection.Declaration, rows []lmcollection.Row) error {
	if len(rows) == 0 {
		return nil
	}
	if err := operationContextError(ctx, "upsert local collection rows"); err != nil {
		return err
	}
	stored, name, err := generic.collection(collectionName)
	if err != nil {
		return err
	}
	prepared := make([]row, 0, len(rows))
	seenIDs := make(map[string]struct{}, len(rows))
	for _, source := range rows {
		if _, duplicate := seenIDs[source.ID]; duplicate {
			err := fmt.Errorf("upsert into %s: row ID %s appears more than once", name, source.ID)
			slog.ErrorContext(ctx, "prepare local collection row failed", "collection", name, "relative_path", source.RelativePath, "err", err)
			return err
		}
		seenIDs[source.ID] = struct{}{}
		next, prepareErr := generic.genericRow(declaration.Scalars, source)
		if prepareErr != nil {
			slog.ErrorContext(ctx, "prepare local collection row failed", "collection", name, "relative_path", source.RelativePath, "err", prepareErr)
			return fmt.Errorf("upsert into %s: %w", name, prepareErr)
		}
		prepared = append(prepared, next)
	}
	return missingAsLibraryError(stored.upsert(prepared))
}

func (generic *GenericStore) genericRow(declared []lmcollection.ScalarColumn, source lmcollection.Row) (row, error) {
	if source.ID == "" {
		return row{}, fmt.Errorf("row %s does not have an ID", source.RelativePath)
	}
	normalized, err := normalizeVector(source.Vector)
	if err != nil {
		return row{}, err
	}
	scalars := make(map[string]lmcollection.ScalarValue, len(declared))
	for _, column := range declared {
		value, present := source.Scalars[column.Name]
		if !present || value.Null {
			if !column.Nullable {
				return row{}, fmt.Errorf("row %s does not have a value for column %s, and the column is not nullable", source.RelativePath, column.Name)
			}
			scalars[column.Name] = lmcollection.ScalarValue{Type: column.Type, Null: true, String: "", Bool: false, Int64: 0}
			continue
		}
		if value.Type != column.Type {
			return row{}, fmt.Errorf("row %s has a %s value for %s column %s", source.RelativePath, value.Type, column.Type, column.Name)
		}
		value.String = strings.ToValidUTF8(value.String, invalidUTF8Replacement)
		scalars[column.Name] = value
	}
	content := strings.ToValidUTF8(source.Content, invalidUTF8Replacement)
	return row{
		Label:             0,
		ID:                source.ID,
		RelativePath:      strings.ToValidUTF8(source.RelativePath, invalidUTF8Replacement),
		StartLine:         source.StartLine,
		EndLine:           source.EndLine,
		Language:          "",
		FileExtension:     strings.ToValidUTF8(source.FileExtension, invalidUTF8Replacement),
		Content:           content,
		ContentVectorKey:  semantic.ContentVectorKey(content),
		Vector:            normalized,
		SplitPart:         source.SplitPart,
		SplitPartRecorded: source.SplitPartRecorded,
		Scalars:           scalars,
		Metadata:          strings.ToValidUTF8(source.Metadata, invalidUTF8Replacement),
		EmbeddingModel:    generic.embeddingModel,
	}, nil
}

// Search ranks the rows the filter matches by cosine similarity with
// request.Vector. It scores every matching row when at most
// lmcollection.RankingDepth rows match, and it reads the nearest
// lmcollection.RankingDepth rows from the vector index otherwise. Search
// ignores request.Query.
func (generic *GenericStore) Search(ctx context.Context, request lmcollection.SearchRequest) ([]lmcollection.Hit, error) {
	if err := operationContextError(ctx, "search local collection"); err != nil {
		return nil, err
	}
	stored, name, err := generic.collection(request.Collection)
	if err != nil {
		return nil, err
	}
	if _, compileErr := lmcollection.Compile(request.Filter); compileErr != nil {
		slog.ErrorContext(ctx, "validate collection filter failed", "collection", name, "err", compileErr)
		return nil, fmt.Errorf("validate filter for %s: %w", name, compileErr)
	}
	query, err := normalizeVector(request.Vector)
	if err != nil {
		slog.ErrorContext(ctx, "normalize local collection query failed", "collection", name, "err", err)
		return nil, fmt.Errorf("search %s: query %w", name, err)
	}
	width, exists, err := stored.width()
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, lmcollection.ErrCollectionMissing
	}
	if len(query) != width {
		return nil, fmt.Errorf("search %s: query vector has width %d, want %d", name, len(query), width)
	}
	declared := request.Declaration.Scalars
	matchesFilter := func(candidate row) bool {
		return request.Filter == nil || evaluateFilter(*request.Filter, candidate, declared) == truthTrue
	}
	candidates, err := stored.rankCandidates(ctx, query, matchesFilter, lmcollection.RankingDepth)
	if err != nil {
		return nil, missingAsLibraryError(err)
	}
	sortScoredRows(candidates)
	groupLimit := int32(0)
	if request.GroupBy != "" && request.PerGroupLimit > 0 {
		groupLimit = request.PerGroupLimit
	}
	perGroup := make(map[string]int32)
	selected := limitScoredRows(candidates, effectiveLimit(request.Limit), func(candidate scoredRow) bool {
		if request.MinScore > 0 && candidate.score < request.MinScore {
			return false
		}
		if !matchesFilter(candidate.stored) {
			return false
		}
		if groupLimit <= 0 {
			return true
		}
		key := storedCell(candidate.stored, request.GroupBy).GroupKey()
		if perGroup[key] >= groupLimit {
			return false
		}
		perGroup[key]++
		return true
	})
	hits := make([]lmcollection.Hit, 0, len(selected))
	for _, candidate := range selected {
		hits = append(hits, genericHit(candidate.stored, candidate.score, declared))
	}
	return hits, nil
}

// Query returns the rows a filter matches in ascending ID order. A request
// without a filter must specify a positive Limit.
func (generic *GenericStore) Query(ctx context.Context, request lmcollection.QueryRequest) ([]lmcollection.Hit, error) {
	if err := operationContextError(ctx, "query local collection"); err != nil {
		return nil, err
	}
	stored, name, err := generic.collection(request.Collection)
	if err != nil {
		return nil, err
	}
	if _, compileErr := lmcollection.Compile(request.Filter); compileErr != nil {
		slog.ErrorContext(ctx, "validate collection filter failed", "collection", name, "err", compileErr)
		return nil, fmt.Errorf("validate filter for %s: %w", name, compileErr)
	}
	if request.Filter == nil && request.Limit <= 0 {
		return nil, fmt.Errorf("query %s: a query without a filter needs a positive limit", name)
	}
	declared := request.Declaration.Scalars
	rows, err := stored.selectRows(func(candidate row) bool {
		return request.Filter == nil || evaluateFilter(*request.Filter, candidate, declared) == truthTrue
	}, request.Limit, false)
	if err != nil {
		return nil, missingAsLibraryError(err)
	}
	hits := make([]lmcollection.Hit, 0, len(rows))
	for _, selected := range rows {
		hits = append(hits, genericHit(selected, 0, declared))
	}
	return hits, nil
}

// Delete removes every row for which the complete filter evaluates to true.
// It reads each column with the type the row stores.
func (generic *GenericStore) Delete(ctx context.Context, collectionName string, filter lmcollection.Filter) (int64, error) {
	if err := operationContextError(ctx, "delete local collection rows"); err != nil {
		return 0, err
	}
	stored, name, err := generic.collection(collectionName)
	if err != nil {
		return 0, err
	}
	if _, compileErr := lmcollection.CompileInline(&filter); compileErr != nil {
		slog.ErrorContext(ctx, "validate collection filter failed", "collection", name, "err", compileErr)
		return 0, fmt.Errorf("validate filter for %s: %w", name, compileErr)
	}
	removed, err := stored.deleteWhere(func(candidate row) bool {
		return evaluateFilterCells(filter, func(columnName string) lmcollection.ScalarCell {
			return storedCell(candidate, columnName)
		}) == truthTrue
	})
	return removed, missingAsLibraryError(err)
}

// QueryRows returns rows in ascending ID order. Each returned vector has unit
// length.
func (generic *GenericStore) QueryRows(ctx context.Context, request lmcollection.RowsRequest) ([]lmcollection.StoredRow, error) {
	if err := operationContextError(ctx, "query local collection item rows"); err != nil {
		return nil, err
	}
	stored, name, err := generic.collection(request.Collection)
	if err != nil {
		return nil, err
	}
	if len(request.ItemIDs) == 0 && len(request.PathPrefixes) == 0 {
		return nil, fmt.Errorf("query rows of %s: request does not select an item or a path prefix", name)
	}
	if len(request.ItemIDs) > 0 && request.Declaration.ItemIDColumn == "" {
		return nil, fmt.Errorf("query rows of %s: declaration does not have an item ID column", name)
	}
	selects := itemSelector(request.Declaration.ItemIDColumn, request.ItemIDs, request.PathPrefixes)
	rows, err := stored.selectRows(selects, 0, request.IncludeVector)
	if err != nil {
		return nil, missingAsLibraryError(err)
	}
	declared := request.Declaration.Scalars
	storedRows := make([]lmcollection.StoredRow, 0, len(rows))
	for _, selected := range rows {
		contentSum := sha256.Sum256([]byte(selected.Content))
		storedRows = append(storedRows, lmcollection.StoredRow{
			ID:                selected.ID,
			RelativePath:      selected.RelativePath,
			Content:           selected.Content,
			SplitPart:         selected.SplitPart,
			SplitPartRecorded: selected.SplitPartRecorded,
			EmbeddingModel:    selected.EmbeddingModel,
			ContentHash:       hex.EncodeToString(contentSum[:]),
			Vector:            selected.Vector,
			Scalars:           genericCells(selected, declared),
		})
	}
	return storedRows, nil
}

// DeleteItems removes every row with an item ID column value in ItemIDs and
// every row with a relativePath that starts with one of PathPrefixes.
// DeleteItems skips an empty prefix.
func (generic *GenericStore) DeleteItems(ctx context.Context, request lmcollection.DeleteItemsRequest) (int64, error) {
	if err := operationContextError(ctx, "delete local collection items"); err != nil {
		return 0, err
	}
	stored, name, err := generic.collection(request.Collection)
	if err != nil {
		return 0, err
	}
	if len(request.ItemIDs) == 0 && len(request.PathPrefixes) == 0 {
		return 0, fmt.Errorf("delete items from %s: request does not select an item or a path prefix", name)
	}
	if len(request.ItemIDs) > 0 && request.Declaration.ItemIDColumn == "" {
		return 0, fmt.Errorf("delete items from %s: declaration does not have an item ID column", name)
	}
	prefixes := slices.DeleteFunc(slices.Clone(request.PathPrefixes), func(prefix string) bool {
		return prefix == ""
	})
	if len(request.ItemIDs) == 0 && len(prefixes) == 0 {
		return 0, nil
	}
	removed, err := stored.deleteWhere(itemSelector(request.Declaration.ItemIDColumn, request.ItemIDs, prefixes))
	return removed, missingAsLibraryError(err)
}

// BackfillScalars returns [lmcollection.ErrCollectionMissing] for an absent
// collection.
func (generic *GenericStore) BackfillScalars(ctx context.Context, collectionName string, backfill lmcollection.ScalarBackfill) (int, int, error) {
	name := strings.TrimSpace(collectionName)
	if name == "" {
		return 0, 0, errors.New("collection name is required")
	}
	if len(backfill.Columns) == 0 {
		return 0, 0, errors.New("scalar backfill does not list a column")
	}
	changed, orphan, err := generic.store.BackfillCollectionScalars(ctx, name, backfill)
	return changed, orphan, missingAsLibraryError(err)
}

func missingAsLibraryError(err error) error {
	if errors.Is(err, semantic.ErrCollectionMissing) {
		return lmcollection.ErrCollectionMissing
	}
	return err
}

func itemSelector(itemColumn string, itemIDs []string, pathPrefixes []string) func(row) bool {
	selected := make(map[string]struct{}, len(itemIDs))
	for _, itemID := range itemIDs {
		selected[itemID] = struct{}{}
	}
	return func(candidate row) bool {
		if itemID, present := candidate.itemID(itemColumn); present {
			if _, found := selected[itemID]; found {
				return true
			}
		}
		for _, prefix := range pathPrefixes {
			if strings.HasPrefix(candidate.RelativePath, prefix) {
				return true
			}
		}
		return false
	}
}

func storedCell(stored row, columnName string) lmcollection.ScalarCell {
	value, stores := stored.Scalars[columnName]
	if !stores {
		return lmcollection.AbsentCell(columnName)
	}
	if value.Null {
		return lmcollection.NullCell(columnName)
	}
	return lmcollection.ValueCell(columnName, value)
}

func genericCells(stored row, declared []lmcollection.ScalarColumn) map[string]lmcollection.ScalarCell {
	cells := make(map[string]lmcollection.ScalarCell, len(declared))
	for _, column := range declared {
		cells[column.Name] = rowScalarCell(stored, column.Name, declared)
	}
	return cells
}

func genericHit(stored row, score float64, declared []lmcollection.ScalarColumn) lmcollection.Hit {
	return lmcollection.Hit{
		ID:                stored.ID,
		Content:           stored.Content,
		Score:             score,
		RelativePath:      stored.RelativePath,
		StartLine:         stored.StartLine,
		EndLine:           stored.EndLine,
		FileExtension:     stored.FileExtension,
		Metadata:          stored.Metadata,
		SplitPart:         stored.SplitPart,
		SplitPartRecorded: stored.SplitPartRecorded,
		Scalars:           genericCells(stored, declared),
	}
}

func (stored *collection) ensure(dimensions int) error {
	stored.mutex.Lock()
	defer stored.mutex.Unlock()
	if err := stored.loadLocked(); err != nil {
		return err
	}
	if stored.exists && len(stored.rows) > 0 {
		if dimensions > 0 && stored.dimensions != dimensions {
			return fmt.Errorf("collection stores vectors of width %d, and the request declares width %d", stored.dimensions, dimensions)
		}
		return nil
	}
	if stored.exists && (dimensions <= 0 || stored.dimensions == dimensions) {
		return nil
	}
	if dimensions <= 0 {
		return fmt.Errorf("dimension %d is not positive", dimensions)
	}
	stored.dimensions = dimensions
	return stored.persistLocked(nil)
}

func (stored *collection) width() (int, bool, error) {
	stored.mutex.Lock()
	defer stored.mutex.Unlock()
	if err := stored.loadLocked(); err != nil {
		return 0, false, err
	}
	return stored.dimensions, stored.exists, nil
}

func (stored *collection) upsert(added []row) error {
	stored.mutex.Lock()
	defer stored.mutex.Unlock()
	if err := stored.loadLocked(); err != nil {
		return err
	}
	if !stored.exists {
		return semantic.ErrCollectionMissing
	}
	addedIDs := make(map[string]struct{}, len(added))
	for _, candidate := range added {
		if len(candidate.Vector) != stored.dimensions {
			return fmt.Errorf("row %s has a vector of width %d, want %d", candidate.RelativePath, len(candidate.Vector), stored.dimensions)
		}
		addedIDs[candidate.ID] = struct{}{}
	}
	replaces := false
	for _, existing := range stored.rows {
		if _, found := addedIDs[existing.ID]; found {
			replaces = true
			break
		}
	}
	if !replaces {
		return stored.appendLocked(added)
	}
	rewritten := make([]row, 0, len(stored.rows)+len(added))
	for _, existing := range stored.rows {
		if _, replaced := addedIDs[existing.ID]; !replaced {
			rewritten = append(rewritten, existing)
		}
	}
	rewritten = append(rewritten, added...)
	return stored.persistLocked(rewritten)
}

func (stored *collection) selectRows(keep func(row) bool, limit int, withVectors bool) ([]row, error) {
	stored.mutex.Lock()
	defer stored.mutex.Unlock()
	if err := stored.loadLocked(); err != nil {
		return nil, err
	}
	if !stored.exists {
		return nil, semantic.ErrCollectionMissing
	}
	selected := make([]row, 0)
	for _, candidate := range stored.rows {
		if !keep(candidate) {
			continue
		}
		if withVectors {
			candidate.Vector = slices.Clone(candidate.Vector)
		} else {
			candidate.Vector = nil
		}
		selected = append(selected, candidate)
	}
	slices.SortFunc(selected, func(left row, right row) int {
		return strings.Compare(left.ID, right.ID)
	})
	if limit > 0 && len(selected) > limit {
		selected = selected[:limit]
	}
	return selected, nil
}

func (stored *collection) deleteWhere(remove func(row) bool) (int64, error) {
	stored.mutex.Lock()
	defer stored.mutex.Unlock()
	if err := stored.loadLocked(); err != nil {
		return 0, err
	}
	if !stored.exists {
		return 0, semantic.ErrCollectionMissing
	}
	kept := make([]row, 0, len(stored.rows))
	for _, candidate := range stored.rows {
		if !remove(candidate) {
			kept = append(kept, candidate)
		}
	}
	removed := int64(len(stored.rows) - len(kept))
	if removed == 0 {
		return 0, nil
	}
	return removed, stored.persistLocked(kept)
}
