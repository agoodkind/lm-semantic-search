package daemon

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// maxCollectionRowKeyBytes bounds a client row key. The stored relativePath
// column accepts 1024 bytes, and the key leaves room for a split part suffix.
const maxCollectionRowKeyBytes = 1000

// maxCollectionContinuationPrefixBytes bounds a client row's continuation
// prefix. The split starts every part after the first with the prefix and a
// newline, and a prefix of at least the split budget less one would make each
// such part larger than the budget.
const maxCollectionContinuationPrefixBytes = 1024

// collectionScalarInput is one scalar value a client row sets. Value.Null
// marks a value the wire left unset.
type collectionScalarInput struct {
	Column string
	Value  collection.ScalarValue
}

// collectionRowInput is one client row before validation against the saved
// declaration.
type collectionRowInput struct {
	RowKey             string
	ItemID             string
	Text               string
	Scalars            []collectionScalarInput
	ContinuationPrefix string
}

// collectionItemsRequest is one generic item upsert. Manifest is nil when the
// client sent no manifest chunk.
type collectionItemsRequest struct {
	CollectionID string
	Client       model.ClientInfo
	Rows         []collectionRowInput
	Manifest     map[string]string
	Absence      absencePolicy
	Backfill     bool
	Force        bool
}

// SyncCollectionManifest diffs a registered document collection's item
// manifest against its stored checkpoint and returns the item ids the engine
// needs. It uses the same checkpoint, per-ingest cap, and rotation cursor as
// SyncConversationManifest. An unregistered collection id fails. The generic
// RPCs never register a collection implicitly.
func (manager *Manager) SyncCollectionManifest(ctx context.Context, collectionID string, manifest map[string]string) ([]string, error) {
	codebase, err := manager.registeredCollection(collectionID)
	if err != nil {
		return nil, err
	}
	return manager.syncCollectionManifest(ctx, codebase, manifest), nil
}

// upsertCollectionItems validates client rows against the collection's saved
// declaration and queues the ingest through queueCollectionUpsert. Without a
// manifest it fingerprints each delivered item from its rows, which a retain
// upsert allows and an authoritative upsert rejects.
func (manager *Manager) upsertCollectionItems(ctx context.Context, request collectionItemsRequest) (model.Job, error) {
	if request.Manifest == nil && request.Absence == absenceDeleteGuarded {
		return model.Job{}, adapterr.NewInvalidArgument("authoritative collection upsert requires an explicit manifest")
	}
	codebase, err := manager.registeredCollection(request.CollectionID)
	if err != nil {
		return model.Job{}, err
	}
	rows, err := validateCollectionRows(savedCollectionDeclaration(codebase), request.Rows)
	if err != nil {
		return model.Job{}, err
	}
	manifest := request.Manifest
	if manifest == nil {
		manifest = manifestFromRows(rows)
	}
	return manager.queueCollectionUpsert(ctx, codebase, request.Client, collectionUpsert{
		Manifest:  manifest,
		Documents: nil,
		Rows:      rows,
		Absence:   request.Absence,
		Backfill:  request.Backfill,
		Force:     request.Force,
	})
}

// registeredCollection returns the registry record of a document collection
// id, or a not-registered error.
func (manager *Manager) registeredCollection(collectionID string) (model.Codebase, error) {
	trimmedCollectionID := strings.TrimSpace(collectionID)
	if trimmedCollectionID == "" {
		return model.Codebase{}, adapterr.NewMissingArgument("collection_id")
	}
	manager.mu.Lock()
	codebase, found := manager.findConversationCollectionLocked(trimmedCollectionID)
	manager.mu.Unlock()
	if !found {
		return model.Codebase{}, adapterr.NewCollectionNotRegistered(trimmedCollectionID)
	}
	return codebase, nil
}

// savedCollectionDeclaration returns a document collection's saved
// declaration. Conversation registration created every record without one, and
// such a record uses the conversation declaration.
func savedCollectionDeclaration(codebase model.Codebase) collection.Declaration {
	if codebase.Declaration == nil {
		return semantic.ConversationDeclaration()
	}
	return cloneCollectionDeclaration(*codebase.Declaration)
}

// documentItemSource builds the ingest source of one document collection job
// from the collection's saved declaration. The conversation declaration reads
// stored rows through the conversation batch read and selects legacy rows by
// path prefix. Every other declaration reads and selects rows by its declared
// item id column.
func (manager *Manager) documentItemSource(codebaseID string, payload conversationJobPayload) collectionItemSource {
	manager.mu.Lock()
	codebase := manager.codebases[codebaseID]
	manager.mu.Unlock()
	declaration := savedCollectionDeclaration(codebase)
	var stored collectionStoredReader = declaredStoredReader{loader: manager.semantic, itemColumn: declaration.ItemIDColumn}
	if semantic.IsConversationDeclaration(declaration) {
		stored = conversationStoredReader{rowReader: manager.semantic}
	}
	return newDocumentItemSource(payload.CollectionName, declaration, stored, documentDelivery{
		manifest:        payload.Manifest,
		documents:       payload.Documents,
		rows:            payload.Rows,
		absence:         payload.Absence,
		backfill:        payload.Backfill,
		force:           payload.Force,
		chunkByteBudget: manager.conversationChunkByteBudget,
	})
}

// recordCollectionDeclarations records the saved declaration of every
// document collection on the vector store backend.
func (manager *Manager) recordCollectionDeclarations() {
	if manager.semantic == nil {
		return
	}
	manager.mu.Lock()
	defer manager.mu.Unlock()
	for _, codebase := range manager.codebases {
		if codebase.Kind != model.CodebaseKindDocument || codebase.CollectionName == "" {
			continue
		}
		manager.semantic.RecordCollectionDeclaration(codebase.CollectionName, savedCollectionDeclaration(codebase))
	}
}

// validateCollectionRows checks every client row against declaration and
// returns the validated rows. It rejects a duplicate row key, an empty or
// oversized row key or item id, an oversized continuation prefix, an
// undeclared, duplicated, mistyped, oversized, or missing column value, and an
// item id column value that differs from the row's item_id.
func validateCollectionRows(declaration collection.Declaration, inputs []collectionRowInput) ([]collectionRow, error) {
	columns := make(map[string]collection.ScalarColumn, len(declaration.Scalars))
	for _, column := range declaration.Scalars {
		columns[column.Name] = column
	}
	conversation := semantic.IsConversationDeclaration(declaration)
	rowKeys := make(map[string]struct{}, len(inputs))
	rows := make([]collectionRow, 0, len(inputs))
	for _, input := range inputs {
		row, err := validateCollectionRow(declaration, columns, input)
		if err != nil {
			return nil, err
		}
		if _, duplicate := rowKeys[row.RowKey]; duplicate {
			return nil, adapterr.NewInvalidArgument(fmt.Sprintf("row_key %q appears more than once", row.RowKey))
		}
		rowKeys[row.RowKey] = struct{}{}
		if conversation {
			if err := validateConversationRowScalars(row); err != nil {
				return nil, err
			}
		}
		rows = append(rows, row)
	}
	return rows, nil
}

func validateCollectionRow(declaration collection.Declaration, columns map[string]collection.ScalarColumn, input collectionRowInput) (collectionRow, error) {
	if strings.TrimSpace(input.RowKey) == "" {
		return collectionRow{}, adapterr.NewMissingArgument("row_key")
	}
	if len(input.RowKey) > maxCollectionRowKeyBytes || !utf8.ValidString(input.RowKey) {
		return collectionRow{}, adapterr.NewInvalidArgument(fmt.Sprintf("row_key must be valid UTF-8 of at most %d bytes", maxCollectionRowKeyBytes))
	}
	itemID := strings.TrimSpace(input.ItemID)
	if itemID == "" {
		return collectionRow{}, adapterr.NewMissingArgument("item_id")
	}
	itemColumn := columns[declaration.ItemIDColumn]
	if len(itemID) > int(itemColumn.MaxLength) || !utf8.ValidString(itemID) {
		return collectionRow{}, adapterr.NewInvalidColumnValue(itemColumn.Name, fmt.Sprintf("row %q item_id must be valid UTF-8 of at most %d bytes", input.RowKey, itemColumn.MaxLength))
	}
	if len(input.ContinuationPrefix) > maxCollectionContinuationPrefixBytes {
		return collectionRow{}, adapterr.NewInvalidArgument(fmt.Sprintf("row %q continuation_prefix must be at most %d bytes", input.RowKey, maxCollectionContinuationPrefixBytes))
	}
	scalars := make(map[string]collection.ScalarValue, len(columns))
	for _, scalar := range input.Scalars {
		if _, duplicate := scalars[scalar.Column]; duplicate {
			return collectionRow{}, adapterr.NewInvalidColumnValue(scalar.Column, fmt.Sprintf("row %q sets column %q more than once", input.RowKey, scalar.Column))
		}
		if err := validateCollectionScalar(fmt.Sprintf("row %q", input.RowKey), columns, scalar); err != nil {
			return collectionRow{}, err
		}
		if scalar.Column == itemColumn.Name && (scalar.Value.Null || scalar.Value.String != itemID) {
			return collectionRow{}, adapterr.NewInvalidColumnValue(scalar.Column, fmt.Sprintf("row %q sets item id column %q to a value other than its item_id", input.RowKey, scalar.Column))
		}
		scalars[scalar.Column] = scalar.Value
	}
	scalars[itemColumn.Name] = collection.ScalarValue{Type: collection.ScalarTypeString, Null: false, String: itemID, Bool: false, Int64: 0}
	for _, column := range declaration.Scalars {
		value, present := scalars[column.Name]
		if !column.Nullable && (!present || value.Null) {
			return collectionRow{}, adapterr.NewInvalidColumnValue(column.Name, fmt.Sprintf("row %q has no value for column %q, which is not nullable", input.RowKey, column.Name))
		}
	}
	return collectionRow{RowKey: input.RowKey, ItemID: itemID, Text: input.Text, Scalars: scalars, ContinuationPrefix: input.ContinuationPrefix}, nil
}

// validateCollectionScalar checks one scalar value against the declared
// columns. subject identifies the row or item that sets the value in an error
// message, for example `row "doc-a/title"`.
func validateCollectionScalar(subject string, columns map[string]collection.ScalarColumn, scalar collectionScalarInput) error {
	column, declared := columns[scalar.Column]
	if !declared {
		return adapterr.NewInvalidColumnValue(scalar.Column, fmt.Sprintf("%s sets undeclared column %q", subject, scalar.Column))
	}
	if scalar.Value.Null {
		if !column.Nullable {
			return adapterr.NewInvalidColumnValue(column.Name, fmt.Sprintf("%s sets column %q to null, which is not nullable", subject, column.Name))
		}
		return nil
	}
	if scalar.Value.Type != column.Type {
		return adapterr.NewInvalidColumnValue(column.Name, fmt.Sprintf("%s sets %s column %q to a %s value", subject, column.Type, column.Name, scalar.Value.Type))
	}
	if column.Type == collection.ScalarTypeString && (len(scalar.Value.String) > int(column.MaxLength) || !utf8.ValidString(scalar.Value.String)) {
		return adapterr.NewInvalidColumnValue(column.Name, fmt.Sprintf("%s column %q must be valid UTF-8 of at most %d bytes", subject, column.Name, column.MaxLength))
	}
	return nil
}

// validateConversationRowScalars checks a row of a collection with the
// conversation declaration. The conversation stored-row read and the family
// delta require the conversation row key layout: conv/<id>/<message>,
// convtool/<id>/<message>/<tool>, or convthink/<id>/<message>, where <id> is
// the row's item_id and <message> equals the messageIndex value. A
// conversation collection stores the provider derived from the conversation id
// and stores messageIndex as a 32-bit integer.
func validateConversationRowScalars(row collectionRow) error {
	provider, providerPresent := row.Scalars[semantic.ConversationProviderColumn]
	storedProvider := semantic.ProviderFromConversationID(row.ItemID)
	if providerPresent && !provider.Null && provider.String != storedProvider {
		return adapterr.NewInvalidColumnValue(semantic.ConversationProviderColumn, fmt.Sprintf("row %q sets provider %q, and its item_id stores provider %q", row.RowKey, provider.String, storedProvider))
	}
	messageIndex := row.Scalars[semantic.ConversationMessageIndexColumn].Int64
	if messageIndex < math.MinInt32 || messageIndex > math.MaxInt32 {
		return adapterr.NewInvalidColumnValue(semantic.ConversationMessageIndexColumn, fmt.Sprintf("row %q sets messageIndex outside the 32-bit range", row.RowKey))
	}
	if !conversationRowKeyMatches(row.RowKey, row.ItemID, messageIndex) {
		return adapterr.NewInvalidArgument(fmt.Sprintf(
			"row_key %q must be conv/<item_id>/<messageIndex>, convtool/<item_id>/<messageIndex>/<tool>, or convthink/<item_id>/<messageIndex> in a collection with the conversation declaration",
			row.RowKey,
		))
	}
	return nil
}

// conversationRowKeyMatches reports whether rowKey is a message text, tool
// call, or thinking path of conversationID at messageIndex.
func conversationRowKeyMatches(rowKey string, conversationID string, messageIndex int64) bool {
	families := []struct {
		prefix   string
		segments int
	}{
		{prefix: conversationRelativePathPrefix(conversationID), segments: 1},
		{prefix: conversationToolRelativePathPrefix(conversationID), segments: 2},
		{prefix: conversationThinkingRelativePathPrefix(conversationID), segments: 1},
	}
	for _, family := range families {
		remainder, found := strings.CutPrefix(rowKey, family.prefix)
		if !found {
			continue
		}
		parts := strings.Split(remainder, "/")
		if len(parts) != family.segments {
			return false
		}
		for _, part := range parts {
			if _, err := strconv.ParseUint(part, 10, 31); err != nil {
				return false
			}
		}
		return parts[0] == strconv.FormatInt(messageIndex, 10)
	}
	return false
}

// manifestFromRows fingerprints each delivered item from its rows: every row
// key, text, continuation prefix, and scalar value in row key and column order.
func manifestFromRows(rows []collectionRow) map[string]string {
	byItem := make(map[string][]collectionRow)
	for _, row := range rows {
		byItem[row.ItemID] = append(byItem[row.ItemID], row)
	}
	manifest := make(map[string]string, len(byItem))
	for itemID, itemRows := range byItem {
		manifest[itemID] = fingerprintCollectionRows(itemRows)
	}
	return manifest
}

func fingerprintCollectionRows(rows []collectionRow) string {
	sorted := append([]collectionRow(nil), rows...)
	sort.Slice(sorted, func(first int, second int) bool {
		return sorted[first].RowKey < sorted[second].RowKey
	})
	hasher := sha256.New()
	for _, row := range sorted {
		hasher.Write([]byte(row.RowKey))
		hasher.Write([]byte{0})
		hasher.Write([]byte(row.Text))
		hasher.Write([]byte{0})
		// An empty prefix adds nothing, and a row without a prefix keeps the
		// fingerprint it had before the prefix field existed. The prefix section
		// starts with its decimal byte length and a separator. The encoding
		// assumes that declared column names do not start with a digit.
		if row.ContinuationPrefix != "" {
			hasher.Write([]byte(strconv.Itoa(len(row.ContinuationPrefix))))
			hasher.Write([]byte{0})
			hasher.Write([]byte(row.ContinuationPrefix))
			hasher.Write([]byte{0})
		}
		columns := make([]string, 0, len(row.Scalars))
		for column := range row.Scalars {
			columns = append(columns, column)
		}
		sort.Strings(columns)
		for _, column := range columns {
			value := row.Scalars[column]
			fields := []string{
				column,
				string(value.Type),
				strconv.FormatBool(value.Null),
				value.String,
				strconv.FormatBool(value.Bool),
				strconv.FormatInt(value.Int64, 10),
			}
			for _, field := range fields {
				hasher.Write([]byte(field))
				hasher.Write([]byte{0})
			}
		}
	}
	return hex.EncodeToString(hasher.Sum(nil))
}
