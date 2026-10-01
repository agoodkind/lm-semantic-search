package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const (
	// maxDeclaredStringLength is the largest VarChar length Milvus accepts.
	maxDeclaredStringLength = 65_535
	// maxColumnIdentifierLength is the longest field name Milvus accepts.
	maxColumnIdentifierLength = 255
)

// CollectionRegistration is one request to register a document collection
// with a declared scalar schema.
type CollectionRegistration struct {
	CollectionID string
	Declaration  model.CollectionDeclaration
}

func (manager *Manager) RegisterCollection(ctx context.Context, registration CollectionRegistration) (model.Codebase, error) {
	collectionID := strings.TrimSpace(registration.CollectionID)
	if collectionID == "" {
		return model.Codebase{}, adapterr.NewMissingArgument("collection_id")
	}
	declaration := cloneCollectionDeclaration(registration.Declaration)
	if err := validateCollectionDeclaration(declaration); err != nil {
		return model.Codebase{}, err
	}
	if manager.semantic == nil {
		return model.Codebase{}, semantic.ErrUnavailable
	}

	// Registration locks policyMutationMutex until it returns. No other
	// registration creates or changes this collection's record while this one
	// reads the stored schema without manager.mu.
	manager.policyMutationMutex.Lock()
	defer manager.policyMutationMutex.Unlock()

	manager.mu.Lock()
	existing, found := manager.findDocumentCollectionLocked(collectionID)
	manager.mu.Unlock()

	collectionName := manager.semantic.DocumentCollectionName(collectionID)
	legacyRecord := false
	if found {
		collectionName = existing.CollectionName
		saved := model.CollectionDeclaration{}
		if existing.Declaration != nil {
			saved = cloneCollectionDeclaration(*existing.Declaration)
		} else {
			legacyRecord = true
		}
		if err := compareCollectionDeclarations(collectionID, saved, declaration); err != nil {
			return model.Codebase{}, err
		}
	}
	if collectionName == "" {
		return model.Codebase{}, errors.New("document collection name is unavailable")
	}
	if !manager.semantic.Available() {
		slog.WarnContext(ctx, "collection registration skipped stored schema comparison", "collection_id", collectionID, "collection", collectionName, "reason", "vector store unavailable", "legacy_record", legacyRecord)
		if legacyRecord {
			return existing, nil
		}
	} else if err := manager.validateStoredCollectionSchema(ctx, collectionID, collectionName, declaration); err != nil {
		return model.Codebase{}, err
	}

	manager.mu.Lock()
	defer manager.mu.Unlock()
	if found {
		return manager.saveCollectionDeclarationLocked(ctx, collectionID, existing.ID, declaration)
	}
	return manager.createDocumentCollectionLocked(ctx, collectionID, collectionName, declaration)
}

func (manager *Manager) validateStoredCollectionSchema(
	ctx context.Context,
	collectionID string,
	collectionName string,
	declaration model.CollectionDeclaration,
) error {
	columns, exists, err := manager.semantic.DescribeScalarColumns(ctx, collectionName)
	if err != nil {
		slog.ErrorContext(ctx, "describe collection schema for registration failed", "collection_id", collectionID, "collection", collectionName, "err", err)
		return fmt.Errorf("describe collection %s: %w", collectionName, err)
	}
	if !exists {
		return nil
	}

	return compareScalarColumns(collectionID, "stored collection schema", columns, declaration.Scalars)
}

// saveCollectionDeclarationLocked saves declaration on an existing record that
// has none. A record with a saved declaration returns unchanged, because the
// caller already compared that declaration. Caller must hold manager.mu.
func (manager *Manager) saveCollectionDeclarationLocked(
	ctx context.Context,
	collectionID string,
	codebaseID string,
	declaration model.CollectionDeclaration,
) (model.Codebase, error) {
	current, tracked := manager.codebases[codebaseID]
	if !tracked {
		return model.Codebase{}, fmt.Errorf("collection %s was removed during registration", collectionID)
	}
	if current.Declaration != nil {
		return current, nil
	}
	previous := current
	current.Declaration = &declaration
	current.UpdatedAt = clock.Now()
	manager.codebases[codebaseID] = current
	if err := manager.saveLocked(); err != nil {
		// Restore the record without a declaration when the registry write fails,
		// mirroring the create path. Later lookups then never read a declaration
		// the registry file does not contain.
		manager.codebases[codebaseID] = previous
		slog.ErrorContext(ctx, "persist collection declaration failed", "collection_id", collectionID, "err", err)
		return model.Codebase{}, fmt.Errorf("persist collection declaration %s: %w", collectionID, err)
	}
	// Every persisted codebase record write sends one observer signal. For a
	// document collection the signal deletes nothing.
	manager.observer.Invalidate(codebaseID)
	manager.semantic.RecordCollectionDeclaration(current.CollectionName, declaration)
	return current, nil
}

// createDocumentCollectionLocked creates and persists a new document
// collection record with declaration. Caller must hold manager.mu.
func (manager *Manager) createDocumentCollectionLocked(
	ctx context.Context,
	collectionID string,
	collectionName string,
	declaration model.CollectionDeclaration,
) (model.Codebase, error) {
	codebase := newCodebaseRecord(documentCanonicalPath(collectionID))
	codebase.Kind = model.CodebaseKindDocument
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = manager.enrichIndexConfig(emptyAutoIndexConfig())
	codebase.EffectiveConfig.IgnoreDigest = digestIndexConfig(codebase.EffectiveConfig)
	codebase.CollectionName = collectionName
	codebase.Declaration = &declaration
	codebase.UpdatedAt = clock.Now()
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		// Roll the in-memory record back when the registry write fails, mirroring
		// the adopt and worktree paths. Later lookups then never treat an unsaved
		// record as registered until restart.
		delete(manager.codebases, codebase.ID)
		slog.ErrorContext(ctx, "persist collection registration failed", "collection_id", collectionID, "err", err)
		return model.Codebase{}, fmt.Errorf("persist collection %s: %w", collectionID, err)
	}
	// Every persisted codebase record write sends one observer signal. For a
	// document collection the signal deletes nothing.
	manager.observer.Invalidate(codebase.ID)
	manager.semantic.RecordCollectionDeclaration(collectionName, declaration)
	return codebase, nil
}

// validateCollectionDeclaration rejects a declaration with a missing or
// undeclared item id column, a duplicate column, a built-in schema column, an
// invalid column identifier, or an unsupported column type or length.
func validateCollectionDeclaration(declaration model.CollectionDeclaration) error {
	if strings.TrimSpace(declaration.ItemIDColumn) == "" {
		return adapterr.NewMissingArgument("item_id_column")
	}
	builtinColumns := semantic.BuiltinColumnNames()
	declared := make(map[string]model.ScalarColumn, len(declaration.Scalars))
	for _, column := range declaration.Scalars {
		if err := validateScalarColumn(column, builtinColumns); err != nil {
			return err
		}
		if _, duplicate := declared[column.Name]; duplicate {
			return adapterr.NewInvalidColumnDeclaration(
				column.Name,
				fmt.Sprintf("scalar column %q is declared more than once", column.Name),
			)
		}
		declared[column.Name] = column
	}
	itemColumn, itemColumnDeclared := declared[declaration.ItemIDColumn]
	if !itemColumnDeclared {
		return adapterr.NewInvalidColumnDeclaration(
			declaration.ItemIDColumn,
			fmt.Sprintf("item_id_column %q is not a declared scalar column", declaration.ItemIDColumn),
		)
	}
	if itemColumn.Type != model.ScalarTypeString {
		return adapterr.NewInvalidColumnDeclaration(
			declaration.ItemIDColumn,
			fmt.Sprintf("item_id_column %q must be a string column, not %s", declaration.ItemIDColumn, itemColumn.Type),
		)
	}
	return nil
}

func validateScalarColumn(column model.ScalarColumn, builtinColumns []string) error {
	if !isColumnIdentifier(column.Name) {
		return adapterr.NewInvalidColumnDeclaration(
			column.Name,
			fmt.Sprintf(
				"scalar column %q is not a column identifier: start with a letter or underscore, continue with letters, digits, or underscores, and use at most %d characters",
				column.Name,
				maxColumnIdentifierLength,
			),
		)
	}
	if slices.Contains(builtinColumns, column.Name) {
		return adapterr.NewInvalidColumnDeclaration(
			column.Name,
			fmt.Sprintf("scalar column %q is a built-in schema column", column.Name),
		)
	}
	switch column.Type {
	case model.ScalarTypeString:
		if column.MaxLength < 1 || column.MaxLength > maxDeclaredStringLength {
			return adapterr.NewInvalidColumnDeclaration(
				column.Name,
				fmt.Sprintf("string column %q needs max_length from 1 to %d", column.Name, maxDeclaredStringLength),
			)
		}
	case model.ScalarTypeBool, model.ScalarTypeInt64:
		if column.MaxLength != 0 {
			return adapterr.NewInvalidColumnDeclaration(
				column.Name,
				fmt.Sprintf("%s column %q does not accept max_length", column.Type, column.Name),
			)
		}
	default:
		return adapterr.NewInvalidColumnDeclaration(
			column.Name,
			fmt.Sprintf("scalar column %q has unsupported type %q; declare string, bool, or int64", column.Name, column.Type),
		)
	}
	return nil
}

// isColumnIdentifier reports whether name is a letter or underscore followed by
// letters, digits, or underscores, within the Milvus field name length.
func isColumnIdentifier(name string) bool {
	if name == "" || len(name) > maxColumnIdentifierLength {
		return false
	}
	for index, character := range name {
		isLetter := (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z')
		isDigit := character >= '0' && character <= '9'
		if isLetter || character == '_' {
			continue
		}
		if isDigit && index > 0 {
			continue
		}
		return false
	}
	return true
}

// compareCollectionDeclarations reports the first difference between the saved
// declaration and a requested one as a schema mismatch.
func compareCollectionDeclarations(collectionID string, saved model.CollectionDeclaration, requested model.CollectionDeclaration) error {
	if saved.ItemIDColumn != requested.ItemIDColumn {
		return adapterr.NewCollectionSchemaMismatch(
			collectionID,
			requested.ItemIDColumn,
			fmt.Sprintf("the saved declaration uses item id column %q", saved.ItemIDColumn),
		)
	}
	return compareScalarColumns(collectionID, "saved declaration", saved.Scalars, requested.Scalars)
}

// compareScalarColumns reports the first difference between existing columns
// and requested columns as a schema mismatch. It checks requested columns in
// order, then reports the first existing column the request omits, sorted by
// name. The error message uses source as the label of the existing side.
func compareScalarColumns(collectionID string, source string, existing []model.ScalarColumn, requested []model.ScalarColumn) error {
	existingByName := make(map[string]model.ScalarColumn, len(existing))
	for _, column := range existing {
		existingByName[column.Name] = column
	}
	requestedNames := make(map[string]struct{}, len(requested))
	for _, column := range requested {
		requestedNames[column.Name] = struct{}{}
		stored, found := existingByName[column.Name]
		if !found {
			return adapterr.NewCollectionSchemaMismatch(
				collectionID,
				column.Name,
				fmt.Sprintf("the %s has no such column", source),
			)
		}
		if stored != column {
			return adapterr.NewCollectionSchemaMismatch(
				collectionID,
				column.Name,
				fmt.Sprintf("the %s declares %s, and the registration declares %s", source, describeScalarColumn(stored), describeScalarColumn(column)),
			)
		}
	}
	omitted := make([]string, 0)
	for _, column := range existing {
		if _, found := requestedNames[column.Name]; !found {
			omitted = append(omitted, column.Name)
		}
	}
	if len(omitted) > 0 {
		slices.Sort(omitted)
		return adapterr.NewCollectionSchemaMismatch(
			collectionID,
			omitted[0],
			fmt.Sprintf("the %s has this column, and the registration omits it", source),
		)
	}
	return nil
}

func describeScalarColumn(column model.ScalarColumn) string {
	nullability := "not nullable"
	if column.Nullable {
		nullability = "nullable"
	}
	if column.Type == model.ScalarTypeString {
		return fmt.Sprintf("%s(max_length=%d, %s)", column.Type, column.MaxLength, nullability)
	}
	return fmt.Sprintf("%s(%s)", column.Type, nullability)
}

func cloneCollectionDeclaration(declaration model.CollectionDeclaration) model.CollectionDeclaration {
	return model.CollectionDeclaration{
		ItemIDColumn: declaration.ItemIDColumn,
		Scalars:      slices.Clone(declaration.Scalars),
	}
}
