package semantic

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"google.golang.org/grpc/peer"
)

// Conversation collections carry their filterable attributes as native scalar
// columns so Milvus can pre-filter a search by them, rather than the engine
// over-fetching and post-filtering the JSON metadata column. These columns
// exist only on conversation collections. Conversation collections and generic
// document collections share the conv_chunks_ name prefix, and only this
// daemon writes them. The TS-adapter-owned code collections never declare
// these columns, and a generic document collection declares its own scalar
// columns instead. The values are still mirrored into the metadata JSON for
// backward compatibility with rows written before the columns existed.
const (
	conversationCollectionPrefix   = "conv_chunks_"
	conversationIDFieldName        = "conversationId"
	parentConversationIDFieldName  = "parentConversationId"
	roleFieldName                  = "role"
	timestampUnixFieldName         = "timestampUnix"
	messageIndexFieldName          = "messageIndex"
	providerFieldName              = "provider"
	workspaceRootFieldName         = "workspaceRoot"
	archivedFieldName              = "archived"
	loadRulesFieldName             = "loadRules"
	conversationIDFieldMaxLength   = 256
	conversationRoleFieldMaxLength = 64
	conversationProviderMaxLength  = 32
	conversationWorkspaceMaxLength = 1024
	conversationLoadRulesMaxLength = 256
	idFieldMaxLength               = 512
	contentFieldMaxLength          = 65_535
	embeddingModelFieldMaxLength   = 65535
)

// hasConversationCollectionPrefix reports whether a collection name uses the
// document collection prefix, including its staging twin. Conversation
// collections and generic document collections share this prefix. A Service
// decides whether a name stores conversation rows in
// [Service.isConversationCollection].
func hasConversationCollectionPrefix(collectionName string) bool {
	return strings.HasPrefix(collectionName, conversationCollectionPrefix)
}

// isConversationCollection reports whether a collection stores conversation
// rows with the conversation scalar columns. A name with the document
// collection prefix stores conversation rows unless the manager recorded a
// generic declaration for it through [Service.RecordCollectionDeclaration].
// The name-based conversation schema migration, the conversation backfills,
// and the search output column choice use this check. None of them adds
// conversation columns to a generic collection.
func (service *Service) isConversationCollection(collectionName string) bool {
	if !hasConversationCollectionPrefix(collectionName) {
		return false
	}
	_, declared := service.declaredCollections.Load(liveCollectionName(collectionName))
	return !declared
}

// liveCollectionName strips the staging and promotion recovery suffixes. A
// staging or recovery twin then resolves to the live collection it replaces.
func liveCollectionName(collectionName string) string {
	trimmed := strings.TrimSuffix(collectionName, stagingCollectionSuffix)
	return strings.TrimSuffix(trimmed, recoveryCollectionSuffix)
}

// RecordCollectionDeclaration records the saved declaration of a document
// collection. The conversation declaration clears any recorded generic
// declaration. The conversation schema migrations then apply to the
// collection. Any other declaration marks the collection as generic. The
// conversation migrations and backfills then skip it.
func (service *Service) RecordCollectionDeclaration(collectionName string, declaration collection.Declaration) {
	name := liveCollectionName(collectionName)
	if IsConversationDeclaration(declaration) {
		service.declaredCollections.Delete(name)
		return
	}
	service.declaredCollections.Store(name, struct{}{})
}

// IsConversationDeclaration reports whether declaration has the item id column
// and the scalar columns of [ConversationDeclaration] in any order: the same
// column names, types, nullability, and string lengths. Only that declaration
// stores rows in the conversation schema.
// Registration compares declarations without regard to column order.
// A saved declaration preserves the column order supplied during the
// registration that saved it.
func IsConversationDeclaration(declaration collection.Declaration) bool {
	conversation := ConversationDeclaration()
	return declaration.ItemIDColumn == conversation.ItemIDColumn &&
		slices.Equal(scalarColumnsByName(declaration.Scalars), scalarColumnsByName(conversation.Scalars))
}

// scalarColumnsByName returns a copy of columns sorted by column name.
func scalarColumnsByName(columns []collection.ScalarColumn) []collection.ScalarColumn {
	sorted := slices.Clone(columns)
	slices.SortFunc(sorted, func(left collection.ScalarColumn, right collection.ScalarColumn) int {
		return strings.Compare(left.Name, right.Name)
	})
	return sorted
}

type storeColumnKind int

const (
	storeColumnKindCode storeColumnKind = iota
	storeColumnKindConversation
	storeColumnKindDeclared
)

// StoreColumnSet is the set of scalar columns a store write populates and a
// created collection declares. The ingest caller (the item source) passes it
// into the write path. insertBatch never derives the row shape from the
// collection name. A code write sends only the base columns. A conversation
// write also sends the conversation scalar columns from the conversation
// fields of each chunk. A declared write also sends each declared column from
// [model.StoredChunk.Scalars].
type StoreColumnSet struct {
	kind    storeColumnKind
	scalars []collection.ScalarColumn
}

// CodeColumns returns the column set of a code collection.
func CodeColumns() StoreColumnSet {
	return StoreColumnSet{kind: storeColumnKindCode, scalars: nil}
}

// ConversationColumns returns the column set of a conversation collection.
func ConversationColumns() StoreColumnSet {
	return StoreColumnSet{kind: storeColumnKindConversation, scalars: nil}
}

// ColumnsForDeclaration returns the column set of a document collection with
// the given saved declaration. The conversation declaration returns
// [ConversationColumns]. Its rows keep the conversation schema byte for byte.
func ColumnsForDeclaration(declaration collection.Declaration) StoreColumnSet {
	if IsConversationDeclaration(declaration) {
		return ConversationColumns()
	}
	return StoreColumnSet{kind: storeColumnKindDeclared, scalars: slices.Clone(declaration.Scalars)}
}

// ConversationScalars reports whether this column set writes the conversation
// scalar columns. The caller chooses the row shape. The store write never
// infers it from the collection name.
func (columnSet StoreColumnSet) ConversationScalars() bool {
	return columnSet.kind == storeColumnKindConversation
}

// DeclaredScalars returns the declared columns a declared write sends. It is
// nil for the code and conversation column sets.
func (columnSet StoreColumnSet) DeclaredScalars() []collection.ScalarColumn {
	if columnSet.kind != storeColumnKindDeclared {
		return nil
	}
	return columnSet.scalars
}

// creationScalars returns the scalar columns createCollection adds to a
// collection it creates for this column set.
func (columnSet StoreColumnSet) creationScalars() []collection.ScalarColumn {
	switch columnSet.kind {
	case storeColumnKindConversation:
		return ConversationDeclaration().Scalars
	case storeColumnKindDeclared:
		return columnSet.scalars
	case storeColumnKindCode:
		return nil
	default:
		return nil
	}
}

// storeColumnSetForCollection classifies a collection by name for the callers
// that rewrite rows in place and have no item source to ask (CopyChunks copies
// existing rows within one known collection). The source-driven ingest path
// passes its StoreColumnSet directly instead of calling this.
func (service *Service) storeColumnSetForCollection(collectionName string) StoreColumnSet {
	if service.isConversationCollection(collectionName) {
		return ConversationColumns()
	}
	return CodeColumns()
}

// isStagingCollection reports whether a collection name is a transient rebuild
// staging collection, by its suffix. The mmap sweep skips these: a staging
// collection is promoted onto its live name or dropped, may carry no dense index
// yet, and is not a durable surface to migrate, so sweeping it only logs noise.
func isStagingCollection(collectionName string) bool {
	return strings.HasSuffix(collectionName, stagingCollectionSuffix)
}

// conversationScalarFields returns the native scalar columns a conversation
// collection carries so Milvus can pre-filter a search by provider, workspace,
// role, time, message index, and conversation lineage. Every field is nullable
// so the same definitions serve both a freshly created collection and an
// AddCollectionField migration onto a collection with existing rows.
func conversationScalarFields() []*entity.Field {
	return scalarFields(ConversationDeclaration().Scalars)
}

// scalarFields builds the Milvus field definitions for declared scalar columns,
// in declaration order.
func scalarFields(columns []collection.ScalarColumn) []*entity.Field {
	fields := make([]*entity.Field, 0, len(columns))
	for _, column := range columns {
		fields = append(fields, scalarField(column))
	}
	return fields
}

// scalarField builds the Milvus field definition for one declared scalar
// column. A string column becomes a VarChar with the declared maximum length.
func scalarField(column collection.ScalarColumn) *entity.Field {
	field := entity.NewField().WithName(column.Name)
	switch column.Type {
	case collection.ScalarTypeString:
		field = field.WithDataType(entity.FieldTypeVarChar).WithMaxLength(int64(column.MaxLength))
	case collection.ScalarTypeBool:
		field = field.WithDataType(entity.FieldTypeBool)
	case collection.ScalarTypeInt64:
		field = field.WithDataType(entity.FieldTypeInt64)
	default:
		field = field.WithDataType(entity.FieldTypeNone)
	}
	return field.WithNullable(column.Nullable)
}

// BuiltinColumnNames returns the columns the built-in collection schema
// defines. A collection declaration may not declare any of them as a scalar
// column, and a stored schema lists them outside its declared scalars.
func BuiltinColumnNames() []string {
	return []string{
		idFieldName,
		contentFieldName,
		relativePathFieldName,
		startLineFieldName,
		endLineFieldName,
		fileExtensionFieldName,
		metadataFieldName,
		contentHashFieldName,
		embeddingModelFieldName,
		splitPartFieldName,
		denseVectorFieldName,
		sparseVectorFieldName,
	}
}

// DescribeScalarColumns reports the declared scalar columns of a stored
// collection: every schema field outside [BuiltinColumnNames]. exists is false
// when the collection is absent. A field type outside the declarable scalar
// types reports its Milvus type name, which no declaration matches.
func (service *Service) DescribeScalarColumns(
	ctx context.Context,
	collectionName string,
) ([]collection.ScalarColumn, bool, error) {
	if !service.Available() {
		return nil, false, ErrUnavailable
	}
	peerInfo, _ := peer.FromContext(ctx)
	hasCollection, err := service.hasCollection(
		ctx,
		collectionName,
		"check Milvus collection "+collectionName,
	)
	if err != nil {
		return nil, false, err
	}
	if !hasCollection {
		return nil, false, nil
	}
	described, err := service.milvus.DescribeCollection(
		ctx,
		milvusclient.NewDescribeCollectionOption(collectionName),
	)
	if err != nil {
		slog.ErrorContext(ctx, "describe collection for declared scalars failed", "collection", collectionName, "peer", peerInfo.String(), "err", err)
		return nil, false, wrapStoreError(ctx, err, "describe Milvus collection "+collectionName)
	}
	builtin := make(map[string]struct{})
	for _, name := range BuiltinColumnNames() {
		builtin[name] = struct{}{}
	}
	columns := make([]collection.ScalarColumn, 0)
	if described.Schema == nil {
		return columns, true, nil
	}
	for _, field := range described.Schema.Fields {
		if _, isBuiltin := builtin[field.Name]; isBuiltin {
			continue
		}
		column, err := storedScalarColumn(ctx, collectionName, field)
		if err != nil {
			return nil, false, err
		}
		columns = append(columns, column)
	}
	return columns, true, nil
}

// storedScalarColumn converts one stored Milvus field into the declaration
// shape a registration compares against.
func storedScalarColumn(ctx context.Context, collectionName string, field *entity.Field) (collection.ScalarColumn, error) {
	column := collection.ScalarColumn{
		Name:      field.Name,
		Type:      collection.ScalarType("milvus:" + field.DataType.Name()),
		Nullable:  field.Nullable,
		MaxLength: 0,
	}
	if field.DataType == entity.FieldTypeBool {
		column.Type = collection.ScalarTypeBool
	}
	if field.DataType == entity.FieldTypeInt64 {
		column.Type = collection.ScalarTypeInt64
	}
	if field.DataType == entity.FieldTypeVarChar {
		column.Type = collection.ScalarTypeString
		maxLength, err := strconv.ParseInt(field.TypeParams[entity.TypeParamMaxLength], 10, 32)
		if err != nil {
			slog.ErrorContext(ctx, "parse stored field max length failed", "collection", collectionName, "field", field.Name, "err", err)
			return column, fmt.Errorf("parse max length of field %s in collection %s: %w", field.Name, collectionName, err)
		}
		column.MaxLength = int32(maxLength)
	}
	return column, nil
}

func splitPartField() *entity.Field {
	return entity.NewField().
		WithName(splitPartFieldName).
		WithDataType(entity.FieldTypeInt64).
		WithNullable(true)
}

func contentHashField() *entity.Field {
	return entity.NewField().
		WithName(contentHashFieldName).
		WithDataType(entity.FieldTypeVarChar).
		WithMaxLength(64).
		WithNullable(true)
}

func embeddingModelField() *entity.Field {
	return entity.NewField().
		WithName(embeddingModelFieldName).
		WithDataType(entity.FieldTypeVarChar).
		WithMaxLength(embeddingModelFieldMaxLength).
		WithNullable(true)
}

func splitPartFieldsToAdd(schema *entity.Schema) []*entity.Field {
	if schema != nil {
		for _, field := range schema.Fields {
			if field.Name == splitPartFieldName {
				return nil
			}
		}
	}
	return []*entity.Field{splitPartField()}
}

func (service *Service) createCollection(
	ctx context.Context,
	collectionName string,
	dimension int,
	declaredScalars []collection.ScalarColumn,
) (CollectionLease, error) {
	schema := entity.NewSchema().
		WithField(entity.NewField().WithName(idFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(idFieldMaxLength).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName(contentFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(contentFieldMaxLength).WithEnableAnalyzer(true).WithEnableMatch(true)).
		WithField(entity.NewField().WithName(relativePathFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(1024)).
		WithField(entity.NewField().WithName(startLineFieldName).WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName(endLineFieldName).WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName(fileExtensionFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(32)).
		WithField(entity.NewField().WithName(metadataFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(65535)).
		WithField(contentHashField()).
		WithField(embeddingModelField()).
		WithField(splitPartField()).
		WithField(entity.NewField().WithName(denseVectorFieldName).WithDataType(entity.FieldTypeFloatVector).WithDim(int64(dimension)))

	for _, field := range scalarFields(declaredScalars) {
		schema = schema.WithField(field)
	}

	// Milvus 2.6 rejects mmap.enabled on AUTOINDEX creation. The policy is applied
	// through field and index property changes after every required index exists.
	indexOptions := []milvusclient.CreateIndexOption{
		milvusclient.NewCreateIndexOption(collectionName, denseVectorFieldName, index.NewAutoIndex(entity.COSINE)),
		milvusclient.NewCreateIndexOption(collectionName, contentHashFieldName, index.NewInvertedIndex()),
	}

	if service.cfg.HybridMode {
		schema = schema.
			WithField(entity.NewField().WithName(sparseVectorFieldName).WithDataType(entity.FieldTypeSparseVector)).
			WithFunction(entity.NewFunction().WithName("bm25").WithType(entity.FunctionTypeBM25).WithInputFields(contentFieldName).WithOutputFields(sparseVectorFieldName))
		indexOptions = append(indexOptions, milvusclient.NewCreateIndexOption(collectionName, sparseVectorFieldName, index.NewSparseInvertedIndex(entity.BM25, 0.2)))
	}

	maintenance, err := service.residency.Maintain(ctx, collectionName)
	if err != nil {
		return nil, err
	}
	if err := service.milvus.CreateCollection(ctx, milvusclient.NewCreateCollectionOption(collectionName, schema).WithIndexOptions(indexOptions...)); err != nil {
		maintenance.ReleaseContext(ctx)
		return nil, wrapStoreError(ctx, err, "create Milvus collection "+collectionName)
	}
	service.invalidateCollectionCaches(collectionName)
	maintenance.ReleaseContext(ctx)
	if err := service.ensureCreatedCollectionMmapForWrite(ctx, collectionName); err != nil {
		return nil, err
	}
	if err := service.PrepareCollection(ctx, collectionName); err != nil {
		return nil, err
	}
	lease, err := service.AcquireCollection(ctx, collectionName)
	if err != nil {
		return nil, err
	}
	return lease, nil
}

type splitPartMigration struct {
	once sync.Once
	err  error
}

type reuseIdentityMigration struct {
	once sync.Once
	err  error
}

func (service *Service) ensureReuseIdentityColumnsOnce(
	ctx context.Context,
	collectionName string,
) error {
	loaded, _ := service.ensuredReuseIdentityColumns.LoadOrStore(
		collectionName,
		&reuseIdentityMigration{once: sync.Once{}, err: nil},
	)
	migration, ok := loaded.(*reuseIdentityMigration)
	if !ok {
		return fmt.Errorf(
			"reuse identity migration guard for %s has unexpected type %T",
			collectionName,
			loaded,
		)
	}
	migration.once.Do(func() {
		maintenance, maintainErr := service.residency.Maintain(ctx, collectionName)
		if maintainErr != nil {
			migration.err = maintainErr
			service.ensuredReuseIdentityColumns.CompareAndDelete(collectionName, loaded)
			return
		}
		defer maintenance.ReleaseContext(ctx)
		migration.err = service.ensureReuseIdentityColumns(ctx, collectionName)
		if migration.err != nil {
			service.ensuredReuseIdentityColumns.CompareAndDelete(collectionName, loaded)
		}
	})
	return migration.err
}

func (service *Service) ensureReuseIdentityColumns(
	ctx context.Context,
	collectionName string,
) error {
	described, err := service.milvus.DescribeCollection(
		ctx,
		milvusclient.NewDescribeCollectionOption(collectionName),
	)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"describe collection for reuse identity failed",
			"collection", collectionName,
			"err", err,
		)
		return fmt.Errorf("describe collection %s for reuse identity: %w", collectionName, err)
	}
	existingFields := make(map[string]struct{}, len(described.Schema.Fields))
	for _, field := range described.Schema.Fields {
		existingFields[field.Name] = struct{}{}
	}
	for _, field := range []*entity.Field{contentHashField(), embeddingModelField()} {
		if _, found := existingFields[field.Name]; found {
			continue
		}
		if err := service.milvus.AddCollectionField(
			ctx,
			milvusclient.NewAddCollectionFieldOption(collectionName, field),
		); err != nil {
			return fmt.Errorf("add %s column to %s: %w", field.Name, collectionName, err)
		}
		service.invalidateMmapPolicy(collectionName)
	}
	indexes, err := service.milvus.ListIndexes(
		ctx,
		milvusclient.NewListIndexOption(collectionName).
			WithFieldName(contentHashFieldName),
	)
	if err != nil {
		return fmt.Errorf("list content hash indexes on %s: %w", collectionName, err)
	}
	if len(indexes) > 0 {
		return nil
	}
	task, err := service.milvus.CreateIndex(
		ctx,
		milvusclient.NewCreateIndexOption(
			collectionName,
			contentHashFieldName,
			index.NewInvertedIndex(),
		),
	)
	if err != nil {
		return fmt.Errorf("create content hash index on %s: %w", collectionName, err)
	}
	if err := task.Await(ctx); err != nil {
		return fmt.Errorf("await content hash index on %s: %w", collectionName, err)
	}
	service.invalidateMmapPolicy(collectionName)
	return nil
}

func (service *Service) addMissingSplitPartColumn(ctx context.Context, collectionName string) error {
	described, err := service.milvus.DescribeCollection(
		ctx,
		milvusclient.NewDescribeCollectionOption(collectionName),
	)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"describe collection for split part migration failed",
			"collection",
			collectionName,
			"err",
			err,
		)
		return fmt.Errorf("describe collection %s for split part migration: %w", collectionName, err)
	}
	fields := splitPartFieldsToAdd(described.Schema)
	for _, field := range fields {
		if err := service.milvus.AddCollectionField(
			ctx,
			milvusclient.NewAddCollectionFieldOption(collectionName, field),
		); err != nil {
			slog.ErrorContext(
				ctx,
				"add split part column failed",
				"collection",
				collectionName,
				"field",
				field.Name,
				"err",
				err,
			)
			return fmt.Errorf("add split part column to %s: %w", collectionName, err)
		}
		service.invalidateMmapPolicy(collectionName)
	}
	if len(fields) > 0 {
		slog.InfoContext(ctx, "semantic.split_part_column_added", "collection", collectionName)
	}
	return nil
}

func (service *Service) ensureSplitPartColumnOnce(
	ctx context.Context,
	collectionName string,
) error {
	loaded, _ := service.ensuredSplitPartColumns.LoadOrStore(
		collectionName,
		&splitPartMigration{once: sync.Once{}, err: nil},
	)
	migration, ok := loaded.(*splitPartMigration)
	if !ok {
		typeErr := fmt.Errorf(
			"split part migration guard for %s has unexpected type %T",
			collectionName,
			loaded,
		)
		slog.ErrorContext(ctx, "split part migration guard invalid", "err", typeErr)
		return typeErr
	}
	migration.once.Do(func() {
		maintenance, maintainErr := service.residency.Maintain(ctx, collectionName)
		if maintainErr != nil {
			migration.err = maintainErr
			service.ensuredSplitPartColumns.CompareAndDelete(collectionName, loaded)
			return
		}
		defer maintenance.ReleaseContext(ctx)
		migration.err = service.addMissingSplitPartColumn(ctx, collectionName)
		if migration.err != nil {
			service.ensuredSplitPartColumns.CompareAndDelete(collectionName, loaded)
		}
	})
	return migration.err
}

// ensureConversationScalarColumns adds any conversation scalar column the
// existing collection is missing, in place and without re-embedding, using the
// Milvus 2.5 AddCollectionField API. It is idempotent: it describes the
// collection first and only adds columns that are absent, so it is safe to run
// on every conversation-collection load. Every added column is nullable, which
// AddCollectionField requires for a collection that already holds rows. A
// freshly created collection already has the columns from createCollection, so
// this finds nothing to add. Backfilling the column values onto existing rows
// is a separate step; the columns read null until then.
// addMissingConversationScalarColumns describes the collection and adds, via the
// Milvus AddCollectionField API, any conversation scalar column it is missing,
// returning the names it added. It does not backfill values, so a caller can add
// the columns and then decide how to populate them without recursing through the
// on-add backfill trigger. Every added column is nullable, which AddCollectionField
// requires for a collection that already holds rows.
func (service *Service) addMissingConversationScalarColumns(ctx context.Context, collectionName string) ([]string, error) {
	peerInfo, _ := peer.FromContext(ctx)
	described, err := service.milvus.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "describe conversation collection for scalar migration failed", "collection", collectionName, "peer", peerInfo.String(), "err", err)
		return nil, fmt.Errorf("describe conversation collection %s: %w", collectionName, err)
	}
	existing := make(map[string]struct{})
	if described.Schema != nil {
		for _, field := range described.Schema.Fields {
			existing[field.Name] = struct{}{}
		}
	}
	added := make([]string, 0, len(conversationScalarFields()))
	for _, field := range conversationScalarFields() {
		if _, found := existing[field.Name]; found {
			continue
		}
		if err := service.milvus.AddCollectionField(ctx, milvusclient.NewAddCollectionFieldOption(collectionName, field)); err != nil {
			slog.ErrorContext(ctx, "add conversation scalar column failed", "collection", collectionName, "field", field.Name, "peer", peerInfo.String(), "err", err)
			return added, fmt.Errorf("add scalar column %s to %s: %w", field.Name, collectionName, err)
		}
		service.invalidateMmapPolicy(collectionName)
		added = append(added, field.Name)
	}
	return added, nil
}

func (service *Service) ensureConversationScalarColumns(ctx context.Context, collectionName string) error {
	if !service.isConversationCollection(collectionName) {
		return nil
	}
	hasCollection, err := service.hasCollection(
		ctx,
		collectionName,
		"check conversation collection "+collectionName,
	)
	if err != nil {
		return err
	}
	if !hasCollection {
		return nil
	}
	added, err := service.addMissingConversationScalarColumns(ctx, collectionName)
	if err != nil {
		return err
	}
	if len(added) > 0 {
		slog.InfoContext(ctx, "semantic.conversation_scalar_columns_added", "collection", collectionName, "fields", strings.Join(added, ","), "count", len(added))
		// Columns were just added to a collection that already holds rows, so
		// those rows read null until backfilled. Run the no-reindex backfill in
		// the background, detached from this request's cancellation, so this call
		// returns promptly.
		detached := context.WithoutCancel(ctx)
		go func() {
			defer func() {
				if recovered := recover(); recovered != nil {
					slog.ErrorContext(detached, "conversation scalar backfill panic", "collection", collectionName, "err", fmt.Sprintf("panic: %v", recovered))
				}
			}()
			if rows, backfillErr := service.BackfillConversationScalarColumns(detached, collectionName); backfillErr != nil {
				slog.ErrorContext(detached, "conversation scalar backfill failed", "collection", collectionName, "rows", rows, "err", backfillErr)
			}
		}()
	}
	return nil
}

// conversationScalarMigration guards the one-time scalar-column migration for a
// single conversation collection. The [sync.Once] runs the migration exactly once
// even when several goroutines race, and err caches the result so every waiter
// observes the same outcome. A failed migration is not cached as done, so a
// later call retries it.
type conversationScalarMigration struct {
	once sync.Once
	err  error
}

// ensureConversationScalarColumnsOnce runs the scalar-column migration at most
// once per conversation collection per process. The search and insert paths
// both call it so a pre-migration collection gains its native filter columns
// before the first native-filtered search or scalar-populated insert, without
// paying a DescribeCollection on every call. The per-collection [sync.Once] makes
// concurrent callers safe: only one runs DescribeCollection/AddCollectionField,
// and the rest wait and observe its result. A migration error is not retained as
// a success, so a transient failure can be retried on the next call.
func (service *Service) ensureConversationScalarColumnsOnce(ctx context.Context, collectionName string) error {
	if !service.isConversationCollection(collectionName) {
		return nil
	}
	loaded, _ := service.ensuredConvColumns.LoadOrStore(collectionName, &conversationScalarMigration{once: sync.Once{}, err: nil})
	migration, ok := loaded.(*conversationScalarMigration)
	if !ok {
		return fmt.Errorf("conversation scalar migration guard for %s has unexpected type %T", collectionName, loaded)
	}
	migration.once.Do(func() {
		maintenance, maintainErr := service.residency.Maintain(ctx, collectionName)
		if maintainErr != nil {
			migration.err = maintainErr
			service.ensuredConvColumns.CompareAndDelete(collectionName, loaded)
			return
		}
		defer maintenance.ReleaseContext(ctx)
		migration.err = service.ensureConversationScalarColumns(ctx, collectionName)
		if migration.err != nil {
			// Drop the failed guard so a later call retries the migration instead of
			// returning the cached error forever. A concurrent waiter that already
			// observed this run still sees migration.err below.
			service.ensuredConvColumns.CompareAndDelete(collectionName, loaded)
		}
	})
	return migration.err
}

// PrepareCollection applies schema migrations before a caller acquires a data lease.
func (service *Service) PrepareCollection(
	ctx context.Context,
	collectionName string,
) error {
	if isRecoveryCollection(collectionName) ||
		strings.HasPrefix(collectionName, reuseCatalogCollectionPrefix) {
		return nil
	}
	if err := service.ensureSplitPartColumnOnce(ctx, collectionName); err != nil {
		return classifyPrepareCollectionError(collectionName, err)
	}
	if err := service.ensureReuseIdentityColumnsOnce(ctx, collectionName); err != nil {
		return classifyPrepareCollectionError(collectionName, err)
	}
	return classifyPrepareCollectionError(
		collectionName,
		service.ensureConversationScalarColumnsOnce(ctx, collectionName),
	)
}

func classifyPrepareCollectionError(collectionName string, err error) error {
	if err == nil {
		return nil
	}
	if storeUnavailable(err) {
		return adapterr.NewMilvusUnavailable(fmt.Errorf("prepare Milvus collection %s: %w", collectionName, err))
	}
	return err
}

// loadCollection loads collectionName into memory and waits for the load to
// finish. A loaded collection is what makes an expression-filtered delete
// usable: Milvus answers a Delete(WithExpr(...)) on a non-primary field by
// first querying for the matching ids, which requires the collection to be
// loaded, and rejects the delete with "collection not loaded" otherwise.
// createCollection runs it once for a freshly built collection; the
// conversation upsert and delete paths run it against an already-existing
// collection before their prefix delete, since a daemon process that did not
// create the collection itself never loaded it.
//
// Concurrent callers for one collection share one initial request, both polls,
// and one recovery request. A concurrent cohort therefore issues at most two
// LoadCollection calls. The shared load runs detached from every caller, so one
// caller cancelling ends only its own wait and leaves the others waiting on a
// load that is still running; sharedCollectionLoadCeiling is what ends that load
// once no caller remains. A caller's earlier deadline still ends its own wait
// first. A collection that never finishes loading fails as not-ready instead of
// multiplying work across callers. Across different collections the transition
// itself takes a slot from the daemon-wide limiter, so a burst of cold
// collections loads a few at a time rather than all at once.
func (service *Service) loadCollection(ctx context.Context, collectionName string) error {
	return service.collectionLoads.Do(
		ctx,
		collectionName,
		service.sharedCollectionLoadCeiling(),
		func(loadCtx context.Context) error {
			return service.loadCollectionTransition(loadCtx, collectionName)
		},
	)
}

// loadCollectionTransition is the one place a LoadCollection request leaves
// this process, whichever path asked for it, so it is where the operator's
// maintenance mode, the daemon-wide cap on concurrent loads, and the
// memory-exhaustion backoff all apply. The slot covers the request, the
// load-state polls, and the single recovery request, since a collection that is
// still materializing on the query node holds memory for all of them. The
// maintenance gate and the backoff are checked before queueing for a slot,
// which fails a refused load fast. Both are checked again after taking one,
// which stops a load that acquired a slot before either began from bypassing
// them.
func (service *Service) loadCollectionTransition(
	ctx context.Context,
	collectionName string,
) error {
	backoff := service.loadBackoff()
	if err := service.admitCollectionLoad(ctx, collectionName, backoff); err != nil {
		return err
	}
	releaseSlot, err := service.collectionLoadSlots().acquire(ctx, collectionName)
	if err != nil {
		return err
	}
	defer releaseSlot()
	if err := service.admitCollectionLoad(ctx, collectionName, backoff); err != nil {
		return err
	}
	err = service.requestCollectionLoad(ctx, collectionName)
	backoff.noteLoadOutcome(ctx, collectionName, err)
	return err
}

// admitCollectionLoad applies the gates that refuse a load outright: the
// operator's maintenance mode first, then the memory-exhaustion backoff.
func (service *Service) admitCollectionLoad(
	ctx context.Context,
	collectionName string,
	backoff *collectionLoadBackoff,
) error {
	if err := service.refuseLoadDuringMaintenance(ctx, collectionName); err != nil {
		return err
	}
	return backoff.admit(ctx, collectionName)
}

// requestCollectionLoad issues the load and waits for the collection to become
// queryable under the configured bounds.
func (service *Service) requestCollectionLoad(
	ctx context.Context,
	collectionName string,
) error {
	if _, err := service.milvus.LoadCollection(
		ctx,
		milvusclient.NewLoadCollectionOption(collectionName),
	); err != nil {
		return wrapStoreError(ctx, err, "load Milvus collection "+collectionName)
	}
	return service.awaitCollectionLoaded(ctx, collectionName)
}
