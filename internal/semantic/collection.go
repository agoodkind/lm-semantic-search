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
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/grpc/peer"
)

const (
	idFieldMaxLength             = 512
	contentFieldMaxLength        = 65_535
	embeddingModelFieldMaxLength = 65_535
)

// liveCollectionName strips the staging and promotion recovery suffixes. A
// staging or recovery twin then resolves to the live collection it replaces.
func liveCollectionName(collectionName string) string {
	trimmed := strings.TrimSuffix(collectionName, stagingCollectionSuffix)
	return strings.TrimSuffix(trimmed, recoveryCollectionSuffix)
}

type storeColumnKind int

const (
	storeColumnKindCode storeColumnKind = iota
	storeColumnKindDeclared
)

// StoreColumnSet selects code columns or caller-declared scalar columns.
type StoreColumnSet struct {
	kind    storeColumnKind
	scalars []model.ScalarColumn
}

// CodeColumns returns the column set of a code collection.
func CodeColumns() StoreColumnSet {
	return StoreColumnSet{kind: storeColumnKindCode, scalars: nil}
}

// ColumnsForDeclaration copies the scalar columns of a collection declaration.
func ColumnsForDeclaration(declaration model.CollectionDeclaration) StoreColumnSet {
	return StoreColumnSet{kind: storeColumnKindDeclared, scalars: slices.Clone(declaration.Scalars)}
}

// DeclaredScalars returns the scalar columns for a declared collection.
func (columnSet StoreColumnSet) DeclaredScalars() []model.ScalarColumn {
	if columnSet.kind != storeColumnKindDeclared {
		return nil
	}
	return columnSet.scalars
}

// creationScalars returns the scalar columns createCollection adds to a
// collection it creates for this column set.
func (columnSet StoreColumnSet) creationScalars() []model.ScalarColumn {
	switch columnSet.kind {
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
	if stored, found := service.declaredCollections.Load(liveCollectionName(collectionName)); found {
		if declaration, valid := stored.(model.CollectionDeclaration); valid {
			return ColumnsForDeclaration(declaration)
		}
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

// scalarFields builds the Milvus field definitions for declared scalar columns,
// in declaration order.
func scalarFields(columns []model.ScalarColumn) []*entity.Field {
	fields := make([]*entity.Field, 0, len(columns))
	for _, column := range columns {
		fields = append(fields, scalarField(column))
	}
	return fields
}

// scalarField builds the Milvus field definition for one declared scalar
// column. A string column becomes a VarChar with the declared maximum length.
func scalarField(column model.ScalarColumn) *entity.Field {
	field := entity.NewField().WithName(column.Name)
	switch column.Type {
	case model.ScalarTypeString:
		field = field.WithDataType(entity.FieldTypeVarChar).WithMaxLength(int64(column.MaxLength))
	case model.ScalarTypeBool:
		field = field.WithDataType(entity.FieldTypeBool)
	case model.ScalarTypeInt64:
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
) ([]model.ScalarColumn, bool, error) {
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
	collection, err := service.milvus.DescribeCollection(
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
	columns := make([]model.ScalarColumn, 0)
	if collection.Schema == nil {
		return columns, true, nil
	}
	for _, field := range collection.Schema.Fields {
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
func storedScalarColumn(ctx context.Context, collectionName string, field *entity.Field) (model.ScalarColumn, error) {
	column := model.ScalarColumn{
		Name:      field.Name,
		Type:      model.ScalarType("milvus:" + field.DataType.Name()),
		Nullable:  field.Nullable,
		MaxLength: 0,
	}
	if field.DataType == entity.FieldTypeBool {
		column.Type = model.ScalarTypeBool
	}
	if field.DataType == entity.FieldTypeInt64 {
		column.Type = model.ScalarTypeInt64
	}
	if field.DataType == entity.FieldTypeVarChar {
		column.Type = model.ScalarTypeString
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
	declaredScalars []model.ScalarColumn,
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
	collection, err := service.milvus.DescribeCollection(
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
	existingFields := make(map[string]struct{}, len(collection.Schema.Fields))
	for _, field := range collection.Schema.Fields {
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
	collection, err := service.milvus.DescribeCollection(
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
	fields := splitPartFieldsToAdd(collection.Schema)
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
	return nil
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

// RecordCollectionDeclaration stores scalar definitions for collection operations.
func (service *Service) RecordCollectionDeclaration(collectionName string, declaration model.CollectionDeclaration) {
	service.declaredCollections.Store(liveCollectionName(collectionName), declaration)
}
