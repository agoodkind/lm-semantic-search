package semantic

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strconv"
	"strings"
	"sync"

	"goodkind.io/lm-semantic-search/internal/model"

	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"google.golang.org/grpc/peer"
)

const (
	idFieldMaxLength             = 512
	contentFieldMaxLength        = 65_535
	embeddingModelFieldMaxLength = 65535
)

type storeColumnKind int

const (
	storeColumnKindCode storeColumnKind = iota
	storeColumnKindDeclared
)

// StoreColumnSet is the set of scalar columns a store write populates and a
// created collection declares. The ingest caller (the item source) passes it
// into the write path. insertBatch never derives the row shape from the
// collection name. A code write sends only the base columns. A declared write
// also sends each declared column from [model.StoredChunk.Scalars].
type StoreColumnSet struct {
	kind    storeColumnKind
	scalars []collection.ScalarColumn
}

// CodeColumns returns the column set of a code collection.
func CodeColumns() StoreColumnSet {
	return StoreColumnSet{kind: storeColumnKindCode, scalars: nil}
}

// ColumnsForDeclaration returns the column set of a document collection with
// the given saved declaration.
func ColumnsForDeclaration(declaration collection.Declaration) StoreColumnSet {
	return StoreColumnSet{kind: storeColumnKindDeclared, scalars: slices.Clone(declaration.Scalars)}
}

// DeclaredScalars returns the declared columns a declared write sends. It is
// nil for the code column set.
func (columnSet StoreColumnSet) DeclaredScalars() []collection.ScalarColumn {
	if columnSet.kind != storeColumnKindDeclared {
		return nil
	}
	return columnSet.scalars
}

// creationScalars returns the scalar columns createCollection adds to a
// collection it creates for this column set.
func (columnSet StoreColumnSet) creationScalars() []collection.ScalarColumn {
	if columnSet.kind == storeColumnKindDeclared {
		return columnSet.scalars
	}
	return nil
}

// rowScalars returns the scalar values an insert of chunk writes for this
// column set. A declared row stores the chunk's own values, and a code row has
// none.
func (columnSet StoreColumnSet) rowScalars(chunk model.StoredChunk) map[string]collection.ScalarValue {
	if columnSet.kind == storeColumnKindDeclared {
		return chunk.Scalars
	}
	return nil
}

// isStagingCollection reports whether a collection name is a transient rebuild
// staging collection, by its suffix. The mmap sweep skips these: a staging
// collection is promoted onto its live name or dropped, may carry no dense index
// yet, and is not a durable surface to migrate, so sweeping it only logs noise.
func isStagingCollection(collectionName string) bool {
	return strings.HasSuffix(collectionName, stagingCollectionSuffix)
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
	for _, name := range milvusstore.BuiltinColumnNames() {
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

func splitPartFieldsToAdd(schema *entity.Schema) []*entity.Field {
	if schema != nil {
		for _, field := range schema.Fields {
			if field.Name == splitPartFieldName {
				return nil
			}
		}
	}
	return []*entity.Field{milvusstore.SplitPartFieldSchema()}
}

func (service *Service) createCollection(
	ctx context.Context,
	collectionName string,
	dimension int,
	declaredScalars []collection.ScalarColumn,
) (CollectionLease, error) {
	maintenance, err := service.residency.Maintain(ctx, collectionName)
	if err != nil {
		return nil, err
	}
	createOption := service.collectionStore().CreateCollectionOption(collectionName, dimension, declaredScalars)
	if err := service.milvus.CreateCollection(ctx, createOption); err != nil {
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
	for _, field := range []*entity.Field{milvusstore.ContentHashFieldSchema(), milvusstore.EmbeddingModelFieldSchema()} {
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
	return classifyPrepareCollectionError(
		collectionName,
		service.ensureReuseIdentityColumnsOnce(ctx, collectionName),
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
