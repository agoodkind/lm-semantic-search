package semantic

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// Conversation collections carry their filterable attributes as native scalar
// columns so Milvus can pre-filter a search by them, rather than the engine
// over-fetching and post-filtering the JSON metadata column. These columns
// exist only on conversation collections (conv_chunks_*), which are owned
// solely by this daemon, so they never reach the TS-adapter-owned code
// collections. The values are still mirrored into the metadata JSON for
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
	conversationIDFieldMaxLength   = 256
	conversationRoleFieldMaxLength = 64
	conversationProviderMaxLength  = 32
	conversationWorkspaceMaxLength = 1024
)

// isConversationCollection reports whether a collection name addresses a
// conversation document collection (including its staging twin), which is the
// only kind that carries the conversation scalar columns.
func isConversationCollection(collectionName string) bool {
	return strings.HasPrefix(collectionName, conversationCollectionPrefix)
}

// conversationScalarFields returns the native scalar columns a conversation
// collection carries so Milvus can pre-filter a search by provider, workspace,
// role, time, message index, and conversation lineage. Every field is nullable
// so the same definitions serve both a freshly created collection and an
// AddCollectionField migration onto a collection with existing rows.
func conversationScalarFields() []*entity.Field {
	return []*entity.Field{
		entity.NewField().WithName(conversationIDFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(conversationIDFieldMaxLength).WithNullable(true),
		entity.NewField().WithName(parentConversationIDFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(conversationIDFieldMaxLength).WithNullable(true),
		entity.NewField().WithName(roleFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(conversationRoleFieldMaxLength).WithNullable(true),
		entity.NewField().WithName(providerFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(conversationProviderMaxLength).WithNullable(true),
		entity.NewField().WithName(workspaceRootFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(conversationWorkspaceMaxLength).WithNullable(true),
		entity.NewField().WithName(archivedFieldName).WithDataType(entity.FieldTypeBool).WithNullable(true),
		entity.NewField().WithName(timestampUnixFieldName).WithDataType(entity.FieldTypeInt64).WithNullable(true),
		entity.NewField().WithName(messageIndexFieldName).WithDataType(entity.FieldTypeInt64).WithNullable(true),
	}
}

func (service *Service) createCollection(ctx context.Context, collectionName string, dimension int) error {
	schema := entity.NewSchema().
		WithField(entity.NewField().WithName(idFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(512).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName(contentFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(65535).WithEnableAnalyzer(true).WithEnableMatch(true)).
		WithField(entity.NewField().WithName(relativePathFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(1024)).
		WithField(entity.NewField().WithName(startLineFieldName).WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName(endLineFieldName).WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName(fileExtensionFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(32)).
		WithField(entity.NewField().WithName(metadataFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(65535)).
		WithField(entity.NewField().WithName(denseVectorFieldName).WithDataType(entity.FieldTypeFloatVector).WithDim(int64(dimension)).WithTypeParams(mmapEnabledKey, mmapEnabledValue))

	if isConversationCollection(collectionName) {
		for _, field := range conversationScalarFields() {
			schema = schema.WithField(field)
		}
	}

	// Born mmapped: the dense vector field carries its mmap type param above and
	// the dense index carries the matching index-level mmap here, so a freshly
	// created collection never needs the in-place migration sweep. WithExtraParam
	// mutates in place and returns nothing, so the option is built as a variable
	// rather than chained into the slice literal.
	denseIndexOption := milvusclient.NewCreateIndexOption(collectionName, denseVectorFieldName, index.NewAutoIndex(entity.COSINE))
	denseIndexOption.WithExtraParam(mmapEnabledKey, mmapEnabledValue)
	indexOptions := []milvusclient.CreateIndexOption{denseIndexOption}

	if service.cfg.HybridMode {
		schema = schema.
			WithField(entity.NewField().WithName(sparseVectorFieldName).WithDataType(entity.FieldTypeSparseVector)).
			WithFunction(entity.NewFunction().WithName("bm25").WithType(entity.FunctionTypeBM25).WithInputFields(contentFieldName).WithOutputFields(sparseVectorFieldName))
		indexOptions = append(indexOptions, milvusclient.NewCreateIndexOption(collectionName, sparseVectorFieldName, index.NewSparseInvertedIndex(entity.BM25, 0.2)))
	}

	if err := service.milvus.CreateCollection(ctx, milvusclient.NewCreateCollectionOption(collectionName, schema).WithIndexOptions(indexOptions...)); err != nil {
		slog.ErrorContext(ctx, "create Milvus collection failed", "collection", collectionName, "err", err)
		return fmt.Errorf("create Milvus collection %s: %w", collectionName, err)
	}
	return service.loadCollection(ctx, collectionName)
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
	collection, err := service.milvus.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "describe conversation collection for scalar migration failed", "collection", collectionName, "err", err)
		return nil, fmt.Errorf("describe conversation collection %s: %w", collectionName, err)
	}
	existing := make(map[string]struct{})
	if collection.Schema != nil {
		for _, field := range collection.Schema.Fields {
			existing[field.Name] = struct{}{}
		}
	}
	added := make([]string, 0, len(conversationScalarFields()))
	for _, field := range conversationScalarFields() {
		if _, found := existing[field.Name]; found {
			continue
		}
		if err := service.milvus.AddCollectionField(ctx, milvusclient.NewAddCollectionFieldOption(collectionName, field)); err != nil {
			slog.ErrorContext(ctx, "add conversation scalar column failed", "collection", collectionName, "field", field.Name, "err", err)
			return added, fmt.Errorf("add scalar column %s to %s: %w", field.Name, collectionName, err)
		}
		added = append(added, field.Name)
	}
	return added, nil
}

func (service *Service) ensureConversationScalarColumns(ctx context.Context, collectionName string) error {
	if !isConversationCollection(collectionName) {
		return nil
	}
	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "check conversation collection for scalar migration failed", "collection", collectionName, "err", err)
		return fmt.Errorf("check conversation collection %s: %w", collectionName, err)
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
		needsBackfill, backfillCheckErr := service.conversationCollectionNeedsBackfill(ctx, collectionName)
		if backfillCheckErr != nil {
			slog.ErrorContext(ctx, "check conversation scalar backfill need failed", "collection", collectionName, "err", backfillCheckErr)
			return nil
		}
		if !needsBackfill {
			return nil
		}
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
	if !isConversationCollection(collectionName) {
		return nil
	}
	loaded, _ := service.ensuredConvColumns.LoadOrStore(collectionName, &conversationScalarMigration{once: sync.Once{}, err: nil})
	migration, ok := loaded.(*conversationScalarMigration)
	if !ok {
		return fmt.Errorf("conversation scalar migration guard for %s has unexpected type %T", collectionName, loaded)
	}
	migration.once.Do(func() {
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

// loadCollection loads collectionName into memory and waits for the load to
// finish. A loaded collection is what makes an expression-filtered delete
// usable: Milvus answers a Delete(WithExpr(...)) on a non-primary field by
// first querying for the matching ids, which requires the collection to be
// loaded, and rejects the delete with "collection not loaded" otherwise.
// createCollection runs it once for a freshly built collection; the
// conversation upsert and delete paths run it against an already-existing
// collection before their prefix delete, since a daemon process that did not
// create the collection itself never loaded it.
func (service *Service) loadCollection(ctx context.Context, collectionName string) error {
	loadTask, err := service.milvus.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "load Milvus collection failed", "collection", collectionName, "err", err)
		return fmt.Errorf("load Milvus collection %s: %w", collectionName, err)
	}
	if err := loadTask.Await(ctx); err != nil {
		slog.ErrorContext(ctx, "await Milvus collection load failed", "collection", collectionName, "err", err)
		return fmt.Errorf("await Milvus collection load %s: %w", collectionName, err)
	}
	return nil
}
