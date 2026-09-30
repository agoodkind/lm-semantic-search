// Package milvus stores canonical library vectors in one Milvus collection
// through a caller-owned client. The collection uses an exact FLAT index with
// the COSINE metric and never an automatically selected index.
package milvus

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"strings"
	"sync"

	"goodkind.io/lm-semantic-search/library/observation"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/internal/storebinding"
	"goodkind.io/lm-semantic-search/library/internal/vectorcodec"
)

// Collection field and index names.
const (
	fieldVectorID       = "vector_id"
	fieldVector         = "vector"
	fieldIdentityDigest = "identity_digest"
	fieldChecksum       = "vector_checksum"
	vectorIndexName     = "vector_flat_cosine"
	hashFieldMaxLength  = 64
	indexTypeParam      = "index_type"
	metricTypeParam     = "metric_type"
	flatIndexType       = "FLAT"
	cosineMetricType    = "COSINE"
	idsTemplateName     = "vector_ids"
	idsFilterExpression = fieldVectorID + " in {" + idsTemplateName + "}"
)

// maxSearchLimit is the largest result count one Milvus search returns.
const maxSearchLimit = 16384

// Config selects the collection that stores one vector pool. The caller
// creates the client for Database, and the adapter uses the client's
// database for every request.
type Config struct {
	// Observer receives logical SDK upserts and strong verification outcomes.
	Observer   observation.Observer
	Database   string
	Collection string
}

// Store is a [library.VectorStore] backed by one Milvus collection. It never
// closes the client.
type Store struct {
	client *milvusclient.Client
	config Config
	mutex  sync.Mutex
	bound  *storebinding.Binding
}

// New returns an adapter for config.Collection. An empty database or
// collection name, or a nil client, returns an error that wraps
// [library.ErrInvalidRequest].
func New(client *milvusclient.Client, config Config) (*Store, error) {
	if client == nil || strings.TrimSpace(config.Database) == "" || strings.TrimSpace(config.Collection) == "" {
		err := fmt.Errorf("%w: milvus adapter requires a client, a database, and a collection", library.ErrInvalidRequest)
		slog.Warn("milvus adapter configuration rejected", "err", err)
		return nil, err
	}
	return &Store{client: client, config: config, mutex: sync.Mutex{}, bound: nil}, nil
}

// PoolIdentity returns the database and collection of the pool.
func (store *Store) PoolIdentity() string {
	return "milvus:" + store.config.Database + "/" + store.config.Collection
}

// BindCatalog creates the collection with the binding as its schema
// description when the collection is absent. After a successful create, an
// already-existing collection, or an ambiguous failure, it reads the
// collection schema and compares the saved catalog UUID and dimension with the
// binding. A different catalog returns an error that wraps
// [library.ErrStoreMismatch]. It never alters or drops the collection. It
// then requires the FLAT COSINE index and loads the collection.
func (store *Store) BindCatalog(ctx context.Context, binding string) error {
	requested, err := storebinding.Decode(binding)
	if err != nil {
		return fmt.Errorf("%w: %w", library.ErrInvalidRequest, err)
	}
	createErr := store.client.CreateCollection(ctx, milvusclient.NewCreateCollectionOption(store.config.Collection, collectionSchema(requested.Dimension, binding)))
	if createErr != nil {
		slog.WarnContext(ctx, "create vector collection returned an error; reading the saved binding", "collection", store.config.Collection, "err", createErr)
	}
	saved, err := store.savedBinding(ctx, requested.Dimension)
	if err != nil {
		return errors.Join(err, createErr)
	}
	if saved.CatalogUUID != requested.CatalogUUID {
		err := fmt.Errorf(
			"%w: collection %s is bound to catalog %s at %s on %s, not catalog %s",
			library.ErrStoreMismatch, store.config.Collection, saved.CatalogUUID, saved.CatalogPath, saved.WriterHost, requested.CatalogUUID,
		)
		slog.ErrorContext(ctx, "vector collection bound to another catalog", "err", err)
		return err
	}
	if err := store.ensureIndex(ctx); err != nil {
		return err
	}
	loadTask, err := store.client.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(store.config.Collection))
	if err != nil {
		slog.ErrorContext(ctx, "load vector collection failed", "collection", store.config.Collection, "err", err)
		return fmt.Errorf("load vector collection %s: %w", store.config.Collection, err)
	}
	if err := loadTask.Await(ctx); err != nil {
		slog.ErrorContext(ctx, "wait for vector collection load failed", "collection", store.config.Collection, "err", err)
		return fmt.Errorf("wait for vector collection %s load: %w", store.config.Collection, err)
	}
	store.mutex.Lock()
	store.bound = &saved
	store.mutex.Unlock()
	return nil
}

func collectionSchema(dimension int, description string) *entity.Schema {
	return entity.NewSchema().
		WithDescription(description).
		WithField(entity.NewField().WithName(fieldVectorID).WithDataType(entity.FieldTypeVarChar).WithMaxLength(hashFieldMaxLength).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName(fieldVector).WithDataType(entity.FieldTypeFloatVector).WithDim(int64(dimension))).
		WithField(entity.NewField().WithName(fieldIdentityDigest).WithDataType(entity.FieldTypeVarChar).WithMaxLength(hashFieldMaxLength)).
		WithField(entity.NewField().WithName(fieldChecksum).WithDataType(entity.FieldTypeVarChar).WithMaxLength(hashFieldMaxLength))
}

// savedBinding reads the collection schema and decodes its binding. A schema
// without the expected fields or dimension returns an error that wraps
// [library.ErrStoreMismatch].
func (store *Store) savedBinding(ctx context.Context, dimension int) (storebinding.Binding, error) {
	collection, err := store.client.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(store.config.Collection))
	if err != nil {
		slog.ErrorContext(ctx, "read vector collection schema failed", "collection", store.config.Collection, "err", err)
		return storebinding.Binding{}, fmt.Errorf("read vector collection %s schema: %w", store.config.Collection, err)
	}
	saved, err := storebinding.Decode(collection.Schema.Description)
	if err != nil {
		mismatch := fmt.Errorf("%w: collection %s has no library catalog binding: %w", library.ErrStoreMismatch, store.config.Collection, err)
		slog.ErrorContext(ctx, "vector collection binding unreadable", "err", mismatch)
		return storebinding.Binding{}, mismatch
	}
	if saved.Dimension != dimension || !hasField(collection.Schema, fieldVector) || !hasField(collection.Schema, fieldChecksum) {
		mismatch := fmt.Errorf("%w: collection %s schema does not match dimension %d", library.ErrStoreMismatch, store.config.Collection, dimension)
		slog.ErrorContext(ctx, "vector collection schema mismatch", "err", mismatch)
		return storebinding.Binding{}, mismatch
	}
	return saved, nil
}

func hasField(schema *entity.Schema, name string) bool {
	for _, field := range schema.Fields {
		if field.Name == name {
			return true
		}
	}
	return false
}

// ensureIndex creates the FLAT COSINE index when it is absent and rejects any
// other index on the vector field.
func (store *Store) ensureIndex(ctx context.Context) error {
	existing, err := store.client.DescribeIndex(ctx, milvusclient.NewDescribeIndexOption(store.config.Collection, vectorIndexName))
	if err == nil {
		params := existing.Params()
		if params[indexTypeParam] != flatIndexType || params[metricTypeParam] != cosineMetricType {
			mismatch := fmt.Errorf(
				"%w: collection %s index %s is %s %s, want %s %s",
				library.ErrStoreMismatch, store.config.Collection, vectorIndexName,
				params[indexTypeParam], params[metricTypeParam], flatIndexType, cosineMetricType,
			)
			slog.ErrorContext(ctx, "vector index mismatch", "err", mismatch)
			return mismatch
		}
		return nil
	}
	slog.InfoContext(ctx, "create FLAT COSINE vector index", "collection", store.config.Collection, "read_index_err", err)
	task, err := store.client.CreateIndex(
		ctx,
		milvusclient.NewCreateIndexOption(store.config.Collection, fieldVector, index.NewFlatIndex(entity.COSINE)).WithIndexName(vectorIndexName),
	)
	if err != nil {
		slog.ErrorContext(ctx, "create vector index failed", "collection", store.config.Collection, "err", err)
		return fmt.Errorf("create vector index on %s: %w", store.config.Collection, err)
	}
	if err := task.Await(ctx); err != nil {
		slog.ErrorContext(ctx, "wait for vector index failed", "collection", store.config.Collection, "err", err)
		return fmt.Errorf("wait for vector index on %s: %w", store.config.Collection, err)
	}
	return nil
}

func (store *Store) binding(ctx context.Context) (storebinding.Binding, error) {
	store.mutex.Lock()
	defer store.mutex.Unlock()
	if store.bound == nil {
		err := fmt.Errorf("milvus adapter for %s is not bound to a catalog", store.config.Collection)
		slog.ErrorContext(ctx, "milvus adapter used before binding", "err", err)
		return storebinding.Binding{}, err
	}
	return *store.bound, nil
}

// PutCanonical upserts one vector. The record values must pass validation, and
// the record checksum must equal the checksum of those values. An upsert of
// identical bytes leaves the live vector unchanged.
func (store *Store) PutCanonical(ctx context.Context, record library.VectorRecord) error {
	bound, err := store.binding(ctx)
	if err != nil {
		return err
	}
	if err := vectorcodec.Validate(record.Values, bound.Dimension); err != nil {
		slog.WarnContext(ctx, "reject canonical vector", "vector_id", record.ID, "err", err)
		return fmt.Errorf("%w: vector %s: %w", library.ErrInvalidRequest, record.ID, err)
	}
	if vectorcodec.Checksum(record.Values) != record.Checksum {
		err := fmt.Errorf("%w: vector %s checksum does not match its values", library.ErrInvalidRequest, record.ID)
		slog.WarnContext(ctx, "reject canonical vector", "err", err)
		return err
	}
	option := milvusclient.NewColumnBasedInsertOption(
		store.config.Collection,
		column.NewColumnVarChar(fieldVectorID, []string{record.ID}),
		column.NewColumnFloatVector(fieldVector, bound.Dimension, [][]float32{record.Values}),
		column.NewColumnVarChar(fieldIdentityDigest, []string{record.IdentityDigest}),
		column.NewColumnVarChar(fieldChecksum, []string{record.Checksum}),
	)
	ctx, span := observation.Start(ctx, store.config.Observer, observation.UpsertCall)
	result, err := store.client.Upsert(ctx, option)
	counts := observation.VectorData{Requested: 1}
	if err == nil {
		counts.Acknowledged = result.UpsertCount
	}
	span.End(ctx, err, observation.Data{Vector: counts})
	if err != nil {
		slog.ErrorContext(ctx, "upsert canonical vector failed", "vector_id", record.ID, "err", err)
		return fmt.Errorf("upsert canonical vector %s: %w", record.ID, err)
	}
	return nil
}

// VerifyStrong reads every identity with strong consistency. A missing ID
// returns an error that wraps [library.ErrVectorMissing]. A stored digest or
// checksum that differs from the identity, or stored values that do not match
// the stored checksum, return an error that wraps [library.ErrVectorCorrupt].
func (store *Store) VerifyStrong(ctx context.Context, identities []library.VectorIdentity) (err error) {
	ctx, span := observation.Start(ctx, store.config.Observer, observation.StrongVerification)
	counts := observation.VectorData{Requested: len(identities)}
	defer func() { span.End(ctx, err, observation.Data{Vector: counts}) }()
	if _, err := store.binding(ctx); err != nil {
		return err
	}
	if len(identities) == 0 {
		return nil
	}
	ids := make([]string, 0, len(identities))
	for _, identity := range identities {
		ids = append(ids, identity.ID)
	}
	saved, err := store.readVectors(ctx, ids)
	if err != nil {
		return err
	}
	for _, identity := range identities {
		vector, found := saved[identity.ID]
		if !found {
			missing := fmt.Errorf("%w: vector %s is absent from %s", library.ErrVectorMissing, identity.ID, store.config.Collection)
			slog.WarnContext(ctx, "canonical vector missing", "err", missing)
			return missing
		}
		if vector.digest != identity.IdentityDigest || vector.checksum != identity.Checksum || vectorcodec.Checksum(vector.values) != vector.checksum {
			corrupt := fmt.Errorf("%w: vector %s in %s does not match its identity or checksum", library.ErrVectorCorrupt, identity.ID, store.config.Collection)
			slog.ErrorContext(ctx, "canonical vector corrupt", "err", corrupt)
			return corrupt
		}
	}
	counts.Verified = len(identities)
	return nil
}

type savedVector struct {
	digest   string
	checksum string
	values   []float32
}

func (store *Store) readVectors(ctx context.Context, ids []string) (map[string]savedVector, error) {
	result, err := store.client.Query(
		ctx,
		milvusclient.NewQueryOption(store.config.Collection).
			WithIDs(column.NewColumnVarChar(fieldVectorID, ids)).
			WithOutputFields(fieldVectorID, fieldIdentityDigest, fieldChecksum, fieldVector).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		slog.ErrorContext(ctx, "strong read of canonical vectors failed", "vectors", len(ids), "err", err)
		return nil, fmt.Errorf("strong read of %d canonical vectors: %w", len(ids), err)
	}
	saved := make(map[string]savedVector, result.ResultCount)
	if result.ResultCount == 0 {
		return saved, nil
	}
	vectorColumn, ok := result.GetColumn(fieldVector).(*column.ColumnFloatVector)
	if !ok {
		err := fmt.Errorf("strong read of %s returned no float vector column", store.config.Collection)
		slog.ErrorContext(ctx, "strong read of canonical vectors failed", "err", err)
		return nil, err
	}
	vectors := vectorColumn.Data()
	for row := range result.ResultCount {
		id, idErr := result.GetColumn(fieldVectorID).GetAsString(row)
		digest, digestErr := result.GetColumn(fieldIdentityDigest).GetAsString(row)
		checksum, checksumErr := result.GetColumn(fieldChecksum).GetAsString(row)
		if joined := errors.Join(idErr, digestErr, checksumErr); joined != nil {
			slog.ErrorContext(ctx, "decode strong read row failed", "err", joined)
			return nil, fmt.Errorf("decode strong read row: %w", joined)
		}
		saved[id] = savedVector{digest: digest, checksum: checksum, values: vectors[row]}
	}
	return saved, nil
}

// ScoreExact scores every requested ID against query with an exhaustive FLAT
// COSINE search limited to those IDs. It returns one finite score for each ID
// in request order. A missing ID returns an error that wraps
// [library.ErrVectorMissing]. A duplicate or unexpected ID, or a nonfinite
// score, returns an error that wraps [library.ErrVectorCorrupt].
func (store *Store) ScoreExact(ctx context.Context, query []float32, ids []string) ([]library.VectorScore, error) {
	bound, err := store.binding(ctx)
	if err != nil {
		return nil, err
	}
	if err := validateScoreRequest(ctx, query, ids, bound.Dimension); err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return []library.VectorScore{}, nil
	}
	results, err := store.client.Search(
		ctx,
		milvusclient.NewSearchOption(store.config.Collection, len(ids), []entity.Vector{entity.FloatVector(query)}).
			WithANNSField(fieldVector).
			WithFilter(idsFilterExpression).
			WithTemplateParam(idsTemplateName, ids).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		slog.ErrorContext(ctx, "exact vector search failed", "vectors", len(ids), "err", err)
		return nil, fmt.Errorf("exact vector search over %d vectors: %w", len(ids), err)
	}
	scores, err := collectScores(ctx, results)
	if err != nil {
		return nil, err
	}
	ordered := make([]library.VectorScore, 0, len(ids))
	for _, id := range ids {
		score, found := scores[id]
		if !found {
			missing := fmt.Errorf("%w: exact search of %s returned no score for %s", library.ErrVectorMissing, store.config.Collection, id)
			slog.ErrorContext(ctx, "exact vector search missed an ID", "err", missing)
			return nil, missing
		}
		ordered = append(ordered, library.VectorScore{ID: id, Score: score})
	}
	if len(scores) != len(ids) {
		unexpected := fmt.Errorf("%w: exact search returned %d IDs for %d requested", library.ErrVectorCorrupt, len(scores), len(ids))
		slog.ErrorContext(ctx, "exact vector search returned extra IDs", "err", unexpected)
		return nil, unexpected
	}
	return ordered, nil
}

func validateScoreRequest(ctx context.Context, query []float32, ids []string, dimension int) error {
	if err := vectorcodec.Validate(query, dimension); err != nil {
		invalid := fmt.Errorf("%w: query vector: %w", library.ErrInvalidRequest, err)
		slog.WarnContext(ctx, "reject exact score request", "err", invalid)
		return invalid
	}
	if len(ids) > maxSearchLimit {
		invalid := fmt.Errorf("%w: %d IDs exceed the %d-result search limit", library.ErrInvalidRequest, len(ids), maxSearchLimit)
		slog.WarnContext(ctx, "reject exact score request", "err", invalid)
		return invalid
	}
	seen := make(map[string]bool, len(ids))
	for _, id := range ids {
		if seen[id] {
			invalid := fmt.Errorf("%w: ID %s appears twice", library.ErrInvalidRequest, id)
			slog.WarnContext(ctx, "reject exact score request", "err", invalid)
			return invalid
		}
		seen[id] = true
	}
	return nil
}

func collectScores(ctx context.Context, results []milvusclient.ResultSet) (map[string]float64, error) {
	scores := make(map[string]float64)
	for _, result := range results {
		if result.Err != nil {
			slog.ErrorContext(ctx, "exact vector search result failed", "err", result.Err)
			return nil, fmt.Errorf("exact vector search result: %w", result.Err)
		}
		for row := range result.ResultCount {
			id, err := result.IDs.GetAsString(row)
			if err != nil {
				slog.ErrorContext(ctx, "decode exact search ID failed", "err", err)
				return nil, fmt.Errorf("decode exact search ID: %w", err)
			}
			score := float64(result.Scores[row])
			if math.IsNaN(score) || math.IsInf(score, 0) {
				corrupt := fmt.Errorf("%w: exact search score for %s is not finite", library.ErrVectorCorrupt, id)
				slog.ErrorContext(ctx, "exact vector search result invalid", "err", corrupt)
				return nil, corrupt
			}
			if _, duplicate := scores[id]; duplicate {
				corrupt := fmt.Errorf("%w: exact search returned %s twice", library.ErrVectorCorrupt, id)
				slog.ErrorContext(ctx, "exact vector search result invalid", "err", corrupt)
				return nil, corrupt
			}
			scores[id] = score
		}
	}
	return scores, nil
}
