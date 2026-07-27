package semantic

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/spans"
)

// The content-vector store answers one question for the whole corpus: has this
// exact content already been embedded by this exact embedder? A dense vector is
// a pure function of the text handed to the embedder, so the answer is a
// property of the content and not of where the content happens to be stored.
//
// Keeping that answer in its own collection is what makes the question cheap.
// The searchable chunk rows stay denormalized, one row per occurrence, because
// a filtered search pre-filters on scalar columns that sit on the same row as
// the vector and the per-conversation cap reads the conversation id off the
// returned row. Sharing one stored row between occurrences would break both.
// So occurrences stay copies for retrieval, and only the embedding work is
// shared, keyed by content.
const (
	contentVectorCollectionPrefix = "content_vectors_"
	contentHashFieldName          = "contentHash"
	// contentHashFieldMaxLength holds a hex SHA-256 (64 characters).
	contentHashFieldMaxLength = 64
	// contentVectorLookupBatchSize bounds how many hashes go into one primary-key
	// membership clause, mirroring the conversation id batching so a large ingest
	// splits across several queries instead of overflowing the expression-size
	// limit.
	contentVectorLookupBatchSize = 512
	// contentVectorInsertBatchSize bounds one insert of newly embedded vectors.
	contentVectorInsertBatchSize = 256
	// contentVectorIdentityDigestLength is how many hex characters of the
	// embedding-identity digest go into the collection name.
	contentVectorIdentityDigestLength = 16
)

// ReusePolicy decides whether an ingest may serve a chunk's vector from the
// content-addressed store instead of calling the embedder. It is an explicit
// parameter rather than something the store infers, because the one case that
// needs the embedder called for content the corpus already holds is a forced
// rebuild, and nothing about the content itself distinguishes that case.
type ReusePolicy int

const (
	// ReuseFromCorpus lets content the corpus has already embedded under the
	// current embedding identity skip the embedder. This is every ordinary
	// ingest, and it is the zero value so a caller that has no opinion gets the
	// cheap path.
	ReuseFromCorpus ReusePolicy = iota
	// ReuseDisabled sends every chunk to the embedder even when the corpus holds
	// a vector for its content, and records nothing back. A forced rebuild uses
	// it so an operator can replace rows that are present but stale.
	//
	// It also declines to record, which keeps the store single-valued: a hash
	// resolves to one vector, and a forced run that produced a different vector
	// for the same content would otherwise leave two entries under one hash with
	// no way to say which is current. An embedder change that genuinely
	// invalidates stored vectors should change the configured model name, which
	// changes the embedding identity and so selects a different collection.
	ReuseDisabled
)

// ContentVector pairs one content hash with the dense vector the embedder
// produced for that content.
type ContentVector struct {
	ContentHash string
	Vector      []float32
}

// EmbeddingIdentity names the embedder that produced a vector. A stored vector
// is reusable only for the identity that produced it, so the identity selects
// the collection rather than being a field inside one. That makes serving a
// vector across an embedder change structurally impossible instead of a rule a
// caller has to remember.
//
// The identity deliberately excludes the splitter settings that
// digestIndexConfig also covers. Splitter settings change where chunk
// boundaries fall, which changes the content, which changes the content hash on
// its own. They do not change the mapping from a given text to a vector.
type EmbeddingIdentity struct {
	Provider  string
	Model     string
	Dimension int32
}

// embeddingIdentity reports the identity of the embedder this service is
// configured to call.
func (service *Service) embeddingIdentity() EmbeddingIdentity {
	return EmbeddingIdentity{
		Provider:  service.cfg.EmbeddingProvider,
		Model:     service.cfg.EmbeddingModel,
		Dimension: service.cfg.EmbeddingDimension,
	}
}

// CollectionName returns the content-vector collection that holds vectors
// produced by this identity.
func (identity EmbeddingIdentity) CollectionName() string {
	payload := fmt.Sprintf("%s\x00%s\x00%d", identity.Provider, identity.Model, identity.Dimension)
	sum := sha256.Sum256([]byte(payload))
	return contentVectorCollectionPrefix + hex.EncodeToString(sum[:])[:contentVectorIdentityDigestLength]
}

// ResolveContentVectors returns the already-embedded vector for each content
// hash the store holds, for the whole corpus rather than for one conversation,
// one path, or one batch. Hashes the store does not hold are absent from the
// result and their content must be embedded.
//
// The read is a primary-key membership query, so it costs one point lookup per
// hash and returns exactly one row per hit. It does not scan the searchable
// chunk rows and it is unaffected by how many occurrences of that content those
// rows contain.
func (service *Service) ResolveContentVectors(ctx context.Context, contentHashes []string) (map[string][]float32, error) {
	resolved := make(map[string][]float32, len(contentHashes))
	if !service.Available() || len(contentHashes) == 0 {
		return resolved, nil
	}
	collectionName := service.embeddingIdentity().CollectionName()

	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "check content vector collection failed", "collection", collectionName, "err", err)
		return nil, fmt.Errorf("check Milvus collection %s: %w", collectionName, err)
	}
	if !hasCollection {
		return resolved, nil
	}
	if err := service.loadCollectionForRead(ctx, collectionName); err != nil {
		return nil, err
	}

	for _, hashBatch := range batchContentHashes(contentHashes, contentVectorLookupBatchSize) {
		if err := service.resolveContentVectorBatch(ctx, collectionName, hashBatch, resolved); err != nil {
			return nil, err
		}
	}
	return resolved, nil
}

func (service *Service) resolveContentVectorBatch(ctx context.Context, collectionName string, contentHashes []string, resolved map[string][]float32) error {
	iterator, err := service.milvus.QueryIterator(ctx, milvusclient.NewQueryIteratorOption(collectionName).
		WithBatchSize(contentVectorLookupBatchSize).
		WithFilter(inStringClause(contentHashFieldName, contentHashes)).
		WithOutputFields(contentHashFieldName, denseVectorFieldName))
	if err != nil {
		slog.ErrorContext(ctx, "open content vector query iterator failed", "collection", collectionName, "err", err)
		return fmt.Errorf("open content vector iterator for %s: %w", collectionName, err)
	}
	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			return nil
		}
		if nextErr != nil {
			slog.ErrorContext(ctx, "content vector query iterator next failed", "collection", collectionName, "err", nextErr)
			return fmt.Errorf("iterate %s for content vectors: %w", collectionName, nextErr)
		}
		hashColumn := resultSet.GetColumn(contentHashFieldName)
		vectorColumn := resultSet.GetColumn(denseVectorFieldName)
		if hashColumn == nil || vectorColumn == nil {
			return ErrSearchResultIncomplete
		}
		for rowIndex := range resultSet.ResultCount {
			contentHash, hashErr := hashColumn.GetAsString(rowIndex)
			if hashErr != nil {
				slog.ErrorContext(ctx, "read content hash column failed", "index", rowIndex, "err", hashErr)
				return fmt.Errorf("read content hash column at %d: %w", rowIndex, hashErr)
			}
			vector, vectorErr := vectorAt(vectorColumn, rowIndex)
			if vectorErr != nil {
				slog.ErrorContext(ctx, "read content vector column failed", "index", rowIndex, "err", vectorErr)
				return fmt.Errorf("read content vector column at %d: %w", rowIndex, vectorErr)
			}
			resolved[contentHash] = vector
		}
	}
}

// RecordContentVectors stores vectors the embedder has just produced so no
// later ingest embeds that content again, anywhere in the corpus.
//
// It only ever inserts. It never deletes and never upserts, so it cannot remove
// or rewrite a stored row. A hash inserted twice by two concurrent jobs leaves a
// duplicate entity, which resolution tolerates because it reads whichever copy
// the store returns and both carry the same vector for the same content.
func (service *Service) RecordContentVectors(ctx context.Context, contentVectors []ContentVector) (err error) {
	if !service.Available() || len(contentVectors) == 0 {
		return nil
	}
	ctx, done := spans.Open(ctx, "semantic.recordContentVectors")
	defer done(&err)

	collectionName := service.embeddingIdentity().CollectionName()
	dimension := len(contentVectors[0].Vector)
	if dimension == 0 {
		slog.ErrorContext(ctx, "record content vectors got a zero-dimension vector", "collection", collectionName, "err", ErrSearchResultIncomplete)
		return fmt.Errorf("record content vectors for %s: %w", collectionName, ErrSearchResultIncomplete)
	}
	if err := service.ensureContentVectorCollection(ctx, collectionName, dimension); err != nil {
		return err
	}

	for _, insertBatch := range batchContentVectors(contentVectors, contentVectorInsertBatchSize) {
		hashes := make([]string, 0, len(insertBatch))
		vectors := make([][]float32, 0, len(insertBatch))
		for _, contentVector := range insertBatch {
			if len(contentVector.Vector) != dimension {
				continue
			}
			hashes = append(hashes, contentVector.ContentHash)
			vectors = append(vectors, contentVector.Vector)
		}
		if len(hashes) == 0 {
			continue
		}
		insertOption := milvusclient.NewColumnBasedInsertOption(collectionName).
			WithVarcharColumn(contentHashFieldName, hashes).
			WithFloatVectorColumn(denseVectorFieldName, dimension, vectors)
		if _, err := service.milvus.Insert(ctx, insertOption); err != nil {
			slog.ErrorContext(ctx, "insert content vectors failed", "collection", collectionName, "rows", len(hashes), "err", err)
			return wrapStoreError(ctx, err, "insert content vectors into "+collectionName)
		}
	}
	slog.InfoContext(ctx, "semantic.content_vectors_recorded", "collection", collectionName, "rows", len(contentVectors), "dimension", dimension)
	return nil
}

// ensureContentVectorCollection creates the content-vector collection when it is
// missing. The vector field carries a FLAT index because this collection is
// never searched by similarity: it is read only by primary key. FLAT needs no
// training, so creating it stays cheap no matter how many distinct contents the
// corpus holds, and it still satisfies the Milvus rule that a loadable
// collection has every vector field indexed. mmap keeps the dense payload off
// the heap, the same reason the chunk collections enable it.
func (service *Service) ensureContentVectorCollection(ctx context.Context, collectionName string, dimension int) error {
	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "check content vector collection for create failed", "collection", collectionName, "err", err)
		return fmt.Errorf("check Milvus collection %s: %w", collectionName, err)
	}
	if hasCollection {
		return nil
	}
	schema := entity.NewSchema().
		WithField(entity.NewField().WithName(contentHashFieldName).WithDataType(entity.FieldTypeVarChar).WithMaxLength(contentHashFieldMaxLength).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName(denseVectorFieldName).WithDataType(entity.FieldTypeFloatVector).WithDim(int64(dimension)))

	createOption := milvusclient.NewCreateCollectionOption(collectionName, schema).
		WithIndexOptions(milvusclient.NewCreateIndexOption(collectionName, denseVectorFieldName, index.NewFlatIndex(entity.COSINE)))
	if err := service.milvus.CreateCollection(ctx, createOption); err != nil {
		// A concurrent job may have created it between the check and here, which is
		// not a failure: the collection now exists with the same schema.
		hasCollectionNow, checkErr := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
		if checkErr == nil && hasCollectionNow {
			return nil
		}
		slog.ErrorContext(ctx, "create content vector collection failed", "collection", collectionName, "dimension", dimension, "err", err)
		return wrapStoreError(ctx, err, "create Milvus collection "+collectionName)
	}
	slog.InfoContext(ctx, "semantic.content_vector_collection_created", "collection", collectionName, "dimension", dimension)
	outcome, err := service.ensureMmapEnabledOnce(ctx, collectionName)
	if err != nil {
		return err
	}
	if outcome == mmapOutcomeSkipped {
		return service.loadCollection(ctx, collectionName)
	}
	return nil
}

// distinctContentHashes returns each chunk's content hash once, preserving first
// appearance order. Two chunks holding identical text share one hash here, which
// is what collapses a repeated preamble into a single lookup and a single
// embedding call.
func distinctContentHashes(chunks []model.StoredChunk) []string {
	seen := make(map[string]struct{}, len(chunks))
	hashes := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		contentHash := contentVectorKey(chunk.Content)
		if _, found := seen[contentHash]; found {
			continue
		}
		seen[contentHash] = struct{}{}
		hashes = append(hashes, contentHash)
	}
	return hashes
}

// unrecordedContentVectors returns the (hash, vector) pairs among chunks whose
// hash the resolution did not already cover, each pair once. These are exactly
// the vectors this run paid the embedder for.
func unrecordedContentVectors(chunks []model.StoredChunk, vectors [][]float32, resolved map[string][]float32) []ContentVector {
	if len(chunks) != len(vectors) {
		return nil
	}
	recorded := make(map[string]struct{}, len(chunks))
	contentVectors := make([]ContentVector, 0, len(chunks))
	for index, chunk := range chunks {
		vector := vectors[index]
		if len(vector) == 0 {
			continue
		}
		contentHash := contentVectorKey(chunk.Content)
		if _, found := resolved[contentHash]; found {
			continue
		}
		if _, found := recorded[contentHash]; found {
			continue
		}
		recorded[contentHash] = struct{}{}
		contentVectors = append(contentVectors, ContentVector{ContentHash: contentHash, Vector: vector})
	}
	return contentVectors
}

func batchContentHashes(contentHashes []string, size int) [][]string {
	if size <= 0 {
		return [][]string{contentHashes}
	}
	batches := make([][]string, 0, (len(contentHashes)+size-1)/size)
	for start := 0; start < len(contentHashes); start += size {
		end := min(start+size, len(contentHashes))
		batches = append(batches, contentHashes[start:end])
	}
	return batches
}

func batchContentVectors(contentVectors []ContentVector, size int) [][]ContentVector {
	if size <= 0 {
		return [][]ContentVector{contentVectors}
	}
	batches := make([][]ContentVector, 0, (len(contentVectors)+size-1)/size)
	for start := 0; start < len(contentVectors); start += size {
		end := min(start+size, len(contentVectors))
		batches = append(batches, contentVectors[start:end])
	}
	return batches
}
