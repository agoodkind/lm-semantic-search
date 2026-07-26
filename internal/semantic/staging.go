package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/spans"
)

// StageReindex embeds chunks into the staging collection that PromoteStaging
// later swaps onto the live name. The daemon calls it once per file during a
// first index (and a forced rebuild), so the live collection a search reads is
// never a partially built one: it either holds the previous index or, after
// the swap, the complete new one. The staging collection is created lazily on
// the first inserted batch with the embedding dimension taken from the first
// returned vector, so the dimension is never guessed up front.
//
// removal drops the item's prior rows from staging first when staging already
// exists, which keeps a re-embedded item idempotent: if a crash lands between an
// item's insert and its checkpoint, the resumed run re-embeds that one item and
// its prior staging rows are removed before the fresh rows land. A nil chunk
// slice with an empty removal is a no-op.
func (service *Service) StageReindex(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removal Removal, progress func(Progress), reuse map[string][]float32, columnSet StoreColumnSet) (err error) {
	ctx, done := spans.Open(ctx, "semantic.stageReindex")
	defer done(&err)

	if !service.Available() {
		return nil
	}

	stagingName := stagingCollectionName(service.CollectionName(codebasePath))
	hasStaging, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(stagingName))
	if err != nil {
		return wrapStoreError(ctx, err, "check staging collection "+stagingName)
	}

	if hasStaging && !removal.Empty() {
		if err := service.deleteByRemoval(ctx, stagingName, removal); err != nil {
			return err
		}
	}
	if len(chunks) == 0 {
		return nil
	}
	chunks = service.guardrailExpand(ctx, codebasePath, chunks, "stage")
	return service.insertChunksBatched(ctx, stagingName, chunks, hasStaging, "Generating embeddings and writing to Milvus...", progress, reuse, columnSet)
}

// PromoteStaging atomically swaps the staging collection onto the live
// collection name: it drops the current live collection, which is a no-op on a
// first index where none exists, then renames staging onto it. The daemon runs
// it once, after every file's chunks are staged. It returns
// ErrCollectionMissing when no staging collection exists to promote.
func (service *Service) PromoteStaging(ctx context.Context, codebasePath string) (err error) {
	ctx, done := spans.Open(ctx, "semantic.promoteStaging")
	defer done(&err)

	if !service.Available() {
		return nil
	}

	collectionName := service.CollectionName(codebasePath)
	stagingName := stagingCollectionName(collectionName)
	hasStaging, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(stagingName))
	if err != nil {
		return wrapStoreError(ctx, err, "check staging collection "+stagingName)
	}
	if !hasStaging {
		return ErrCollectionMissing
	}

	// A failure before this point leaves the previous live collection serving
	// queries; only these two metadata operations replace it.
	if err := service.dropIfExists(ctx, collectionName); err != nil {
		return err
	}
	return service.renameCollection(ctx, stagingName, collectionName)
}

// HasStaging reports whether a staging collection exists for the codebase.
// The daemon uses it on resume to decide whether a persisted checkpoint can be
// trusted: a checkpoint plus a present staging collection means the partial
// build survived, so embedded files are skipped; a missing staging collection
// means the partial build was lost, so the build restarts from the first file.
func (service *Service) HasStaging(ctx context.Context, codebasePath string) (bool, error) {
	if !service.Available() {
		return false, nil
	}
	stagingName := stagingCollectionName(service.CollectionName(codebasePath))
	hasStaging, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(stagingName))
	if err != nil {
		slog.ErrorContext(ctx, "check staging collection presence failed", "collection", stagingName, "err", err)
		return false, fmt.Errorf("check staging collection %s: %w", stagingName, err)
	}
	return hasStaging, nil
}

// DropStaging removes any staging collection for the codebase. The daemon
// calls it before a fresh build so a stale partial staging from an abandoned
// run never contaminates the new one. Safe when no staging collection exists.
func (service *Service) DropStaging(ctx context.Context, codebasePath string) error {
	if !service.Available() {
		return nil
	}
	return service.dropIfExists(ctx, stagingCollectionName(service.CollectionName(codebasePath)))
}

const (
	defaultEmbeddingBatchRows        = 32
	defaultEmbeddingBatchTokenBudget = 6000
)

func (service *Service) packForEmbedding(chunks []model.StoredChunk) [][]model.StoredChunk {
	batchRows := service.cfg.EmbeddingBatchSize
	if batchRows <= 0 {
		batchRows = defaultEmbeddingBatchRows
	}
	tokenBudget := service.cfg.EmbeddingBatchTokenBudget
	if tokenBudget <= 0 {
		tokenBudget = defaultEmbeddingBatchTokenBudget
	}
	return packChunksByEstimatedTokens(chunks, batchRows, tokenBudget)
}

// insertChunksBatched embeds chunks in row-count and estimated-token capped
// batches and inserts them into collectionName. When collectionReady is false
// the collection is created on the first batch using the dimension of the first
// returned vector, which is how both the staging build and an empty live
// collection learn their dimension without an up-front guess. The caller
// guarantees chunks is non-empty and already guardrail-expanded.
func (service *Service) insertChunksBatched(ctx context.Context, collectionName string, chunks []model.StoredChunk, collectionReady bool, phase string, progress func(Progress), reuse map[string][]float32, columnSet StoreColumnSet) error {
	packs := service.packForEmbedding(chunks)
	totalBatches := len(packs)
	var writtenRows int32
	var reusedRows int32
	var embeddedRows int32

	for batchIndex, chunkBatch := range packs {
		vectors, reused, err := service.embedChunkBatch(ctx, chunkBatch, reuse)
		if err != nil {
			return err
		}

		keptChunks, keptVectors := filterEmbeddedChunks(chunkBatch, vectors)

		// An input the endpoint rejected as un-embeddable (for example a dense chunk
		// still over the model's context window after the byte-budget pre-split) has
		// a nil vector. Re-split each such chunk smaller and retry its pieces until
		// every piece embeds or a piece is a single indivisible codepoint, so an
		// oversized input is divided and indexed instead of dropped.
		oversized := collectOversizedChunks(chunkBatch, vectors)
		if len(oversized) > 0 {
			retryChunks, retryVectors, _, retryErr := EmbedChunksSplittingOversize(ctx, oversized, service.embedder.EmbedBatch)
			if retryErr != nil {
				return retryErr
			}
			keptChunks = append(keptChunks, retryChunks...)
			keptVectors = append(keptVectors, retryVectors...)
		}
		if len(keptChunks) == 0 {
			continue
		}

		if !collectionReady {
			dimension := len(keptVectors[0])
			if err := service.createCollection(ctx, collectionName, dimension); err != nil {
				return err
			}
			collectionReady = true
		}

		if err := service.insertBatch(ctx, collectionName, keptChunks, keptVectors, columnSet); err != nil {
			return err
		}

		writtenRows += safeInt32FromInt(len(keptChunks))
		reusedRows += safeInt32FromInt(reused)
		embeddedRows += safeInt32FromInt(len(keptChunks) - reused)
		if progress != nil {
			progress(Progress{
				Phase:                     phase,
				OverallPercent:            90 + (float64(batchIndex+1)/float64(totalBatches))*10,
				EmbeddingBatchesTotal:     safeInt32FromInt(totalBatches),
				EmbeddingBatchesCompleted: safeInt32FromInt(batchIndex + 1),
				CollectionRowsWritten:     writtenRows,
				ChunksProcessed:           writtenRows,
				ChunksReused:              reusedRows,
				ChunksEmbedded:            embeddedRows,
			})
		}
	}
	return nil
}

// embedChunkBatch returns one dense vector per chunk in chunkBatch, in order,
// plus the count of chunks served from the reuse map (which never reach the
// embedder). When reuse holds a vector for a chunk's content (keyed by content
// hash) that vector is taken directly; only the remaining misses are embedded in
// a single batch. A nil or empty reuse map makes this embed every chunk, which
// is the ordinary first-index behavior, and reports zero reused.
func (service *Service) embedChunkBatch(ctx context.Context, chunkBatch []model.StoredChunk, reuse map[string][]float32) ([][]float32, int, error) {
	vectors := make([][]float32, len(chunkBatch))
	missTexts := make([]string, 0, len(chunkBatch))
	missIndexes := make([]int, 0, len(chunkBatch))
	for index, chunk := range chunkBatch {
		if len(reuse) > 0 {
			if vector, hit := reuse[contentVectorKey(chunk.Content)]; hit {
				vectors[index] = vector
				continue
			}
		}
		missTexts = append(missTexts, chunk.Content)
		missIndexes = append(missIndexes, index)
	}
	reused := len(chunkBatch) - len(missTexts)
	if len(missTexts) == 0 {
		return vectors, reused, nil
	}

	result, err := service.embedder.EmbedBatch(ctx, missTexts)
	if err != nil {
		// EmbedBatch already returns a typed adapterr error; %w keeps that class
		// visible to errors.As so the index and search paths classify an embedding
		// failure the same way.
		slog.ErrorContext(ctx, "embed batch failed", "err", err)
		return nil, 0, fmt.Errorf("embed chunk batch: %w", err)
	}
	if len(result.Vectors) != len(missTexts) {
		slog.ErrorContext(ctx, "embedding batch returned unexpected vector count", "want", len(missTexts), "got", len(result.Vectors), "err", errors.New("vector count mismatch"))
		return nil, 0, fmt.Errorf("embedding batch returned %d vectors for %d chunks", len(result.Vectors), len(missTexts))
	}
	for position, vectorIndex := range missIndexes {
		// A skipped input carries a nil vector; the caller re-splits and retries that
		// chunk through EmbedChunksSplittingOversize instead of dropping it, so an
		// oversized input is divided and indexed rather than lost.
		vectors[vectorIndex] = result.Vectors[position]
	}
	return vectors, reused, nil
}

// collectOversizedChunks returns the chunks whose vector is nil, meaning the
// embedding endpoint rejected the input as individually un-embeddable. The caller
// feeds these back through the splitter and retries them instead of dropping
// them, so an oversized input is divided and indexed.
func collectOversizedChunks(chunks []model.StoredChunk, vectors [][]float32) []model.StoredChunk {
	oversized := make([]model.StoredChunk, 0)
	for index, vector := range vectors {
		if vector == nil {
			oversized = append(oversized, chunks[index])
		}
	}
	return oversized
}

// filterEmbeddedChunks keeps only the chunks whose vector is non-nil, pairing
// each kept chunk with its vector in the original order. A nil vector marks an
// input the embedding endpoint skipped as un-embeddable, which is dropped here so
// it is never inserted.
func filterEmbeddedChunks(chunks []model.StoredChunk, vectors [][]float32) ([]model.StoredChunk, [][]float32) {
	keptChunks := make([]model.StoredChunk, 0, len(chunks))
	keptVectors := make([][]float32, 0, len(chunks))
	for index, vector := range vectors {
		if vector == nil {
			continue
		}
		keptChunks = append(keptChunks, chunks[index])
		keptVectors = append(keptVectors, vector)
	}
	return keptChunks, keptVectors
}

// stagingCollectionName derives the transient rebuild collection name, kept
// within the Milvus name-length cap.
func stagingCollectionName(collectionName string) string {
	maxBase := maxCollectionNameLength - len(stagingCollectionSuffix)
	if len(collectionName) > maxBase {
		collectionName = collectionName[:maxBase]
	}
	return collectionName + stagingCollectionSuffix
}
