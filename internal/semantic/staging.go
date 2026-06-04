package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/adapterr"
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
// removedOrModifiedRelativePaths are deleted from staging first when staging
// already exists, which keeps a re-embedded file idempotent: if a crash lands
// between a file's insert and its checkpoint, the resumed run re-embeds that
// one file and its prior staging rows are removed before the fresh rows land.
// A nil chunk slice with nothing to remove is a no-op.
func (service *Service) StageReindex(ctx context.Context, codebasePath string, chunks []model.StoredChunk, removedOrModifiedRelativePaths []string, progress func(Progress), reuse map[string][]float32) (err error) {
	ctx, done := spans.Open(ctx, "semantic.stageReindex")
	defer done(&err)

	if !service.Available() {
		return nil
	}

	stagingName := stagingCollectionName(service.CollectionName(codebasePath))
	hasStaging, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(stagingName))
	if err != nil {
		slog.ErrorContext(ctx, "check staging collection failed", "collection", stagingName, "err", err)
		return fmt.Errorf("check staging collection %s: %w", stagingName, err)
	}

	if hasStaging && len(removedOrModifiedRelativePaths) > 0 {
		if err := service.deleteByRelativePaths(ctx, stagingName, removedOrModifiedRelativePaths); err != nil {
			return err
		}
	}
	if len(chunks) == 0 {
		return nil
	}
	chunks = service.guardrailExpand(ctx, codebasePath, chunks, "stage")
	return service.insertChunksBatched(ctx, stagingName, chunks, hasStaging, "Generating embeddings and writing to Milvus...", progress, reuse)
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
		slog.ErrorContext(ctx, "check staging collection before promote failed", "collection", stagingName, "err", err)
		return fmt.Errorf("check staging collection %s: %w", stagingName, err)
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

// insertChunksBatched embeds chunks in EmbeddingBatchSize batches and inserts
// them into collectionName. When collectionReady is false the collection is
// created on the first batch using the dimension of the first returned vector,
// which is how both the staging build and an empty live collection learn their
// dimension without an up-front guess. The caller guarantees chunks is
// non-empty and already guardrail-expanded.
func (service *Service) insertChunksBatched(ctx context.Context, collectionName string, chunks []model.StoredChunk, collectionReady bool, phase string, progress func(Progress), reuse map[string][]float32) error {
	batchSize := service.cfg.EmbeddingBatchSize
	if batchSize <= 0 {
		batchSize = 32
	}

	totalBatches := (len(chunks) + batchSize - 1) / batchSize
	var writtenRows int32

	for batchIndex := range totalBatches {
		start := batchIndex * batchSize
		end := min(start+batchSize, len(chunks))

		chunkBatch := chunks[start:end]
		vectors, err := service.embedChunkBatch(ctx, chunkBatch, reuse)
		if err != nil {
			return err
		}

		if !collectionReady {
			dimension := len(vectors[0])
			if err := service.createCollection(ctx, collectionName, dimension); err != nil {
				return err
			}
			collectionReady = true
		}

		if err := service.insertBatch(ctx, collectionName, chunkBatch, vectors); err != nil {
			return err
		}

		writtenRows += safeInt32FromInt(len(chunkBatch))
		if progress != nil {
			progress(Progress{
				Phase:                     phase,
				OverallPercent:            90 + (float64(batchIndex+1)/float64(totalBatches))*10,
				EmbeddingBatchesTotal:     safeInt32FromInt(totalBatches),
				EmbeddingBatchesCompleted: safeInt32FromInt(batchIndex + 1),
				CollectionRowsWritten:     writtenRows,
			})
		}
	}
	return nil
}

// embedChunkBatch returns one dense vector per chunk in chunkBatch, in order.
// When reuse holds a vector for a chunk's content (keyed by content hash) that
// vector is taken directly and the embedder is never called for it; only the
// remaining misses are embedded in a single batch. A nil or empty reuse map
// makes this embed every chunk, which is the ordinary first-index behavior.
func (service *Service) embedChunkBatch(ctx context.Context, chunkBatch []model.StoredChunk, reuse map[string][]float32) ([][]float32, error) {
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
	if len(missTexts) == 0 {
		return vectors, nil
	}

	embedded, err := service.embedder.EmbedBatch(ctx, missTexts)
	if err != nil {
		slog.ErrorContext(ctx, "embed batch failed", "err", err)
		return nil, adapterr.NewEmbedderUnreachable(err)
	}
	if len(embedded) != len(missTexts) {
		slog.ErrorContext(ctx, "embedding batch returned unexpected vector count", "want", len(missTexts), "got", len(embedded), "err", errors.New("vector count mismatch"))
		return nil, fmt.Errorf("embedding batch returned %d vectors for %d chunks", len(embedded), len(missTexts))
	}
	for position, vectorIndex := range missIndexes {
		vectors[vectorIndex] = embedded[position]
	}
	return vectors, nil
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
