package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/metrics"
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
	hasStaging, err := service.hasCollection(ctx, stagingName, "check staging collection "+stagingName)
	if err != nil {
		return err
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
	hasStaging, err := service.hasCollection(ctx, stagingName, "check staging collection "+stagingName)
	if err != nil {
		return err
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
	hasStaging, err := service.hasCollection(ctx, stagingName, "check staging collection "+stagingName)
	if err != nil {
		return false, err
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

// NewChunkPacker returns a packer that groups chunks under a row count and an
// estimated token budget, substituting the built-in defaults for either value
// when it is not positive. Both storage backends embed through the same
// split-and-retry loop, and that loop packs every round, so both need a packer
// built the same way.
func NewChunkPacker(batchRows int, tokenBudget int) ChunkPackFunc {
	if batchRows <= 0 {
		batchRows = defaultEmbeddingBatchRows
	}
	if tokenBudget <= 0 {
		tokenBudget = defaultEmbeddingBatchTokenBudget
	}
	return func(chunks []model.StoredChunk) [][]model.StoredChunk {
		return packChunksByEstimatedTokens(chunks, batchRows, tokenBudget, nil)
	}
}

// packForEmbedding groups chunks into embedding requests under the configured
// row ceiling and estimated-token budget. reuse is the same map embedChunkBatch
// consults, so the packer charges the token budget only for the chunks that will
// actually be sent to the embedder.
func (service *Service) packForEmbedding(chunks []model.StoredChunk, reuse map[string][]float32) [][]model.StoredChunk {
	batchRows := service.cfg.EmbeddingBatchSize
	if batchRows <= 0 {
		batchRows = defaultEmbeddingBatchRows
	}
	tokenBudget := service.cfg.EmbeddingBatchTokenBudget
	if tokenBudget <= 0 {
		tokenBudget = defaultEmbeddingBatchTokenBudget
	}
	return packChunksByEstimatedTokens(chunks, batchRows, tokenBudget, reuse)
}

// insertChunksBatched packs embedding requests by row count and estimated
// tokens, then independently packs store inserts by estimated serialized bytes.
// When collectionReady is false the collection is created on the first batch
// using the dimension of the first returned vector, which is how both the
// staging build and an empty live collection learn their dimension without an
// up-front guess. The caller guarantees chunks is non-empty and already
// guardrail-expanded.
//
// The span bounds the whole embed-and-insert loop, so a reindex splits into its
// removal, guardrail, and write phases without any external measurement. The
// per-batch semantic.embedBatch and semantic.insertBatch spans nested inside it
// separate the embedder round trip from the Milvus write, and
// semantic.chunks_written closes the loop with the totals that make those
// durations a rate.
func (service *Service) insertChunksBatched(ctx context.Context, collectionName string, chunks []model.StoredChunk, collectionReady bool, phase string, progress func(Progress), reuse map[string][]float32, columnSet StoreColumnSet) (err error) {
	ctx, done := spans.Open(ctx, "semantic.insertChunksBatched")
	defer done(&err)

	embeddingPacks := service.packForEmbedding(chunks, reuse)
	totalEmbeddingBatches := len(embeddingPacks)
	totalInsertBatches := 0
	var writtenRows int32
	var reusedRows int32
	var embeddedRows int32
	var droppedInputs int32

	for embeddingBatchIndex, embeddingBatch := range embeddingPacks {
		vectors, reused, err := service.embedChunkBatch(ctx, embeddingBatch, reuse)
		if err != nil {
			return err
		}

		keptChunks, keptVectors := filterEmbeddedChunks(embeddingBatch, vectors)

		// A reused chunk never reaches the embedder, so it cannot be refused and
		// cannot enter this retry queue. The chunks-only constructor therefore
		// correctly charges every retry input against the token budget.
		oversized := collectOversizedChunks(embeddingBatch, vectors)
		if len(oversized) > 0 {
			retryChunks, retryVectors, dropped, retryErr := EmbedChunksSplittingOversize(
				ctx,
				oversized,
				config.ActiveEmbedTokenLimit(service.cfg),
				NewChunkPacker(service.cfg.EmbeddingBatchSize, service.cfg.EmbeddingBatchTokenBudget),
				service.embedder.EmbedBatch,
			)
			if retryErr != nil {
				return retryErr
			}
			keptChunks = append(keptChunks, retryChunks...)
			keptVectors = append(keptVectors, retryVectors...)
			droppedInputs += safeInt32FromInt(dropped)
		}
		if len(keptChunks) > 0 {
			dimension := len(keptVectors[0])
			if !collectionReady {
				if err := service.createCollection(ctx, collectionName, dimension); err != nil {
					return err
				}
				collectionReady = true
			}

			insertPacks := packChunksByEstimatedInsertBytes(
				keptChunks,
				dimension,
				insertBatchEstimatedByteBudget,
				columnSet,
			)
			vectorStart := 0
			for _, insertBatch := range insertPacks {
				vectorEnd := vectorStart + len(insertBatch)
				if err := service.insertBatch(
					ctx,
					collectionName,
					insertBatch,
					keptVectors[vectorStart:vectorEnd],
					columnSet,
				); err != nil {
					return err
				}
				vectorStart = vectorEnd
				totalInsertBatches++
				writtenRows += safeInt32FromInt(len(insertBatch))
			}

			reusedRows += safeInt32FromInt(reused)
			embeddedRows += safeInt32FromInt(len(keptChunks) - reused)
		}
		if progress != nil {
			progress(Progress{
				Phase:                     phase,
				OverallPercent:            90 + (float64(embeddingBatchIndex+1)/float64(totalEmbeddingBatches))*10,
				EmbeddingBatchesTotal:     safeInt32FromInt(totalEmbeddingBatches),
				EmbeddingBatchesCompleted: safeInt32FromInt(embeddingBatchIndex + 1),
				CollectionRowsWritten:     writtenRows,
				ChunksProcessed:           writtenRows,
				ChunksReused:              reusedRows,
				ChunksEmbedded:            embeddedRows,
				ChunksDropped:             droppedInputs,
			})
		}
	}
	if droppedInputs > 0 {
		slog.WarnContext(
			ctx,
			"semantic.embed_inputs_dropped_summary",
			"collection", collectionName,
			"dropped_inputs", droppedInputs,
		)
	}
	slog.InfoContext(ctx, "semantic.chunks_written", "collection", collectionName, "embedding_batches", totalEmbeddingBatches, "insert_batches", totalInsertBatches, "chunks", writtenRows, "embedded", embeddedRows, "reused", reusedRows)
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
	// Counted here rather than at the insert, because the hit is decided here
	// and a caller that never inserts still avoided the embed. It is the same
	// boundary metrics.EmbedBatchDone counts a batch at, so the two counters
	// together report what a batch cost and what it skipped.
	metrics.ChunksReused(reused)
	if len(missTexts) == 0 {
		return vectors, reused, nil
	}

	result, err := service.embedMissedTexts(ctx, missTexts, reused)
	if err != nil {
		return nil, 0, err
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

// embedMissedTexts sends one embedding request for the chunk contents no reuse
// vector covered.
//
// It carries its own span because embedding is the phase that dominates a
// reindex and nothing inside semantic.reindex timed it before: the round trip
// blocks on the embedding host's queue and its forward pass, so a slow reindex
// is attributable to the embedder or to Milvus only when the two are timed
// apart. The span opens only when there is something to embed, so an all-reuse
// batch stays silent instead of logging an empty phase.
// semantic.embed_batch_started carries the input and estimated-token counts
// under the same span id, which is what turns duration_ms into a rate.
func (service *Service) embedMissedTexts(ctx context.Context, missTexts []string, reusedCount int) (result embedding.BatchResult, err error) {
	ctx, done := spans.Open(ctx, "semantic.embedBatch")
	defer done(&err)
	slog.InfoContext(ctx, "semantic.embed_batch_started", "inputs", len(missTexts), "estimated_tokens", estimatedBatchTokenCount(missTexts), "reused", reusedCount)

	result, err = service.embedder.EmbedBatch(ctx, missTexts)
	if err != nil {
		// EmbedBatch already returns a typed adapterr error; %w keeps that class
		// visible to errors.As so the index and search paths classify an embedding
		// failure the same way.
		slog.ErrorContext(ctx, "embed batch failed", "err", err)
		return embedding.BatchResult{}, fmt.Errorf("embed chunk batch: %w", err)
	}
	if len(result.Vectors) != len(missTexts) {
		slog.ErrorContext(ctx, "embedding batch returned unexpected vector count", "want", len(missTexts), "got", len(result.Vectors), "err", errors.New("vector count mismatch"))
		return embedding.BatchResult{}, fmt.Errorf("embedding batch returned %d vectors for %d chunks", len(result.Vectors), len(missTexts))
	}
	return result, nil
}

// estimatedBatchTokenCount sums the packer's per-input token estimate across one
// embedding request, so the embed span reports the size of what it sent.
func estimatedBatchTokenCount(texts []string) int {
	total := 0
	for _, text := range texts {
		total += estimatedTokenCount(text)
	}
	return total
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
