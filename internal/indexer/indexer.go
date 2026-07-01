// Package indexer walks codebases and produces file and chunk counts.
package indexer

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"

	"goodkind.io/gksyntax/chunk"
	"goodkind.io/lm-semantic-search/internal/discovery"
	"goodkind.io/lm-semantic-search/internal/indexability"
	"goodkind.io/lm-semantic-search/internal/model"
)

// Runner executes the local discovery and splitting pipeline for one codebase.
// The size and content gates are not held here; every gate decision routes
// through the indexability resolver the caller threads in, so the resolver is
// the single owner of the file-set verdict.
type Runner struct {
	dispatcher *chunk.Dispatcher
}

// Result captures file and chunk totals for one indexing pass.
//
// SkippedFiles names files the indexer refused to embed. A file is skipped
// when its bytes are not valid UTF-8 (Milvus requires every VarChar field
// to be valid UTF-8 on the gRPC wire) or when its size exceeds the
// per-file cap.
type Result struct {
	IndexedFiles      int32
	TotalChunks       int32
	Chunks            []model.StoredChunk
	FileHashes        map[string]string
	SkippedFiles      []string
	SkippedOversize   int32
	SkippedUnreadable int32
	// SkippedPending counts changed items whose content was not delivered this
	// pass (the conversation-ingest undelivered case). They are transient, not
	// errors, and are re-requested on the next sync.
	SkippedPending int32
}

// SkipReason names why the indexer declined to embed a changed file. The empty
// value means the file was not skipped.
type SkipReason string

const (
	// SkipNone marks a file that was embedded or removed, not skipped.
	SkipNone SkipReason = ""
	// SkipOversize marks a file past the per-file size cap.
	SkipOversize SkipReason = "oversize"
	// SkipUnreadable marks a file whose bytes are not valid UTF-8.
	SkipUnreadable SkipReason = "unreadable"
	// SkipPending marks a changed item whose content was not delivered this pass,
	// so it cannot be embedded yet and will be re-requested on the next sync. It
	// is the conversation-ingest case where clyde listed a conversation as changed
	// but has not sent its documents. It is transient, not an error.
	SkipPending SkipReason = "pending"
)

// Progress describes one visible indexing progress update.
type Progress struct {
	Phase                  string
	OverallPercent         float64
	FilesTotal             int32
	FilesProcessed         int32
	FilesEmbedded          int32
	FilesSkippedOversize   int32
	FilesSkippedUnreadable int32
	// FilesPending counts changed items whose content was not delivered this pass
	// (the conversation-ingest undelivered case). Transient, re-requested next sync.
	FilesPending int32
	// ChunksProcessed counts chunks handled by this run. ChunksEmbedded counts
	// chunks sent to the embedder, ChunksReused counts chunks served from stored
	// vectors, and ChunksGenerated is the legacy alias for ChunksEmbedded.
	ChunksProcessed    int32
	ChunksReused       int32
	ChunksEmbedded     int32
	ChunksGenerated    int32
	ReuseVectorsLoaded int32
}

func newEmbeddedProgress(
	phase string,
	overallPercent float64,
	filesTotal int32,
	filesProcessed int32,
	filesEmbedded int32,
	filesSkippedOversize int32,
	filesSkippedUnreadable int32,
	chunksProcessed int32,
	chunksEmbedded int32,
) Progress {
	return Progress{
		Phase:                  phase,
		OverallPercent:         overallPercent,
		FilesTotal:             filesTotal,
		FilesProcessed:         filesProcessed,
		FilesEmbedded:          filesEmbedded,
		FilesSkippedOversize:   filesSkippedOversize,
		FilesSkippedUnreadable: filesSkippedUnreadable,
		FilesPending:           0,
		ChunksProcessed:        chunksProcessed,
		ChunksReused:           0,
		ChunksEmbedded:         chunksEmbedded,
		ChunksGenerated:        chunksEmbedded,
		ReuseVectorsLoaded:     0,
	}
}

// NewRunner constructs the local indexing runner.
func NewRunner() *Runner {
	return &Runner{
		dispatcher: chunk.NewDispatcher(),
	}
}

// processedFile is the per-file output of one splitter pass. Skipped=true
// means the file's bytes are not valid UTF-8; Chunks and FileHash are then
// empty and callers add the path to Result.SkippedFiles. Removed=true means
// the file was absent on disk when the task ran, so the converge operation
// for this path is a removal: callers delete its rows and drop it from the
// snapshot rather than treating the absence as an error.
type processedFile struct {
	Chunks     []model.StoredChunk
	FileHash   string
	Skipped    bool
	SkipReason SkipReason
	Removed    bool
}

// OneFileResult mirrors the per-file accumulator output for callers that
// drive their own iteration (the daemon's per-file delta loop).
type OneFileResult = processedFile

// IndexOne reads, gates, and splits a single file. The daemon's per-file
// delta loop calls this so the merkle snapshot can be flushed after each
// successful semantic.Reindex call. The resolver and codebaseID let the stat
// and content gates route through the shared indexability decider.
func (runner *Runner) IndexOne(ctx context.Context, resolver *indexability.Resolver, codebaseID string, root string, relativePath string, cfg model.IndexConfig) (OneFileResult, error) {
	fullPath := filepath.Join(root, relativePath)
	statResult, keep, err := runner.statEligibility(ctx, resolver, codebaseID, root, relativePath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			slog.DebugContext(ctx, "source file absent; converging to removal", "path", fullPath)
			return OneFileResult{Chunks: nil, FileHash: "", Skipped: false, SkipReason: SkipNone, Removed: true}, nil
		}
		return OneFileResult{Chunks: nil, FileHash: "", Skipped: false, SkipReason: SkipNone, Removed: false}, err
	}
	if !keep {
		return statResult, nil
	}
	data, err := os.ReadFile(fullPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			slog.DebugContext(ctx, "source file absent; converging to removal", "path", fullPath)
			return OneFileResult{Chunks: nil, FileHash: "", Skipped: false, SkipReason: SkipNone, Removed: true}, nil
		}
		if readErrorMeansRemoved(fullPath) {
			slog.DebugContext(ctx, "source path is no longer a regular file; converging to removal", "path", fullPath)
			return OneFileResult{Chunks: nil, FileHash: "", Skipped: false, SkipReason: SkipNone, Removed: true}, nil
		}
		slog.ErrorContext(ctx, "read source file failed", "path", fullPath, "err", err)
		return OneFileResult{Chunks: nil, FileHash: "", Skipped: false, SkipReason: SkipNone, Removed: false}, fmt.Errorf("read source file %s: %w", fullPath, err)
	}
	return runner.processFile(ctx, resolver, fullPath, relativePath, data, cfg.SplitterType)
}

// statEligibility routes the pre-read file-set verdict through the resolver's
// Decide. Discovery and the converge gate already applied scope and ignore, so
// a path reaches here only for its size and not-regular verdict, but routing the
// whole Decision keeps the indexer from owning a second copy of any gate. An
// oversize file is a recorded skip; a not-regular, out-of-scope, or ignored
// path converges to a removal.
func (runner *Runner) statEligibility(ctx context.Context, resolver *indexability.Resolver, codebaseID string, root string, relativePath string) (processedFile, bool, error) {
	fullPath := filepath.Join(root, relativePath)
	info, err := os.Stat(fullPath)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			slog.ErrorContext(ctx, "stat source file failed", "path", fullPath, "err", err)
		}
		return processedFile{Chunks: nil, FileHash: "", Skipped: false, SkipReason: SkipNone, Removed: false}, false, fmt.Errorf("stat source file %s: %w", fullPath, err)
	}
	decision := resolver.Decide(ctx, codebaseID, root, filepath.ToSlash(relativePath), info)
	if decision.Indexed {
		return processedFile{Chunks: nil, FileHash: "", Skipped: false, SkipReason: SkipNone, Removed: false}, true, nil
	}
	switch decision.Reason {
	case indexability.ReasonOversize:
		slog.WarnContext(ctx, "indexer.skipped_oversize", "path", relativePath, "bytes", info.Size())
		return processedFile{Chunks: nil, FileHash: "", Skipped: true, SkipReason: SkipOversize, Removed: false}, false, nil
	case indexability.ReasonNotRegular, indexability.ReasonOutOfScope, indexability.ReasonIgnored, indexability.ReasonSubmodule:
		slog.DebugContext(ctx, "source path is not indexable; converging to removal", "path", fullPath, "reason", string(decision.Reason))
		return processedFile{Chunks: nil, FileHash: "", Skipped: false, SkipReason: SkipNone, Removed: true}, false, nil
	case indexability.Keep, indexability.ReasonNonUTF8:
		return processedFile{Chunks: nil, FileHash: "", Skipped: false, SkipReason: SkipNone, Removed: false}, false, fmt.Errorf("unexpected stat-stage indexability reason %q for %s", decision.Reason, fullPath)
	default:
		return processedFile{Chunks: nil, FileHash: "", Skipped: false, SkipReason: SkipNone, Removed: false}, false, fmt.Errorf("unknown indexability reason %q for %s", decision.Reason, fullPath)
	}
}

func readErrorMeansRemoved(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		// The file can vanish between the failed read and this stat. A gone file
		// converges to removal; any other stat error is treated as a real read
		// failure to surface.
		return errors.Is(err, os.ErrNotExist)
	}
	return !info.Mode().IsRegular()
}

// processFile splits one file into chunks. When the file's bytes are not
// valid UTF-8 the splitter is skipped and Skipped=true is returned. Milvus
// rejects non-UTF-8 VarChar payloads at the gRPC marshal boundary, so the
// skip happens at the indexer boundary.
func (runner *Runner) processFile(ctx context.Context, resolver *indexability.Resolver, fullPath string, relativePath string, data []byte, splitterType string) (processedFile, error) {
	if !resolver.DecideContent(data).Indexed {
		slog.WarnContext(ctx, "indexer.skipped_invalid_utf8", "path", relativePath, "bytes", len(data))
		return processedFile{Chunks: nil, FileHash: "", Skipped: true, SkipReason: SkipUnreadable, Removed: false}, nil
	}
	splitResult, err := runner.dispatcher.SplitFileWithType(ctx, fullPath, data, splitterType)
	if err != nil {
		return processedFile{}, fmt.Errorf("split source file %s: %w", fullPath, err)
	}
	chunks := make([]model.StoredChunk, 0, len(splitResult.Chunks))
	for _, splitChunk := range splitResult.Chunks {
		chunks = append(chunks, model.StoredChunk{
			Content:              splitChunk.Content,
			RelativePath:         relativePath,
			StartLine:            safeInt32(splitChunk.StartLine),
			EndLine:              safeInt32(splitChunk.EndLine),
			Language:             splitChunk.Language,
			FileExtension:        filepath.Ext(relativePath),
			ConversationID:       "",
			ParentConversationID: "",
			MessageIndex:         0,
			Role:                 "",
			TimestampUnix:        0,
			WorkspaceRoot:        "",
			Archived:             false,
			Score:                0,
		})
	}
	return processedFile{Chunks: chunks, FileHash: digestFileBytes(data), Skipped: false, SkipReason: SkipNone, Removed: false}, nil
}

// indexAccumulator collects per-file output across one indexing pass.
type indexAccumulator struct {
	totalChunks       int32
	indexedCount      int32
	storedChunks      []model.StoredChunk
	fileHashes        map[string]string
	skippedFiles      []string
	skippedOversize   int32
	skippedUnreadable int32
}

// ingestFile reads one file, routes the size and UTF-8 gates through the
// resolver, and routes the result into the accumulator. It returns false when
// the file was skipped.
func (runner *Runner) ingestFile(ctx context.Context, resolver *indexability.Resolver, codebaseID string, root string, relativePath string, splitterType string, accumulator *indexAccumulator) error {
	fullPath := filepath.Join(root, relativePath)
	statResult, keep, err := runner.statEligibility(ctx, resolver, codebaseID, root, relativePath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			slog.DebugContext(ctx, "source file absent during walk; excluding from rebuild", "path", fullPath)
			return nil
		}
		return err
	}
	if !keep {
		accumulator.addSkipped(relativePath, statResult)
		return nil
	}
	data, err := os.ReadFile(fullPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			slog.DebugContext(ctx, "source file absent during walk; excluding from rebuild", "path", fullPath)
			return nil
		}
		if readErrorMeansRemoved(fullPath) {
			slog.DebugContext(ctx, "source path is no longer a regular file during walk; excluding from rebuild", "path", fullPath)
			return nil
		}
		slog.ErrorContext(ctx, "read source file failed", "path", fullPath, "err", err)
		return fmt.Errorf("read source file %s: %w", fullPath, err)
	}
	processed, err := runner.processFile(ctx, resolver, fullPath, relativePath, data, splitterType)
	if err != nil {
		slog.ErrorContext(ctx, "split source file failed", "path", fullPath, "err", err)
		return err
	}
	if processed.Skipped {
		accumulator.addSkipped(relativePath, processed)
		return nil
	}
	accumulator.totalChunks += safeInt32(len(processed.Chunks))
	accumulator.indexedCount++
	accumulator.fileHashes[relativePath] = processed.FileHash
	accumulator.storedChunks = append(accumulator.storedChunks, processed.Chunks...)
	return nil
}

func (accumulator *indexAccumulator) addSkipped(relativePath string, processed processedFile) {
	if !processed.Skipped {
		return
	}
	accumulator.skippedFiles = append(accumulator.skippedFiles, relativePath)
	if processed.SkipReason == SkipOversize {
		accumulator.skippedOversize++
		return
	}
	accumulator.skippedUnreadable++
}

// Index walks the codebase and splits files into chunks. The resolver and
// codebaseID are threaded into discovery so the walk routes its scope and ignore
// decisions through the daemon's one shared indexability resolver.
func (runner *Runner) Index(ctx context.Context, resolver *indexability.Resolver, codebaseID string, root string, indexConfig model.IndexConfig, progress func(Progress)) (Result, error) {
	if progress != nil {
		progress(newEmbeddedProgress("Preparing and scanning files...", 0, 0, 0, 0, 0, 0, 0, 0))
	}

	discoveryResult, err := discovery.Discover(ctx, resolver, codebaseID, root)
	if err != nil {
		slog.ErrorContext(ctx, "discover source files failed", "root", root, "err", err)
		return Result{}, fmt.Errorf("discover source files under %s: %w", root, err)
	}

	totalFiles := safeInt32(len(discoveryResult.Files))
	if progress != nil {
		progress(newEmbeddedProgress("Processing files and generating embeddings...", 10, totalFiles, 0, 0, 0, 0, 0, 0))
	}

	accumulator := &indexAccumulator{
		totalChunks:       0,
		indexedCount:      0,
		storedChunks:      make([]model.StoredChunk, 0),
		fileHashes:        make(map[string]string, len(discoveryResult.Files)),
		skippedFiles:      []string{},
		skippedOversize:   0,
		skippedUnreadable: 0,
	}
	for index, path := range discoveryResult.Files {
		if err := ctx.Err(); err != nil {
			slog.ErrorContext(ctx, "indexing cancelled before file read", "path", path, "err", err)
			return Result{}, fmt.Errorf("indexing cancelled before file read %s: %w", path, err)
		}
		relativePath, err := filepath.Rel(root, path)
		if err != nil {
			slog.ErrorContext(ctx, "compute relative chunk path failed", "root", root, "path", path, "err", err)
			return Result{}, fmt.Errorf("compute relative chunk path for %s: %w", path, err)
		}
		if err := runner.ingestFile(ctx, resolver, codebaseID, root, relativePath, indexConfig.SplitterType, accumulator); err != nil {
			return Result{}, err
		}
		if progress != nil {
			progress(newEmbeddedProgress(
				"Processing files and generating embeddings...",
				calculateOverallPercent(index+1, len(discoveryResult.Files)),
				totalFiles,
				safeInt32(index+1),
				safeInt32(index+1),
				accumulator.skippedOversize,
				accumulator.skippedUnreadable,
				accumulator.totalChunks,
				accumulator.totalChunks,
			))
		}
	}

	if progress != nil {
		progress(newEmbeddedProgress("completed", 100, totalFiles, totalFiles, totalFiles, accumulator.skippedOversize, accumulator.skippedUnreadable, accumulator.totalChunks, accumulator.totalChunks))
	}

	return Result{
		IndexedFiles:      accumulator.indexedCount,
		TotalChunks:       accumulator.totalChunks,
		Chunks:            accumulator.storedChunks,
		FileHashes:        accumulator.fileHashes,
		SkippedFiles:      accumulator.skippedFiles,
		SkippedOversize:   accumulator.skippedOversize,
		SkippedUnreadable: accumulator.skippedUnreadable,
		SkippedPending:    0,
	}, nil
}

// IndexFiles processes the explicit relative-path allowlist instead of
// re-walking the codebase. Use it for delta reindex passes where the caller
// has already computed the changed-file set (via merkle.DiffSnapshots).
//
// FileHashes in the result covers only the supplied files. Callers should
// merge this map into the previous snapshot before persisting. The resolver and
// codebaseID route the per-file gates through the shared indexability decider.
func (runner *Runner) IndexFiles(ctx context.Context, resolver *indexability.Resolver, codebaseID string, root string, relativePaths []string, indexConfig model.IndexConfig, progress func(Progress)) (Result, error) {
	totalFiles := safeInt32(len(relativePaths))
	if progress != nil {
		progress(newEmbeddedProgress("Processing changed files...", 10, totalFiles, 0, 0, 0, 0, 0, 0))
	}

	accumulator := &indexAccumulator{
		totalChunks:       0,
		indexedCount:      0,
		storedChunks:      make([]model.StoredChunk, 0),
		fileHashes:        make(map[string]string, len(relativePaths)),
		skippedFiles:      []string{},
		skippedOversize:   0,
		skippedUnreadable: 0,
	}
	for index, relativePath := range relativePaths {
		if err := ctx.Err(); err != nil {
			slog.ErrorContext(ctx, "delta indexing cancelled before file read", "path", relativePath, "err", err)
			return Result{}, fmt.Errorf("delta indexing cancelled before file read %s: %w", relativePath, err)
		}
		if err := runner.ingestFile(ctx, resolver, codebaseID, root, relativePath, indexConfig.SplitterType, accumulator); err != nil {
			return Result{}, err
		}
		if progress != nil {
			progress(newEmbeddedProgress(
				"Processing changed files...",
				calculateOverallPercent(index+1, len(relativePaths)),
				totalFiles,
				safeInt32(index+1),
				safeInt32(index+1),
				accumulator.skippedOversize,
				accumulator.skippedUnreadable,
				accumulator.totalChunks,
				accumulator.totalChunks,
			))
		}
	}

	if progress != nil {
		progress(newEmbeddedProgress("completed", 100, totalFiles, totalFiles, totalFiles, accumulator.skippedOversize, accumulator.skippedUnreadable, accumulator.totalChunks, accumulator.totalChunks))
	}

	return Result{
		IndexedFiles:      accumulator.indexedCount,
		TotalChunks:       accumulator.totalChunks,
		Chunks:            accumulator.storedChunks,
		FileHashes:        accumulator.fileHashes,
		SkippedFiles:      accumulator.skippedFiles,
		SkippedOversize:   accumulator.skippedOversize,
		SkippedUnreadable: accumulator.skippedUnreadable,
		SkippedPending:    0,
	}, nil
}

func digestFileBytes(data []byte) string {
	hashBytes := sha256.Sum256(data)
	return hex.EncodeToString(hashBytes[:])
}

func calculateOverallPercent(processedFiles int, totalFiles int) float64 {
	if totalFiles <= 0 {
		return 100
	}
	return 10 + (float64(processedFiles)/float64(totalFiles))*90
}

func safeInt32(value int) int32 {
	if value > math.MaxInt32 {
		return math.MaxInt32
	}
	if value < math.MinInt32 {
		return math.MinInt32
	}
	return int32(value)
}
