package daemon

import (
	"context"
	"log/slog"
	"sync/atomic"

	"goodkind.io/lm-semantic-search/internal/gitworktree"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/view"
)

type worktreeReuseSelection struct {
	sourcePath       string
	sourceCodebaseID string
	sourceCollection string
	unavailable      view.WorktreeReuseReason
}

type jobSourceReuse struct {
	jobID      string
	sourcePath string
	collection string
	disabled   atomic.Bool
}

type contentReuseLookup int

const (
	contentReuseLookupStoreWide contentReuseLookup = iota
	contentReuseLookupListedCollections
)

type jobContentReusePlan struct {
	lookup contentReuseLookup
	source *jobSourceReuse
}

func (manager *Manager) selectWorktreeReuseSourceLocked(
	sourcePath string,
	indexConfig model.IndexConfig,
) worktreeReuseSelection {
	selection := worktreeReuseSelection{
		sourcePath:       sourcePath,
		sourceCodebaseID: "",
		sourceCollection: "",
		unavailable:      view.WorktreeReuseNoSourceRecorded,
	}
	if sourcePath == "" {
		return selection
	}
	source, found := manager.findCodebaseByExactRoot(sourcePath)
	if !found || source.Kind == model.CodebaseKindDocument {
		selection.unavailable = view.WorktreeReuseSourceNotRegistered
		return selection
	}
	selection.sourceCodebaseID = source.ID
	if source.CollectionName == "" || !ownsLiveCollection(source) {
		selection.unavailable = view.WorktreeReuseSourceNotIndexed
		return selection
	}
	if !reuseModelMatches(source.EffectiveConfig, indexConfig) {
		selection.unavailable = view.WorktreeReuseSourceEmbeddingConfigurationDiffers
		return selection
	}
	selection.sourceCollection = source.CollectionName
	selection.unavailable = view.WorktreeReuseAvailable
	return selection
}

func (manager *Manager) selectWorktreeReuseSource(codebase model.Codebase) worktreeReuseSelection {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	return manager.selectWorktreeReuseSourceLocked(codebase.WorktreeSourcePath, codebase.EffectiveConfig)
}

func (manager *Manager) worktreeReuseStatus(codebase model.Codebase, discovered bool) worktreeReuseSelection {
	if codebase.Kind == model.CodebaseKindDocument || (codebase.WorktreeSourcePath == "" && !discovered) {
		return worktreeReuseSelection{
			sourcePath:       "",
			sourceCodebaseID: "",
			sourceCollection: "",
			unavailable:      view.WorktreeReuseAvailable,
		}
	}
	return manager.selectWorktreeReuseSource(codebase)
}

func (manager *Manager) jobContentReusePlan(ctx context.Context, job model.Job) jobContentReusePlan {
	if _, inRepository := gitworktree.Resolve(job.CanonicalPath); !inRepository {
		return jobContentReusePlan{lookup: contentReuseLookupStoreWide, source: nil}
	}
	manager.mu.Lock()
	target := manager.codebases[job.CodebaseID]
	selection := manager.selectWorktreeReuseSourceLocked(target.WorktreeSourcePath, job.Config)
	manager.mu.Unlock()

	if selection.unavailable != view.WorktreeReuseAvailable {
		slog.InfoContext(ctx, "job reads no source collection",
			"job_id", job.ID,
			"codebase_id", job.CodebaseID,
			"source_path", selection.sourcePath,
			"reason", string(selection.unavailable),
		)
		return jobContentReusePlan{lookup: contentReuseLookupListedCollections, source: nil}
	}
	slog.InfoContext(ctx, "job reuses vectors from the recorded source",
		"job_id", job.ID,
		"codebase_id", job.CodebaseID,
		"source_path", selection.sourcePath,
		"source_codebase_id", selection.sourceCodebaseID,
		"source_collection", selection.sourceCollection,
	)
	source := &jobSourceReuse{
		jobID:      job.ID,
		sourcePath: selection.sourcePath,
		collection: selection.sourceCollection,
		disabled:   atomic.Bool{},
	}
	return jobContentReusePlan{lookup: contentReuseLookupListedCollections, source: source}
}
