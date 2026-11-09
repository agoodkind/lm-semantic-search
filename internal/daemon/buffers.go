package daemon

import (
	"sort"
	"time"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/model"
)

func (syncer *BackgroundSync) convergeProgressReporter(jobID string) func(ConvergeOutcome) {
	const percentScale = 100.0

	lastReportedPaths := int32(-1)

	return func(progress ConvergeOutcome) {
		if progress.PathsProcessed == lastReportedPaths {
			syncer.manager.updateDetachedJobHeartbeat(jobID)

			return
		}

		percent := percentScale
		if progress.PathsGiven > 0 {
			percent = float64(progress.PathsProcessed) / float64(progress.PathsGiven) * percentScale
		}

		syncer.manager.updateDetachedJobProgress(
			jobID,
			indexer.Progress{
				Phase:                  "Converging changed paths...",
				OverallPercent:         percent,
				FilesTotal:             progress.PathsGiven,
				FilesProcessed:         progress.PathsProcessed,
				FilesEmbedded:          0,
				FilesSkippedOversize:   0,
				FilesSkippedUnreadable: 0,
				FilesPending:           0,
				ChunksProcessed:        0,
				ChunksReused:           0,
				ChunksEmbedded:         0,
				ChunksGenerated:        0,
				ChunksDropped:          0,
				ReuseVectorsLoaded:     0,
			},
			"path",
		)
		lastReportedPaths = progress.PathsProcessed
	}
}

func (syncer *BackgroundSync) deferWatcherPaths(codebaseID string, relativePaths []string) {
	if len(relativePaths) == 0 {
		return
	}

	syncer.deferredWatcherMu.Lock()
	defer syncer.deferredWatcherMu.Unlock()
	buffered, found := syncer.deferredWatcherPaths[codebaseID]
	if !found {
		buffered = make(map[string]struct{}, len(relativePaths))
		syncer.deferredWatcherPaths[codebaseID] = buffered
	}

	for _, relativePath := range relativePaths {
		buffered[relativePath] = struct{}{}
	}
}

// deferredPathCounts reports how many changed paths each codebase has buffered
// while its first build runs. That work is real and waiting: it converges the
// moment the build promotes, so a status read that omitted it would report a
// quiet system while edits piled up. The returned map is a copy.
func (syncer *BackgroundSync) deferredPathCounts() map[string]int {
	syncer.deferredWatcherMu.Lock()
	defer syncer.deferredWatcherMu.Unlock()

	counts := make(map[string]int, len(syncer.deferredWatcherPaths))
	for codebaseID, paths := range syncer.deferredWatcherPaths {
		counts[codebaseID] = len(paths)
	}

	return counts
}

func (syncer *BackgroundSync) takeDeferredWatcherPaths(codebaseID string) []string {
	syncer.deferredWatcherMu.Lock()
	defer syncer.deferredWatcherMu.Unlock()
	buffered := syncer.deferredWatcherPaths[codebaseID]
	if len(buffered) == 0 {
		return nil
	}

	relativePaths := make([]string, 0, len(buffered))
	for relativePath := range buffered {
		relativePaths = append(relativePaths, relativePath)
	}

	sort.Strings(relativePaths)
	delete(syncer.deferredWatcherPaths, codebaseID)

	return relativePaths
}

func newConvergeJob(codebase model.Codebase, relativePaths []string, now time.Time) model.Job {
	job := newQueuedJob(
		codebase.ID,
		codebase.CanonicalPath,
		codebase.CanonicalPath,
		model.ClientInfo{Name: "daemon-watcher", PID: 0},
		"converge",
		false,
		codebase.EffectiveConfig,
		emptyAdmissionBudget,
		now,
	)
	job.Progress.FilesTotal = safeInt32(len(relativePaths))
	job.Progress.Unit = "path"

	return job
}

func (syncer *BackgroundSync) watcherCodebase(codebaseID string) (model.Codebase, bool) {
	syncer.manager.mu.Lock()
	defer syncer.manager.mu.Unlock()
	codebase, found := syncer.manager.codebases[codebaseID]

	return codebase, found
}

func (syncer *BackgroundSync) hasActiveJob(codebase model.Codebase) bool {
	syncer.manager.mu.Lock()
	defer syncer.manager.mu.Unlock()
	current, found := syncer.manager.codebases[codebase.ID]
	if !found {
		return false
	}

	return syncer.manager.activeJobSnapshotLocked(current) != nil
}

// beginConverge claims the per-codebase converge slot, returning false when a
// converge for that codebase is already running.
func (syncer *BackgroundSync) beginConverge(codebaseID string) bool {
	syncer.convergeMu.Lock()
	defer syncer.convergeMu.Unlock()
	if _, admitted := syncer.converging[codebaseID]; admitted {
		return false
	}

	// A zero start time means admitted but not yet running: the converge still
	// has to win an index slot and the shared advisory lock. markConvergeRunning
	// stamps the real time once both waits are behind it, so a status read never
	// reports a queued converge as running work whose age keeps growing.
	syncer.converging[codebaseID] = time.Time{}

	return true
}

// markConvergeRunning records that an admitted converge has taken its index
// slot and the shared advisory lock, so it is doing work rather than waiting
// for capacity.
func (syncer *BackgroundSync) markConvergeRunning(codebaseID string) {
	syncer.convergeMu.Lock()
	defer syncer.convergeMu.Unlock()
	if _, admitted := syncer.converging[codebaseID]; !admitted {
		return
	}

	syncer.converging[codebaseID] = clock.Now()
}

// endConverge releases the per-codebase converge slot.
func (syncer *BackgroundSync) endConverge(codebaseID string) {
	syncer.convergeMu.Lock()
	defer syncer.convergeMu.Unlock()
	delete(syncer.converging, codebaseID)
}

// requeuePaths re-enqueues a deferred converge's paths so the change is picked
// up on the next debounce rather than dropped.
func (syncer *BackgroundSync) requeuePaths(codebaseID string, relativePaths []string) {
	for _, relativePath := range relativePaths {
		syncer.queue.Enqueue(codebaseID, relativePath)
	}
}
