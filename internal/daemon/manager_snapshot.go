package daemon

import (
	"sort"
	"time"

	"goodkind.io/lm-semantic-search/internal/model"
)

// StatusSnapshot is everything a status read reports about work and codebases,
// taken so no caller can observe two of these facts from different instants.
//
// A depth-1 pending slot becomes a job, and a running job changes a codebase's
// display status, so reading those through separate accessors lets a transition
// land between two reads: the work shows twice, or in the worse direction, it
// shows in neither and a caller sees nothing outstanding while the daemon is
// busy. One snapshot removes both outcomes rather than choosing between them.
type StatusSnapshot struct {
	// StartedAt is when this daemon process built its manager.
	StartedAt time.Time
	// IndexSlotsInUse and IndexSlotsTotal explain a job that is queued behind
	// capacity rather than behind a dependency.
	IndexSlotsInUse int
	IndexSlotsTotal int
	// Health is the shared-dependency record, resolved the same way every other
	// surface resolves it. It is read once and carried here, because reading it
	// is not pure: it clears a boot-time store banner once the client has
	// reconnected, so two reads in one reply would describe two instants.
	Health dependencyHealth
	// ActiveJobs holds the non-terminal jobs only. Job history is deliberately
	// absent: a status reply stays a few kilobytes rather than growing with it.
	ActiveJobs []model.Job
	// Pending holds the coalesced requests that hold no job record yet.
	Pending []PendingWork
	// Codebases carries each tracked codebase with its resolved display status,
	// so a status count and an activity row cannot disagree about one codebase.
	Codebases []CodebaseView
	// Watcher holds the file-change converges, which register no job. The syncer
	// owns them under its own lock and none of them ever becomes a job, so they
	// need no atomicity with the job store.
	Watcher []WatcherActivity
}

// StatusSnapshot reads every fact a status reply needs. The watcher activity is
// gathered first, under the syncer's own locks, so the manager lock is never
// held across a call into the syncer. Everything the manager owns is then read
// under one hold of that lock.
func (manager *Manager) StatusSnapshot() StatusSnapshot {
	watcher := manager.WatcherActivity()

	manager.mu.Lock()
	defer manager.mu.Unlock()

	// Health resolves first because resolving it can clear a stale store banner,
	// and the codebase views fold the resulting mode into every display status.
	// Reading it after them would report a recovered store beside codebases
	// still marked as waiting on it.
	health := manager.dependencyHealthLocked()

	activeJobs := make([]model.Job, 0, len(manager.jobs))
	for _, job := range manager.jobs {
		if isTerminalJobState(job.State) {
			continue
		}
		activeJobs = append(activeJobs, job)
	}
	sort.Slice(activeJobs, func(first int, second int) bool {
		return activeJobs[first].StartedAt.After(activeJobs[second].StartedAt)
	})

	return StatusSnapshot{
		StartedAt:       manager.startedAt,
		IndexSlotsInUse: len(manager.indexSlots),
		IndexSlotsTotal: cap(manager.indexSlots),
		Health:          health,
		ActiveJobs:      activeJobs,
		Pending:         manager.pendingWorkLocked(),
		Codebases:       manager.codebaseViewsLocked(),
		Watcher:         watcher,
	}
}

// CanonicalPaths indexes the snapshot's codebases by id, so a watcher or
// pending row can name its path from the same instant the rest of the reply
// describes.
func (snapshot StatusSnapshot) CanonicalPaths() map[string]string {
	paths := make(map[string]string, len(snapshot.Codebases))
	for _, view := range snapshot.Codebases {
		paths[view.Codebase.ID] = view.Codebase.CanonicalPath
	}
	return paths
}
