package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"

	"goodkind.io/gklog/correlation"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/spans"
)

const (
	documentCanonicalPathPrefix = "chat:///"
	collectionChunkMaxBytes     = 60000
)

func resolveCollectionChunkBudget(chunkByteBudget []int) int {
	if len(chunkByteBudget) > 0 && chunkByteBudget[0] > 0 {
		return chunkByteBudget[0]
	}
	return collectionChunkMaxBytes
}

type collectionJobKind string

const (
	collectionJobKindUpsert collectionJobKind = "upsert"
	collectionJobKindDelete collectionJobKind = "delete"
)

type collectionJobPayload struct {
	Kind           collectionJobKind
	CollectionName string
	Manifest       map[string]string

	Rows     []collectionRow
	ItemID   string
	Absence  absencePolicy
	Backfill bool
	Force    bool
}

// syncCollectionManifest diffs a document collection's manifest against its
// stored checkpoint and returns the new and changed item ids, capped per ingest
// by capNeededItems with the collection's rotation cursor.
func (manager *Manager) syncCollectionManifest(ctx context.Context, codebase model.Codebase, manifest map[string]string) []string {
	configDigest := codebase.EffectiveConfig.IgnoreDigest
	seed := manager.loadLiveCheckpoint(ctx, codebase, configDigest).snapshot
	current := merkle.Snapshot{ConfigDigest: configDigest, Files: manifest, Inodes: nil}
	diff := merkle.DiffSnapshots(seed, current)

	manager.mu.Lock()
	cursor := manager.collectionSyncCursors[codebase.ID]
	needed, nextCursor := capNeededItems(diff.Added, diff.Modified, manager.config.MaxItemsPerIngest, cursor)
	if nextCursor != "" {
		manager.collectionSyncCursors[codebase.ID] = nextCursor
	}
	manager.mu.Unlock()
	return needed
}

// One shared cursor tracks pre-sort rotation order across modified overflow and
// added windows. A cursor past a list's end wraps to that list's start so
// rotation stays live without per-list cursors.
func capNeededItems(added []string, modified []string, limit int, cursor string) ([]string, string) {
	if limit <= 0 {
		needed := make([]string, 0, len(added)+len(modified))
		needed = append(needed, added...)
		needed = append(needed, modified...)
		sort.Strings(needed)
		return needed, cursor
	}

	capped := make([]string, 0, min(limit, len(added)+len(modified)))
	nextCursor := ""
	if len(modified) > limit {
		modifiedWindow := firstN(rotateAfter(modified, cursor), limit)
		capped = append(capped, modifiedWindow...)
		if len(modifiedWindow) > 0 {
			nextCursor = modifiedWindow[len(modifiedWindow)-1]
		}
		sort.Strings(capped)
		return capped, nextCursor
	}

	sortedModified := append([]string(nil), modified...)
	sort.Strings(sortedModified)
	capped = append(capped, sortedModified...)
	if len(sortedModified) > 0 {
		nextCursor = sortedModified[len(sortedModified)-1]
	}
	remainder := limit - len(capped)
	if remainder > 0 {
		addedWindow := firstN(rotateAfter(added, cursor), remainder)
		capped = append(capped, addedWindow...)
		if len(addedWindow) > 0 {
			nextCursor = addedWindow[len(addedWindow)-1]
		}
	}
	sort.Strings(capped)
	return capped, nextCursor
}

func rotateAfter(values []string, cursor string) []string {
	rotated := append([]string(nil), values...)
	sort.Strings(rotated)
	if len(rotated) == 0 || cursor == "" {
		return rotated
	}
	start := sort.SearchStrings(rotated, cursor)
	if start < len(rotated) && rotated[start] == cursor {
		start++
	}
	if start >= len(rotated) {
		return rotated
	}
	return append(append([]string(nil), rotated[start:]...), rotated[:start]...)
}

func firstN(values []string, limit int) []string {
	if limit >= len(values) {
		return values
	}
	return values[:limit]
}

type collectionUpsert struct {
	Manifest map[string]string

	Rows     []collectionRow
	Absence  absencePolicy
	Backfill bool
	Force    bool
}

func (manager *Manager) queueCollectionUpsert(ctx context.Context, codebase model.Codebase, client model.ClientInfo, upsert collectionUpsert) (model.Job, error) {
	payload := collectionJobPayload{
		Kind:           collectionJobKindUpsert,
		CollectionName: codebase.CollectionName,
		Manifest:       upsert.Manifest,

		Rows:     upsert.Rows,
		ItemID:   "",
		Absence:  upsert.Absence,
		Backfill: upsert.Backfill,
		Force:    upsert.Force,
	}
	return manager.queueCollectionJob(ctx, codebase, client, payload)
}

func (manager *Manager) queueCollectionJob(ctx context.Context, codebase model.Codebase, client model.ClientInfo, payload collectionJobPayload) (model.Job, error) {
	manager.policyMutationMutex.Lock()
	policyLocked := true
	defer func() {
		if policyLocked {
			manager.policyMutationMutex.Unlock()
		}
	}()

	var emptyJob model.Job

	manager.mu.Lock()
	current, found := manager.codebases[codebase.ID]
	if !found {
		manager.mu.Unlock()
		return emptyJob, fmt.Errorf("document collection not tracked: %s", codebase.CanonicalPath)
	}
	activeJob, active, err := manager.activeCollectionJobLocked(current)
	if err != nil {
		manager.mu.Unlock()
		return emptyJob, err
	}
	if active {
		// Coalesce an upsert onto the depth-1 pending slot instead of refusing, so a
		// backfill and a normal ingest on the same collection do not contend; the slot
		// drains into a fresh job on the active job's terminal transition. A delete
		// cannot losslessly fold into the single upsert-shaped slot, so it keeps the
		// refuse behavior.
		if payload.Kind == collectionJobKindUpsert {
			manager.mergePendingCollectionPayloadLocked(current.ID, payload)
			manager.mu.Unlock()
			return activeJob, nil
		}
		manager.mu.Unlock()
		return emptyJob, adapterr.NewActiveJobConflict(activeJob.ID, fmt.Sprintf("conflicting active job %s for document collection %s", activeJob.ID, current.CanonicalPath))
	}

	job, err := manager.enqueueCollectionJobLocked(current, client, payload)
	if err != nil {
		manager.mu.Unlock()
		return emptyJob, err
	}
	manager.mu.Unlock()

	ctx = spans.Attach(
		ctx,
		correlation.IdentityAttribute{Key: "job_id", Value: job.ID},
		correlation.IdentityAttribute{Key: "codebase_id", Value: current.ID},
	)
	manager.policyMutationMutex.Unlock()
	policyLocked = false
	manager.runJobAsync(ctx, job.ID)
	return job, nil
}

func (manager *Manager) activeCollectionJobLocked(codebase model.Codebase) (model.Job, bool, error) {
	var emptyJob model.Job
	if codebase.ActiveJobID == "" {
		return emptyJob, false, nil
	}
	activeJob, found := manager.jobs[codebase.ActiveJobID]
	if !found {
		return emptyJob, false, nil
	}
	switch activeJob.State {
	case model.JobStateCompleted, model.JobStateFailed, model.JobStateCancelled:
		return emptyJob, false, nil
	case model.JobStateQueued, model.JobStateRunning, model.JobStatePaused, model.JobStateCancelling:
		return activeJob, true, nil
	default:
		return emptyJob, false, fmt.Errorf("unknown job state %s for active job %s", activeJob.State, activeJob.ID)
	}
}

// runCollectionIngest runs one document collection job. An upsert runs the
// same delta-then-bootstrap routine code uses, with the collection item source
// that documentItemSource builds from the saved declaration. A delete drops one
// item's rows.
func (manager *Manager) runCollectionIngest(ctx context.Context, job model.Job) {
	payload, found := manager.collectionJobPayload(job.ID)
	if !found {
		manager.updateJobFailed(ctx, job.ID, errors.New("collection job payload missing"))
		return
	}

	select {
	case <-ctx.Done():
		manager.updateJobCancelled(ctx, job.ID)
		return
	default:
	}

	if manager.semantic == nil || !manager.semantic.Available() {
		manager.updateJobFailed(ctx, job.ID, semantic.ErrUnavailable)
		return
	}

	switch payload.Kind {
	case collectionJobKindDelete:
		manager.runCollectionDelete(ctx, job, payload)
	case collectionJobKindUpsert:
		source := manager.documentItemSource(job.CodebaseID, payload)
		if handled, _ := manager.runDeltaSync(ctx, job, source); handled {
			return
		}
		_ = manager.runBootstrap(ctx, job, source)
	default:
		manager.updateJobFailed(ctx, job.ID, fmt.Errorf("unknown collection job kind %s", payload.Kind))
	}
}

func (manager *Manager) runCollectionDelete(ctx context.Context, job model.Job, payload collectionJobPayload) {
	select {
	case <-ctx.Done():
		manager.updateJobCancelled(ctx, job.ID)
		return
	default:
	}

	if manager.semantic == nil || !manager.semantic.Available() {
		manager.updateJobFailed(ctx, job.ID, semantic.ErrUnavailable)
		return
	}
	removal := manager.itemSelector(job.CodebaseID).removal([]string{payload.ItemID})
	if err := manager.semantic.DeleteItemRows(ctx, payload.CollectionName, removal); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
			manager.updateJobCancelled(ctx, job.ID)
			return
		}
		manager.updateJobFailed(ctx, job.ID, err)
		return
	}
	manager.finishCollectionDelete(ctx, job.ID)
}

func (manager *Manager) finishCollectionDelete(ctx context.Context, jobID string) {
	manager.policyMutationMutex.Lock()
	defer manager.policyMutationMutex.Unlock()

	manager.transitionMutex.Lock()
	manager.mu.Lock()
	job, found := manager.jobs[jobID]
	if !found || isTerminalJobState(job.State) {
		delete(manager.collectionJobs, jobID)
		manager.mu.Unlock()
		manager.transitionMutex.Unlock()
		return
	}
	now := clock.Now()
	job.State = model.JobStateCompleted
	job.UpdatedAt = now
	job.CompletedAt = &now
	job.Progress.Phase = "completed"
	job.Progress.OverallPercent = 100
	job.Progress.LastEventAt = now
	job.Progress.HeartbeatAt = now
	jobEvent := model.JobEvent{Event: "job_completed", OccurredAt: clock.Now(), Job: job}
	manager.mu.Unlock()
	journalErr := manager.writeJobTransition(jobEvent)
	manager.mu.Lock()
	manager.jobs[jobID] = job
	if journalErr != nil {
		slog.ErrorContext(ctx, "append completed item delete event failed", "job_id", jobID, "err", journalErr)
	}
	delete(manager.collectionJobs, jobID)
	codebase, found := manager.codebases[job.CodebaseID]
	if !found {
		manager.mu.Unlock()
		manager.transitionMutex.Unlock()
		return
	}
	// Clear ActiveJobID only when it still points at this job, so a raced or
	// duplicate terminal transition never clobbers a drained successor.
	codebase.Status = model.CodebaseStatusIndexed
	if codebase.ActiveJobID == jobID {
		codebase.ActiveJobID = ""
	}
	codebase.UpdatedAt = now
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		slog.ErrorContext(ctx, "write registry after item delete failed", "job_id", jobID, "err", err)
	}
	// Pair the record write with one observer signal so no saveLocked path skips
	// invalidation; for a document collection it is a no-op delete.
	manager.observer.Invalidate(codebase.ID)
	// drainPendingJobLocked no-ops unless ActiveJobID was cleared above, so a raced
	// transition that did not own the slot never drains a duplicate.
	drainedJobID, drained := manager.drainPendingJobLocked(ctx, codebase.ID)
	codebaseID := codebase.ID
	manager.mu.Unlock()
	manager.transitionMutex.Unlock()
	if drained {
		manager.runDrainedJob(ctx, codebaseID, drainedJobID)
	}
}

func (manager *Manager) collectionJobPayload(jobID string) (collectionJobPayload, bool) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	payload, found := manager.collectionJobs[jobID]
	return payload, found
}

func (manager *Manager) findDocumentCollectionLocked(collectionID string) (model.Codebase, bool) {
	canonicalPath := documentCanonicalPath(collectionID)
	for _, codebase := range manager.codebases {
		if codebase.Kind != model.CodebaseKindDocument {
			continue
		}
		if codebase.CanonicalPath == canonicalPath {
			return codebase, true
		}
	}
	var emptyCodebase model.Codebase
	return emptyCodebase, false
}

func documentCanonicalPath(collectionID string) string {
	return documentCanonicalPathPrefix + collectionID
}
