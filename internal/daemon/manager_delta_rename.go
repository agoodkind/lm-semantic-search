package daemon

import (
	"context"
	"log/slog"
	"slices"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// deltaRenamePair is one removed path and one added path with the same inode
// and the same content hash.
type deltaRenamePair struct {
	source      string
	destination string
	fileHash    string
}

// deltaRenamePairs matches each added path to a removed path that has the same
// inode in the seed snapshot and the same content hash. It excludes added paths
// in forced and does not match paths when inodeTrackingDisabled is true.
func deltaRenamePairs(plan deltaPlan, forced map[string]struct{}, inodeTrackingDisabled bool) []deltaRenamePair {
	if inodeTrackingDisabled || len(plan.diff.Added) == 0 || len(plan.diff.Removed) == 0 {
		return nil
	}
	removed := make(map[string]struct{}, len(plan.diff.Removed))
	for _, relativePath := range plan.diff.Removed {
		removed[relativePath] = struct{}{}
	}
	matchedSources := make(map[string]struct{})
	pairs := make([]deltaRenamePair, 0)
	for _, destination := range plan.diff.Added {
		if _, isForced := forced[destination]; isForced {
			continue
		}
		currentInode, found := plan.currentSnapshot.Inodes[destination]
		if !found {
			continue
		}
		candidates := make([]string, 0)
		for _, candidate := range plan.seedSnapshot.LookupByInode(currentInode) {
			if _, isRemoved := removed[candidate]; !isRemoved {
				continue
			}
			if _, matched := matchedSources[candidate]; matched {
				continue
			}
			candidates = append(candidates, candidate)
		}
		slices.Sort(candidates)
		fileHash := plan.currentSnapshot.Files[destination]
		source := pickRenameSource(candidates, plan.seedSnapshot.Files, fileHash)
		if source == "" {
			continue
		}
		matchedSources[source] = struct{}{}
		pairs = append(pairs, deltaRenamePair{source: source, destination: destination, fileHash: fileHash})
	}
	return pairs
}

// applyDeltaRemovals calls applyDeltaRenameCopies first. It then deletes rows
// for the removed paths that remain in the diff, removes those paths from the
// working snapshot, and writes a checkpoint.
func (manager *Manager) applyDeltaRemovals(ctx context.Context, job model.Job, state *deltaState) deltaOutcome {
	if outcome := manager.applyDeltaRenameCopies(ctx, job, state); outcome.handled {
		return outcome
	}
	removed := state.plan.diff.Removed
	if len(removed) == 0 || !state.semantic {
		return deltaOutcome{fallback: false, handled: false, progressed: false}
	}
	if outcome := manager.applyReindexForState(ctx, job, *state, nil, state.source.removalFor(removed), "delta removal"); outcome.fallback || outcome.handled {
		return outcome
	}
	for _, path := range removed {
		delete(state.working, path)
	}
	manager.writeCheckpoint(ctx, *state, "removals")
	if err := manager.checkpointJob(ctx); err != nil {
		return deltaOutcome{fallback: false, handled: true, progressed: false}
	}
	return deltaOutcome{fallback: false, handled: false, progressed: false}
}

// applyDeltaRenameCopies calls semantic.Service.CopyChunks for the pairs from
// deltaRenamePairs. CopyChunks inserts stored rows under each destination path
// and deletes the source rows. After a successful call, applyDeltaRenameCopies
// removes each pair with copied rows from the diff, deletes its source path from
// the working snapshot, and records its destination hash. It records each copied
// pair in renamedPaths. A copy error leaves every pair in the diff for the normal
// delete and embed steps.
func (manager *Manager) applyDeltaRenameCopies(ctx context.Context, job model.Job, state *deltaState) deltaOutcome {
	if !state.semantic {
		return deltaOutcome{fallback: false, handled: false, progressed: false}
	}
	pairs := deltaRenamePairs(state.plan, state.forced, state.inodeTrackingDisabled)
	if len(pairs) == 0 {
		return deltaOutcome{fallback: false, handled: false, progressed: false}
	}
	copies := make([]semantic.ChunkCopy, 0, len(pairs))
	for _, pair := range pairs {
		copies = append(copies, semantic.ChunkCopy{Source: pair.source, Destination: pair.destination})
	}
	copiedRowCounts, copyErr := manager.semantic.CopyChunks(ctx, job.CanonicalPath, copies)
	for _, count := range copiedRowCounts {
		manager.addJobWriteTotals(job.ID, int64(count), 0)
	}
	if copyErr != nil {
		slog.WarnContext(ctx, "delta.copy_chunks_fallback", "component", "daemon", "subcomponent", "delta", "job_id", job.ID, "copies", len(copies), "err", copyErr)
		return deltaOutcome{fallback: false, handled: false, progressed: false}
	}
	copiedSources := make(map[string]struct{}, len(pairs))
	copiedDestinations := make(map[string]struct{}, len(pairs))
	for index, pair := range pairs {
		if copiedRowCounts[index] == 0 {
			continue
		}
		state.working[pair.destination] = pair.fileHash
		delete(state.working, pair.source)
		state.renamedPaths[pair.source] = pair.destination
		copiedSources[pair.source] = struct{}{}
		copiedDestinations[pair.destination] = struct{}{}
	}
	slog.InfoContext(ctx, "delta.copy_chunks", "component", "daemon", "subcomponent", "delta", "job_id", job.ID, "renames", len(pairs), "copied", len(copiedDestinations))
	if len(copiedDestinations) == 0 {
		return deltaOutcome{fallback: false, handled: false, progressed: false}
	}
	state.plan.diff = merkle.Diff{
		Added:    pathsWithout(state.plan.diff.Added, copiedDestinations),
		Modified: state.plan.diff.Modified,
		Removed:  pathsWithout(state.plan.diff.Removed, copiedSources),
	}
	manager.writeCheckpoint(ctx, *state, "rename copies")
	if err := manager.checkpointJob(ctx); err != nil {
		return deltaOutcome{fallback: false, handled: true, progressed: false}
	}
	return deltaOutcome{fallback: false, handled: false, progressed: true}
}

func pathsWithout(paths []string, excluded map[string]struct{}) []string {
	kept := make([]string, 0, len(paths))
	for _, relativePath := range paths {
		if _, skip := excluded[relativePath]; skip {
			continue
		}
		kept = append(kept, relativePath)
	}
	return kept
}
