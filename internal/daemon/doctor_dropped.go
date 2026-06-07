package daemon

import (
	"fmt"
	"sort"
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
)

// computeDroppedCodebases returns the sorted canonical paths that have a
// completed indexing job but are absent from the current registry while their
// source directory still exists on disk. Such a path was indexed before and
// silently fell out of tracking, so it would otherwise go stale forever without
// surfacing anywhere; a path that was never indexed is intentionally left alone.
// The presence check runs through exists so the computation stays unit-testable
// without touching the filesystem.
func computeDroppedCodebases(jobs []model.Job, codebases []model.Codebase, exists func(string) bool) []string {
	tracked := make(map[string]struct{}, len(codebases))
	for _, codebase := range codebases {
		if codebase.CanonicalPath == "" {
			continue
		}
		tracked[codebase.CanonicalPath] = struct{}{}
	}

	indexedBefore := make(map[string]struct{})
	for _, job := range jobs {
		if job.State != model.JobStateCompleted {
			continue
		}
		if job.CanonicalPath == "" {
			continue
		}
		indexedBefore[job.CanonicalPath] = struct{}{}
	}

	dropped := make([]string, 0)
	for path := range indexedBefore {
		if _, stillTracked := tracked[path]; stillTracked {
			continue
		}
		if !exists(path) {
			continue
		}
		dropped = append(dropped, path)
	}
	sort.Strings(dropped)
	return dropped
}

// renderDroppedSection formats the dropped-codebase section for the doctor
// surface. With no dropped codebases it states that none exist, so the section
// reads as a deliberate clean result rather than a missing check.
func renderDroppedSection(dropped []string) string {
	if len(dropped) == 0 {
		return "Dropped codebases (completed index, now untracked, still on disk): none"
	}

	lines := make([]string, 0, len(dropped)+1)
	lines = append(lines, fmt.Sprintf("Dropped codebases (completed index, now untracked, still on disk): %d", len(dropped)))
	for _, path := range dropped {
		lines = append(lines, "- "+path)
	}
	return strings.Join(lines, "\n")
}

// DroppedCodebases reports the canonical paths that completed an index, are no
// longer tracked, and still exist on disk. It snapshots the job and codebase
// state under the lock so the doctor surface can include the dropped section
// without the MCP adapter recomputing it client-side.
func (manager *Manager) DroppedCodebases() []string {
	manager.mu.Lock()
	jobs := make([]model.Job, 0, len(manager.jobs))
	for _, job := range manager.jobs {
		jobs = append(jobs, job)
	}
	codebases := make([]model.Codebase, 0, len(manager.codebases))
	for _, codebase := range manager.codebases {
		codebases = append(codebases, codebase)
	}
	manager.mu.Unlock()
	return computeDroppedCodebases(jobs, codebases, func(path string) bool {
		return !sourceDirMissing(path)
	})
}
