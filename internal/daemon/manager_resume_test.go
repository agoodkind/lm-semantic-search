package daemon

import (
	"context"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
)

// A document conversation codebase left mid-index must never be re-launched
// by the boot resume pass: its path is a chat URI, not a directory, and the
// conversation trigger path owns its recovery.
func TestResumeOrphanedJobsSkipsDocumentCodebases(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.config.ResumeIndexingOnBoot = true

	codebase := newCodebaseRecord("chat:///clyde-conversations")
	codebase.Kind = model.CodebaseKindDocument
	codebase.Status = model.CodebaseStatusIndexing
	codebase.EffectiveConfig = defaultIndexConfig()
	codebase.EffectiveConfig.IgnoreDigest = "sha256:document-resume"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	snapshot := merkle.Snapshot{
		ConfigDigest: codebase.EffectiveConfig.IgnoreDigest,
		Files:        map[string]string{"conversation.json": "sha256:seed"},
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebase.ID), snapshot); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	manager.ResumeOrphanedJobs(context.Background())

	manager.mu.Lock()
	jobCount := len(manager.jobs)
	manager.mu.Unlock()
	if jobCount != 0 {
		t.Fatalf("resume launched %d job(s) for a document codebase, want 0", jobCount)
	}
}

const resumeCheckpointReadFailure = "read Merkle snapshot failed"

// resumeProbeReached is the line ResumeOrphanedJobs emits once the live
// checkpoint probe has run and come back with nothing to resume from. A test
// asserting that the probe stayed quiet has to assert this too, or it would
// pass just as happily when the codebase was filtered out before the probe and
// nothing was ever checked.
const resumeProbeReached = "skipping unresumable interrupted index"

// requireResumeProbeRan fails when the boot resume never reached the checkpoint
// probe for this codebase, which is what makes the quiet-absence assertions
// below mean something.
func requireResumeProbeRan(t *testing.T, logs *capturedLogs, codebaseID string) {
	t.Helper()
	if len(logs.linesContaining(resumeProbeReached, codebaseID, "reason=no_checkpoint")) == 0 {
		t.Fatalf("boot resume never probed %s for a checkpoint, so a quiet log proves nothing:\n%s", codebaseID, logs.text())
	}
}

// A codebase the registry says already completed a run owns a live checkpoint,
// so boot resume finding that file gone means the index lost state. The
// operator has to see that as an error, because the repair pass cannot force a
// real read when the semantic backend is down and the loss would otherwise
// stay hidden behind a debug line that reads like a normal first build.
func TestResumeOrphanedJobsReportsLostCheckpointAfterSuccessfulRun(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	manager.config.ResumeIndexingOnBoot = true

	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexing
	codebase.EffectiveConfig = defaultIndexConfig()
	codebase.EffectiveConfig.IgnoreDigest = "sha256:lost-checkpoint"
	codebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed"}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	logs := captureLogs(t)
	manager.ResumeOrphanedJobs(context.Background())

	if len(logs.linesContaining("level=ERROR", resumeCheckpointReadFailure, manager.merklePath(codebase.ID))) == 0 {
		t.Fatalf("a checkpoint lost after a completed run was not reported as an error:\n%s", logs.text())
	}
}

// The same probe on a codebase that has never completed a run must stay quiet:
// a first build has no live checkpoint to lose, so reporting its absence as a
// fault would make every healthy first index look broken.
func TestResumeOrphanedJobsKeepsFirstBuildCheckpointAbsenceQuiet(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	manager.config.ResumeIndexingOnBoot = true

	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexing
	codebase.EffectiveConfig = defaultIndexConfig()
	codebase.EffectiveConfig.IgnoreDigest = "sha256:first-build"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	logs := captureLogs(t)
	manager.ResumeOrphanedJobs(context.Background())

	requireResumeProbeRan(t, logs, codebase.ID)
	if lines := logs.linesContaining(resumeCheckpointReadFailure, manager.merklePath(codebase.ID)); len(lines) > 0 {
		t.Fatalf("a first build's absent checkpoint was reported as a read failure:\n%s", strings.Join(lines, "\n"))
	}
}

// A run that completed without indexing a file never wrote a checkpoint, so
// boot resume must read its absence the same way it reads a first build's. This
// is the empty repository, the fully ignored one, and the one holding only
// oversized or unreadable files.
func TestResumeOrphanedJobsKeepsZeroFileRunCheckpointAbsenceQuiet(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	manager.config.ResumeIndexingOnBoot = true

	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexing
	codebase.EffectiveConfig = defaultIndexConfig()
	codebase.EffectiveConfig.IgnoreDigest = "sha256:zero-file-run"
	codebase.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 0, TotalChunks: 0, Status: "completed"}
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	logs := captureLogs(t)
	manager.ResumeOrphanedJobs(context.Background())

	requireResumeProbeRan(t, logs, codebase.ID)
	if lines := logs.linesContaining(resumeCheckpointReadFailure, manager.merklePath(codebase.ID)); len(lines) > 0 {
		t.Fatalf("a completed run that indexed no file was reported as having lost its checkpoint:\n%s", strings.Join(lines, "\n"))
	}
}
