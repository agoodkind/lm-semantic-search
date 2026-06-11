package daemon

import (
	"context"
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
