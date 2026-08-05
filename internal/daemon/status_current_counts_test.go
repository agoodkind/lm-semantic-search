package daemon

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
)

func TestGetIndexReadyStatusShowsCurrentAndLastRunCounts(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonicalPath, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}

	completedAt := time.Date(2026, time.August, 5, 13, 46, 0, 0, time.UTC)
	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
	codebase := newCodebaseRecord(canonicalPath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	codebase.LastSuccessfulRun = &model.IndexRunSummary{
		IndexedFiles: 1,
		TotalChunks:  0,
		CompletedAt:  completedAt,
		SkippedFiles: []string{"legacy.go"},
	}

	files := make(map[string]string, 72)
	for i := range 72 {
		files[filepath.Join("pkg", fmt.Sprintf("file-%d.go", i))] = "hash"
	}
	snapshot := merkle.Snapshot{
		ConfigDigest: indexConfig.IgnoreDigest,
		Files:        files,
		Inodes:       nil,
	}
	if err := merkle.WriteSnapshot(manager.merklePath(codebase.ID), snapshot); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
	snapshotPath := manager.merklePath(codebase.ID)
	snapshotBefore, err := os.ReadFile(snapshotPath)
	if err != nil {
		t.Fatalf("ReadFile before GetIndex returned error: %v", err)
	}

	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	manager.semantic = &fakeSemantic{
		count: func(context.Context, string) (int32, error) {
			return 359, nil
		},
	}

	response, err := NewGRPCServer(manager, nil).GetIndex(
		context.Background(),
		&pb.GetIndexRequest{Path: repoPath},
	)
	if err != nil {
		t.Fatalf("GetIndex returned error: %v", err)
	}

	displayText := response.GetDisplayText()
	for _, want := range []string{
		"📁 " + filepath.Base(canonicalPath),
		"✅ Ready to search",
		"codebase.status: indexed",
		"current_index.indexed_files: 72",
		"current_index.total_chunks: 359",
		"last_successful_run.indexed_files: 1",
		"last_successful_run.total_chunks: 0",
		"last_successful_run.completed_at: " + formatBoundaryStatusTime(completedAt),
		"⏭️  Skipped: 1 non-UTF-8 file: legacy.go",
		"🕐 Semantic: updated",
		"🕸️ Code graph: builds shortly",
	} {
		if !strings.Contains(displayText, want) {
			t.Fatalf("ready status missing %q:\n%s", want, displayText)
		}
	}
	if strings.Contains(displayText, "📊 1 files, 0 chunks") {
		t.Fatalf("ready status still contains the ambiguous count line:\n%s", displayText)
	}

	manager.semantic = &fakeSemantic{
		count: func(context.Context, string) (int32, error) {
			return 0, errors.New("count unavailable")
		},
	}
	response, err = NewGRPCServer(manager, nil).GetIndex(
		context.Background(),
		&pb.GetIndexRequest{Path: repoPath},
	)
	if err != nil {
		t.Fatalf("GetIndex with failed count returned error: %v", err)
	}
	displayText = response.GetDisplayText()
	for _, want := range []string{
		"current_index.total_chunks: null",
		"last_successful_run.indexed_files: 1",
		"last_successful_run.total_chunks: 0",
	} {
		if !strings.Contains(displayText, want) {
			t.Fatalf("failed count status missing %q:\n%s", want, displayText)
		}
	}

	manager.mu.Lock()
	codebaseAfter := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if !reflect.DeepEqual(codebaseAfter, codebase) {
		t.Fatalf("GetIndex mutated codebase:\ngot  %+v\nwant %+v", codebaseAfter, codebase)
	}
	snapshotAfter, err := os.ReadFile(snapshotPath)
	if err != nil {
		t.Fatalf("ReadFile after GetIndex returned error: %v", err)
	}
	if !reflect.DeepEqual(snapshotAfter, snapshotBefore) {
		t.Fatal("GetIndex changed the live Merkle checkpoint")
	}
	if jobs := manager.ListJobs(""); len(jobs) != 0 {
		t.Fatalf("GetIndex created %d jobs, want 0", len(jobs))
	}
}
