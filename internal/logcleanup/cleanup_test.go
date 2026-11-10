package logcleanup

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const (
	activeLog     = "lm-semantic-search-daemon.log"
	newestBackup  = "lm-semantic-search-daemon-2026-07-05T10-00-00.000.log"
	middleBackup  = "lm-semantic-search-daemon-2026-07-03T10-00-00.000.log"
	oldestBackup  = "lm-semantic-search-daemon-2026-07-01T10-00-00.000.log"
	gzBackup      = "semantic-2026-07-02T10-00-00.000.jsonl.gz"
	concernActive = "semantic.jsonl"
)

func seedFile(t *testing.T, path string, size int, modTime time.Time) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, bytes.Repeat([]byte("x"), size), 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	if err := os.Chtimes(path, modTime, modTime); err != nil {
		t.Fatalf("chtimes %s: %v", path, err)
	}
}

func captureSlog(t *testing.T) *bytes.Buffer {
	t.Helper()
	buffer := &bytes.Buffer{}
	original := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(buffer, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(original) })
	return buffer
}

func parseLogEvent(t *testing.T, output string, msg string) map[string]any {
	t.Helper()
	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		if line == "" {
			continue
		}
		event := make(map[string]any)
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			continue
		}
		if event["msg"] == msg {
			return event
		}
	}
	return nil
}

// TestRunOnceDeletesOnlyBackupsOverBudgetKeepsActive proves the retention sweep
// deletes the oldest rotated backups past the byte budget while keeping the
// newer backups and the active file, even when the active file is the largest
// and oldest on disk.
func TestRunOnceDeletesOnlyBackupsOverBudgetKeepsActive(t *testing.T) {
	root := t.TempDir()
	veryOld := time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC)
	seedFile(t, filepath.Join(root, activeLog), 500, veryOld)
	seedFile(t, filepath.Join(root, newestBackup), 100, time.Date(2026, 7, 5, 10, 0, 0, 0, time.UTC))
	seedFile(t, filepath.Join(root, middleBackup), 100, time.Date(2026, 7, 3, 10, 0, 0, 0, time.UTC))
	seedFile(t, filepath.Join(root, oldestBackup), 100, time.Date(2026, 7, 1, 10, 0, 0, 0, time.UTC))

	result := RunOnce(Policy{Root: root, RetentionBytes: 250, Enabled: true})

	if result.Scanned != 4 {
		t.Fatalf("scanned=%d want 4", result.Scanned)
	}
	if result.Candidates != 1 {
		t.Fatalf("candidates=%d want 1", result.Candidates)
	}
	if result.Deleted != 1 {
		t.Fatalf("deleted=%d want 1", result.Deleted)
	}
	if result.BytesDeleted != 100 {
		t.Fatalf("bytes_deleted=%d want 100", result.BytesDeleted)
	}
	if _, err := os.Stat(filepath.Join(root, oldestBackup)); !os.IsNotExist(err) {
		t.Fatalf("oldest backup stat err=%v want not exist", err)
	}
	for _, kept := range []string{activeLog, newestBackup, middleBackup} {
		if _, err := os.Stat(filepath.Join(root, kept)); err != nil {
			t.Fatalf("kept file %s stat err=%v want present", kept, err)
		}
	}
}

// TestRunOnceNeverDeletesActiveEvenOverBudget guards the active-file rule: a
// budget of zero-effect over backups still keeps the current log intact.
func TestRunOnceNeverDeletesActiveFile(t *testing.T) {
	root := t.TempDir()
	seedFile(t, filepath.Join(root, activeLog), 10_000, time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC))
	seedFile(t, filepath.Join(root, concernActive), 10_000, time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC))

	result := RunOnce(Policy{Root: root, RetentionBytes: 1, Enabled: true})

	if result.Candidates != 0 || result.Deleted != 0 {
		t.Fatalf("active-only dir modified: candidates=%d deleted=%d", result.Candidates, result.Deleted)
	}
	for _, kept := range []string{activeLog, concernActive} {
		if _, err := os.Stat(filepath.Join(root, kept)); err != nil {
			t.Fatalf("active file %s removed: %v", kept, err)
		}
	}
}

// TestRunOnceEmitsCompletedEventWithCounts checks the cleanup.completed event
// carries the scanned, deleted, bytes_deleted, and duration_ms counts.
func TestRunOnceEmitsCompletedEventWithCounts(t *testing.T) {
	root := t.TempDir()
	seedFile(t, filepath.Join(root, activeLog), 20, time.Date(2026, 7, 6, 0, 0, 0, 0, time.UTC))
	seedFile(t, filepath.Join(root, newestBackup), 100, time.Date(2026, 7, 5, 10, 0, 0, 0, time.UTC))
	seedFile(t, filepath.Join(root, oldestBackup), 100, time.Date(2026, 7, 1, 10, 0, 0, 0, time.UTC))

	buffer := captureSlog(t)
	RunOnce(Policy{Root: root, RetentionBytes: 100, Enabled: true})

	completed := parseLogEvent(t, buffer.String(), completedEvent)
	if completed == nil {
		t.Fatalf("no %s event in: %s", completedEvent, buffer.String())
	}
	if scanned, ok := completed["scanned"].(float64); !ok || scanned != 3 {
		t.Fatalf("scanned=%v want 3", completed["scanned"])
	}
	if deleted, ok := completed["deleted"].(float64); !ok || deleted != 1 {
		t.Fatalf("deleted=%v want 1", completed["deleted"])
	}
	if bytesDeleted, ok := completed["bytes_deleted"].(float64); !ok || bytesDeleted != 100 {
		t.Fatalf("bytes_deleted=%v want 100", completed["bytes_deleted"])
	}
	if duration, ok := completed["duration_ms"].(float64); !ok || duration <= 0 {
		t.Fatalf("duration_ms=%v want > 0", completed["duration_ms"])
	}
}

// TestRunOnceDisabledAuditsButDeletesNothing proves a disabled policy counts
// candidates and emits an audit event without removing any file.
func TestRunOnceDisabledAuditsButDeletesNothing(t *testing.T) {
	root := t.TempDir()
	seedFile(t, filepath.Join(root, newestBackup), 100, time.Date(2026, 7, 5, 10, 0, 0, 0, time.UTC))
	seedFile(t, filepath.Join(root, oldestBackup), 100, time.Date(2026, 7, 1, 10, 0, 0, 0, time.UTC))

	buffer := captureSlog(t)
	result := RunOnce(Policy{Root: root, RetentionBytes: 100, Enabled: false})

	if result.Candidates != 1 {
		t.Fatalf("candidates=%d want 1", result.Candidates)
	}
	if result.Deleted != 0 || result.BytesDeleted != 0 {
		t.Fatalf("disabled sweep deleted: deleted=%d bytes=%d", result.Deleted, result.BytesDeleted)
	}
	for _, kept := range []string{newestBackup, oldestBackup} {
		if _, err := os.Stat(filepath.Join(root, kept)); err != nil {
			t.Fatalf("disabled sweep removed %s: %v", kept, err)
		}
	}
	audit := parseLogEvent(t, buffer.String(), auditEvent)
	if audit == nil {
		t.Fatalf("no %s event in: %s", auditEvent, buffer.String())
	}
	if would, ok := audit["would_delete_bytes"].(float64); !ok || would != 100 {
		t.Fatalf("would_delete_bytes=%v want 100", audit["would_delete_bytes"])
	}
}

func TestRunOnceMissingRootIsNoOp(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "does-not-exist")
	result := RunOnce(Policy{Root: missing, RetentionBytes: 100, Enabled: true})
	if result.Scanned != 0 || result.Deleted != 0 {
		t.Fatalf("missing root produced work: %+v", result)
	}
}

func TestOverflowBackupsSelectsOldestFirst(t *testing.T) {
	backups := []logFile{
		{path: "new", size: 100, modTime: time.Date(2026, 7, 5, 0, 0, 0, 0, time.UTC)},
		{path: "mid", size: 100, modTime: time.Date(2026, 7, 3, 0, 0, 0, 0, time.UTC)},
		{path: "old", size: 100, modTime: time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)},
	}
	overflow := overflowBackups(backups, 150)
	if len(overflow) != 2 {
		t.Fatalf("overflow len=%d want 2", len(overflow))
	}
	if overflow[0].path != "mid" || overflow[1].path != "old" {
		t.Fatalf("overflow paths=%v want [mid old]", []string{overflow[0].path, overflow[1].path})
	}
	if overflowBackups(backups, 0) != nil {
		t.Fatalf("zero budget should keep everything")
	}
}

func TestIsRotatedBackupClassification(t *testing.T) {
	cases := map[string]bool{
		"lm-semantic-search-daemon.log":                            false,
		"semantic.jsonl":                                           false,
		"lm-semantic-search-daemon-2026-07-01T10-00-00.000.log":    true,
		"semantic-2026-07-01T10-00-00.000.jsonl":                   true,
		"semantic-2026-07-01T10-00-00.000.jsonl.gz":                true,
		"lm-semantic-search-daemon-2026-07-01T10-00-00.000.log.gz": true,
	}
	for name, want := range cases {
		if got := isRotatedBackup(name); got != want {
			t.Errorf("isRotatedBackup(%q)=%v want %v", name, got, want)
		}
	}
}

// TestStartRunsImmediatelyThenStopsOnCancel proves the walker runs one pass at
// boot without waiting for the interval, and stops when the context cancels.
func TestStartRunsImmediatelyThenStopsOnCancel(t *testing.T) {
	root := t.TempDir()
	seedFile(t, filepath.Join(root, newestBackup), 100, time.Date(2026, 7, 5, 10, 0, 0, 0, time.UTC))
	seedFile(t, filepath.Join(root, oldestBackup), 100, time.Date(2026, 7, 1, 10, 0, 0, 0, time.UTC))

	ctx, cancel := context.WithCancel(context.Background())
	Start(ctx, Policy{Root: root, RetentionBytes: 100, Enabled: true}, time.Hour)

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(filepath.Join(root, oldestBackup)); os.IsNotExist(err) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	cancel()

	if _, err := os.Stat(filepath.Join(root, oldestBackup)); !os.IsNotExist(err) {
		t.Fatalf("immediate pass did not delete oldest backup: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, newestBackup)); err != nil {
		t.Fatalf("immediate pass removed newest backup: %v", err)
	}
}
