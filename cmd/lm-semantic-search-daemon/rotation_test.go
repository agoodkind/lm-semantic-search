package main

import (
	"bytes"
	"log/slog"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/gklog"
	"goodkind.io/lm-semantic-search/internal/config"
)

func TestRotationConfigPopulatedFromBytes(t *testing.T) {
	rot := rotationConfig(config.Config{LogRotationMaxBytes: 5 * 1024 * 1024})
	if rot.MaxSizeMB != 5 {
		t.Fatalf("MaxSizeMB = %d want 5", rot.MaxSizeMB)
	}
	if (rot == gklog.RotationConfig{}) {
		t.Fatalf("rotation config is empty; want populated so files rotate")
	}
}

func TestRotationConfigFloorsSubMegabyteCap(t *testing.T) {
	rot := rotationConfig(config.Config{LogRotationMaxBytes: 1024})
	if rot.MaxSizeMB != minRotationMB {
		t.Fatalf("MaxSizeMB = %d want floor %d", rot.MaxSizeMB, minRotationMB)
	}
}

// TestInstallConcernRouterRoutesFilesAndWriteNeverDeletes proves two things:
// the combined service stream and per-concern stream both land in rotating
// files under logsDir, and the log-write path never runs a retention delete, so
// stale-file cleanup stays off the hot path.
//
// The seeded backup belongs to a concern the test never logs, so no lumberjack
// instance manages it. Rotation only ever touches the active files and their
// own backups; a retention delete of an unrelated backup could come only from
// the logcleanup sweep, which the write path never calls.
func TestInstallConcernRouterRoutesFilesAndWriteNeverDeletes(t *testing.T) {
	logsDir := t.TempDir()
	logPath := filepath.Join(logsDir, "lm-semantic-search-daemon.log")

	unmanagedBackup := filepath.Join(logsDir, "retire-2026-07-01T10-00-00.000.jsonl")
	if err := os.WriteFile(unmanagedBackup, bytes.Repeat([]byte("x"), 4096), 0o600); err != nil {
		t.Fatalf("seed backup: %v", err)
	}

	original := slog.Default()
	t.Cleanup(func() { slog.SetDefault(original) })

	installConcernRouter(logsDir, logPath, rotationConfig(config.Config{LogRotationMaxBytes: 5 * 1024 * 1024}))

	for i := 0; i < 50; i++ {
		slog.Info("semantic.reindex", "n", i)
	}

	if _, err := os.Stat(logPath); err != nil {
		t.Fatalf("combined service log not created at %s: %v", logPath, err)
	}
	if _, err := os.Stat(filepath.Join(logsDir, "semantic.jsonl")); err != nil {
		t.Fatalf("per-concern semantic log not created: %v", err)
	}
	if _, err := os.Stat(unmanagedBackup); err != nil {
		t.Fatalf("write path deleted an unrelated backup; retention is not off the hot path: %v", err)
	}
}
