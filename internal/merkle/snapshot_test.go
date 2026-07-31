package merkle

import (
	"bytes"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

// guardedBuffer collects the process-global logger's output. slog.SetDefault is
// process-wide, so the writer is mutex-guarded and its users must not call
// t.Parallel.
type guardedBuffer struct {
	mu     sync.Mutex
	buffer bytes.Buffer
}

func (guarded *guardedBuffer) Write(data []byte) (int, error) {
	guarded.mu.Lock()
	defer guarded.mu.Unlock()
	return guarded.buffer.Write(data)
}

func (guarded *guardedBuffer) text() string {
	guarded.mu.Lock()
	defer guarded.mu.Unlock()
	return guarded.buffer.String()
}

// linesContaining returns the captured lines carrying every supplied substring,
// so an assertion is scoped to the path under test rather than to whatever else
// logged during the window.
func (guarded *guardedBuffer) linesContaining(matches ...string) []string {
	found := make([]string, 0)
	for _, line := range strings.Split(guarded.text(), "\n") {
		carriesAll := true
		for _, match := range matches {
			if !strings.Contains(line, match) {
				carriesAll = false
				break
			}
		}
		if carriesAll {
			found = append(found, line)
		}
	}
	return found
}

func captureDefaultLogger(t *testing.T) *guardedBuffer {
	t.Helper()

	logs := &guardedBuffer{}
	previousLogger := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(previousLogger) })
	return logs
}

// An absent checkpoint is a fact this package reports, never a fault it
// declares. It cannot tell a checkpoint lost after a real run from one a
// zero-file run never wrote, so it returns os.ErrNotExist and leaves the
// verdict to the caller that holds the codebase's history.
func TestLoadSnapshotForConfigMissingFileReportsAbsenceWithoutClaimingFailure(t *testing.T) {
	missingPath := filepath.Join(t.TempDir(), "missing.json")
	logs := captureDefaultLogger(t)

	snapshot, err := LoadSnapshotForConfig(missingPath, "sha256:request", "")
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("err = %v, want one wrapping os.ErrNotExist", err)
	}
	if snapshot.ConfigDigest != "sha256:request" {
		t.Fatalf("ConfigDigest = %q, want sha256:request", snapshot.ConfigDigest)
	}
	if len(snapshot.Files) != 0 {
		t.Fatalf("Files = %#v, want empty", snapshot.Files)
	}
	if len(logs.linesContaining("level=DEBUG", "read Merkle snapshot not found; starting fresh", missingPath)) == 0 {
		t.Fatalf("the absent checkpoint was not reported at debug level:\n%s", logs.text())
	}
	if lines := logs.linesContaining("level=ERROR", missingPath); len(lines) > 0 {
		t.Fatalf("the absent checkpoint was reported as a failure:\n%s", strings.Join(lines, "\n"))
	}
}

// A file that is there but cannot be parsed is a real failure, so it is
// reported as one. This is the half the quiet-absence rule must not swallow.
func TestLoadSnapshotForConfigUnparseableFileReportsFailure(t *testing.T) {
	corruptPath := filepath.Join(t.TempDir(), "corrupt.json")
	if err := os.WriteFile(corruptPath, []byte("{"), 0o600); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	logs := captureDefaultLogger(t)

	snapshot, err := LoadSnapshotForConfig(corruptPath, "sha256:request", "")
	if err == nil {
		t.Fatal("err = nil, want the parse failure")
	}
	if errors.Is(err, os.ErrNotExist) {
		t.Fatalf("err = %v, want a parse failure rather than an absence", err)
	}
	if len(snapshot.Files) != 0 {
		t.Fatalf("Files = %#v, want empty", snapshot.Files)
	}
	if len(logs.linesContaining("level=ERROR", "unmarshal Merkle snapshot failed", corruptPath)) == 0 {
		t.Fatalf("the unparseable checkpoint was not reported as a failure:\n%s", logs.text())
	}
}

// A snapshot written under another index config cannot seed this build, and the
// caller has to be able to tell that apart from an absent file, because only
// the absent one can mean lost state.
func TestLoadSnapshotForConfigMismatchedDigestReportsItsOwnCause(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "snapshot.json")
	stored := Snapshot{ConfigDigest: "sha256:digest-a", Files: map[string]string{"a.go": "hash-a"}}
	if err := WriteSnapshot(path, stored); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	_, err := LoadSnapshotForConfig(path, "sha256:digest-b", "")
	if !errors.Is(err, ErrConfigDigestMismatch) {
		t.Fatalf("err = %v, want one wrapping ErrConfigDigestMismatch", err)
	}
	if errors.Is(err, os.ErrNotExist) {
		t.Fatalf("err = %v, want a mismatch rather than an absence", err)
	}
}

func TestLoadSnapshotForConfigMatchingDigest(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "snapshot.json")
	stored := Snapshot{ConfigDigest: "sha256:digest-a", Files: map[string]string{"a.go": "hash-a"}}
	if err := WriteSnapshot(path, stored); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
	loaded, err := LoadSnapshotForConfig(path, "sha256:digest-a", "")
	if err != nil {
		t.Fatalf("LoadSnapshotForConfig returned error: %v", err)
	}
	if loaded.ConfigDigest != "sha256:digest-a" {
		t.Fatalf("ConfigDigest = %q", loaded.ConfigDigest)
	}
	if !reflect.DeepEqual(loaded.Files, stored.Files) {
		t.Fatalf("Files = %#v, want %#v", loaded.Files, stored.Files)
	}
}

func TestLoadSnapshotForConfigMismatchedDigest(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "snapshot.json")
	stored := Snapshot{ConfigDigest: "sha256:digest-a", Files: map[string]string{"a.go": "hash-a"}}
	if err := WriteSnapshot(path, stored); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
	loaded, _ := LoadSnapshotForConfig(path, "sha256:digest-b", "")
	if loaded.ConfigDigest != "sha256:digest-b" {
		t.Fatalf("ConfigDigest = %q, want fresh sha256:digest-b", loaded.ConfigDigest)
	}
	if len(loaded.Files) != 0 {
		t.Fatalf("Files = %#v, want empty due to digest mismatch", loaded.Files)
	}
}

// TestLoadSnapshotForConfigLegacyAcceptDigest proves a snapshot from before
// the ConfigDigest field existed is salvaged when the legacy fallback
// matches the request digest. The returned snapshot is stamped with the
// request digest so subsequent saves keep it valid.
func TestLoadSnapshotForConfigLegacyAcceptDigest(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "legacy.json")
	stored := Snapshot{ConfigDigest: "", Files: map[string]string{"a.go": "hash-a"}}
	if err := WriteSnapshot(path, stored); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
	loaded, err := LoadSnapshotForConfig(path, "sha256:request", "sha256:request")
	if err != nil {
		t.Fatalf("LoadSnapshotForConfig returned error: %v", err)
	}
	if loaded.ConfigDigest != "sha256:request" {
		t.Fatalf("ConfigDigest = %q, want stamped request digest", loaded.ConfigDigest)
	}
	if !reflect.DeepEqual(loaded.Files, stored.Files) {
		t.Fatalf("Files = %#v, want %#v", loaded.Files, stored.Files)
	}
}

// TestLoadSnapshotForConfigLegacyRejectsOnMismatch proves a legacy snapshot
// is treated as empty when the legacy fallback does not match the request.
func TestLoadSnapshotForConfigLegacyRejectsOnMismatch(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "legacy.json")
	stored := Snapshot{ConfigDigest: "", Files: map[string]string{"a.go": "hash-a"}}
	if err := WriteSnapshot(path, stored); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
	loaded, _ := LoadSnapshotForConfig(path, "sha256:request", "sha256:other")
	if len(loaded.Files) != 0 {
		t.Fatalf("Files = %#v, want empty due to legacy digest mismatch", loaded.Files)
	}
}

func TestDiffSnapshotsEmpty(t *testing.T) {
	t.Parallel()

	prev := Snapshot{Files: map[string]string{"a.go": "hash-a"}}
	current := Snapshot{Files: map[string]string{"a.go": "hash-a"}}
	diff := DiffSnapshots(prev, current)
	if !diff.Empty() {
		t.Fatalf("diff = %#v, want empty", diff)
	}
}

func TestDiffSnapshotsAddOnly(t *testing.T) {
	t.Parallel()

	prev := Snapshot{Files: map[string]string{"a.go": "hash-a"}}
	current := Snapshot{Files: map[string]string{
		"a.go":  "hash-a",
		"b.go":  "hash-b",
		"sub/c": "hash-c",
	}}
	diff := DiffSnapshots(prev, current)
	if !reflect.DeepEqual(diff.Added, []string{"b.go", "sub/c"}) {
		t.Fatalf("added = %#v", diff.Added)
	}
	if len(diff.Modified) != 0 || len(diff.Removed) != 0 {
		t.Fatalf("diff = %#v", diff)
	}
}

func TestDiffSnapshotsModifyOnly(t *testing.T) {
	t.Parallel()

	prev := Snapshot{Files: map[string]string{"a.go": "hash-old", "b.go": "hash-b"}}
	current := Snapshot{Files: map[string]string{"a.go": "hash-new", "b.go": "hash-b"}}
	diff := DiffSnapshots(prev, current)
	if !reflect.DeepEqual(diff.Modified, []string{"a.go"}) {
		t.Fatalf("modified = %#v", diff.Modified)
	}
	if len(diff.Added) != 0 || len(diff.Removed) != 0 {
		t.Fatalf("diff = %#v", diff)
	}
}

func TestDiffSnapshotsRemoveOnly(t *testing.T) {
	t.Parallel()

	prev := Snapshot{Files: map[string]string{"a.go": "hash-a", "old.go": "hash-old"}}
	current := Snapshot{Files: map[string]string{"a.go": "hash-a"}}
	diff := DiffSnapshots(prev, current)
	if !reflect.DeepEqual(diff.Removed, []string{"old.go"}) {
		t.Fatalf("removed = %#v", diff.Removed)
	}
	if len(diff.Added) != 0 || len(diff.Modified) != 0 {
		t.Fatalf("diff = %#v", diff)
	}
}

func TestDiffSnapshotsMixed(t *testing.T) {
	t.Parallel()

	prev := Snapshot{Files: map[string]string{
		"keep.go":   "same",
		"change.go": "old",
		"gone.go":   "old",
	}}
	current := Snapshot{Files: map[string]string{
		"keep.go":   "same",
		"change.go": "new",
		"added.go":  "fresh",
	}}
	diff := DiffSnapshots(prev, current)
	if !reflect.DeepEqual(diff.Added, []string{"added.go"}) {
		t.Fatalf("added = %#v", diff.Added)
	}
	if !reflect.DeepEqual(diff.Modified, []string{"change.go"}) {
		t.Fatalf("modified = %#v", diff.Modified)
	}
	if !reflect.DeepEqual(diff.Removed, []string{"gone.go"}) {
		t.Fatalf("removed = %#v", diff.Removed)
	}
}

func TestDiffSnapshotsBothEmpty(t *testing.T) {
	t.Parallel()

	diff := DiffSnapshots(Snapshot{Files: map[string]string{}}, Snapshot{Files: map[string]string{}})
	if !diff.Empty() {
		t.Fatalf("diff = %#v", diff)
	}
}
