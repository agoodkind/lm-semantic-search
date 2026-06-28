package daemon

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/indexability"
	"goodkind.io/lm-semantic-search/internal/model"
)

// TestCodeItemSourceCaptureExcludesIgnoredFiles proves a code source capture
// routes its file set through the indexability resolver: a file a .gitignore
// excludes never reaches the snapshot, while a tracked source file does. The
// source no longer reports a resolved rule tree; the resolver owns matching.
func TestCodeItemSourceCaptureExcludesIgnoredFiles(t *testing.T) {
	// Isolate HOME and the global git config so the developer's real global
	// excludes and ~/.context/.contextignore cannot change the verdict. t.Setenv
	// forbids t.Parallel, so this test runs serially.
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("GIT_CONFIG_GLOBAL", filepath.Join(home, "absent-gitconfig"))
	tempDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(tempDir, ".gitignore"), []byte("skipped/\n"), 0o644); err != nil {
		t.Fatalf("write .gitignore: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(tempDir, "skipped"), 0o755); err != nil {
		t.Fatalf("mkdir skipped: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tempDir, "skipped", "ignored.go"), []byte("package skipped\n"), 0o644); err != nil {
		t.Fatalf("write ignored.go: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tempDir, "kept.go"), []byte("package x\n"), 0o644); err != nil {
		t.Fatalf("write kept.go: %v", err)
	}

	source := newCodeItemSource(nil, indexability.NewResolver(nil), "cb", tempDir, model.IndexConfig{})

	snapshot, err := source.capture(context.Background())
	if err != nil {
		t.Fatalf("capture returned error: %v", err)
	}
	if !snapshot.HasFile("kept.go") {
		t.Fatal("capture dropped the tracked file kept.go")
	}
	if snapshot.HasFile("skipped/ignored.go") {
		t.Fatal("capture kept a gitignored file the resolver should exclude")
	}
}
