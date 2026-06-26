package daemon

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
)

type filesetFixtureEntry struct {
	relativePath string
	wantCaptured bool
	wantSkipped  bool
	wantRemoved  bool
}

func TestMerkleCaptureConvergesAndAgreesWithIndexerFileSet(t *testing.T) {
	t.Setenv("INDEX_MAX_FILE_BYTES", "64")

	root := t.TempDir()
	entries := writeFilesetFixture(t, root)
	config := model.IndexConfig{
		SplitterType:       "langchain",
		SplitterChunkSize:  1000,
		SplitterOverlap:    200,
		Extensions:         nil,
		IgnorePatterns:     nil,
		IgnoreDigest:       "",
		EmbeddingProvider:  "",
		EmbeddingModel:     "",
		EmbeddingDimension: 0,
		VectorBackend:      "",
		Hybrid:             false,
	}

	first, _, err := merkle.Capture(context.Background(), root, config)
	if err != nil {
		t.Fatalf("first Capture returned error: %v", err)
	}
	second, _, err := merkle.Capture(context.Background(), root, config)
	if err != nil {
		t.Fatalf("second Capture returned error: %v", err)
	}

	if !merkle.Equal(first, second) {
		t.Fatalf("successive captures differ: first=%#v second=%#v", first.Files, second.Files)
	}
	diff := merkle.DiffSnapshots(first, second)
	if !diff.Empty() {
		t.Fatalf("DiffSnapshots(first, second) = %#v, want empty", diff)
	}

	wantSnapshotPaths := []string{"nested/nested.go", "small.go"}
	gotSnapshotPaths := sortedSnapshotPaths(first)
	if !reflect.DeepEqual(gotSnapshotPaths, wantSnapshotPaths) {
		t.Fatalf("captured paths = %#v, want %#v", gotSnapshotPaths, wantSnapshotPaths)
	}

	runner := indexer.NewRunner()
	for _, entry := range entries {
		result, indexErr := runner.IndexOne(context.Background(), root, entry.relativePath, config)
		if indexErr != nil {
			t.Fatalf("IndexOne(%q) returned error: %v", entry.relativePath, indexErr)
		}
		gotCaptured := first.HasFile(entry.relativePath)
		gotKeptByIndexer := !result.Skipped && !result.Removed
		if gotCaptured != gotKeptByIndexer {
			t.Fatalf(
				"Capture and IndexOne disagree for %q: captured=%t skipped=%t removed=%t hash=%q",
				entry.relativePath,
				gotCaptured,
				result.Skipped,
				result.Removed,
				result.FileHash,
			)
		}
		if gotCaptured != entry.wantCaptured {
			t.Fatalf("Capture HasFile(%q) = %t, want %t", entry.relativePath, gotCaptured, entry.wantCaptured)
		}
		if result.Skipped != entry.wantSkipped {
			t.Fatalf("IndexOne(%q).Skipped = %t, want %t", entry.relativePath, result.Skipped, entry.wantSkipped)
		}
		if result.Removed != entry.wantRemoved {
			t.Fatalf("IndexOne(%q).Removed = %t, want %t", entry.relativePath, result.Removed, entry.wantRemoved)
		}
		if entry.wantCaptured && (result.FileHash == "" || len(result.Chunks) == 0) {
			t.Fatalf("IndexOne(%q) kept the file but returned hash=%q chunks=%d", entry.relativePath, result.FileHash, len(result.Chunks))
		}
		t.Logf(
			"fileset fixture path=%s captured=%t skipped=%t removed=%t",
			entry.relativePath,
			gotCaptured,
			result.Skipped,
			result.Removed,
		)
	}

	// The full bootstrap walk (Runner.Index) must agree with Capture too: it skips
	// the oversize, binary, directory, and symlink-to-directory entries without a
	// fatal error and indexes exactly the regular files Capture kept.
	walkResult, walkErr := runner.Index(context.Background(), root, config, nil)
	if walkErr != nil {
		t.Fatalf("Runner.Index walk returned error over the fixture: %v", walkErr)
	}
	if int(walkResult.IndexedFiles) != len(wantSnapshotPaths) {
		t.Fatalf("Runner.Index IndexedFiles = %d, want %d (the captured regular files)", walkResult.IndexedFiles, len(wantSnapshotPaths))
	}
	if walkResult.SkippedOversize < 1 {
		t.Fatalf("Runner.Index SkippedOversize = %d, want >= 1 (oversize.go)", walkResult.SkippedOversize)
	}
}

// TestMerkleCaptureExcludesIgnoredFilesAndConverges proves an ignored file is
// kept out of the snapshot and that the tree still converges on a second
// capture, so an ignore rule cannot reintroduce the re-sync loop.
func TestMerkleCaptureExcludesIgnoredFilesAndConverges(t *testing.T) {
	root := t.TempDir()
	writeFixtureFile(t, root, "keep.go", []byte("package p\nfunc keep() {}\n"))
	writeFixtureFile(t, root, "skip.log", []byte("log noise\n"))
	config := model.IndexConfig{
		SplitterType:       "langchain",
		SplitterChunkSize:  1000,
		SplitterOverlap:    200,
		Extensions:         nil,
		IgnorePatterns:     []string{"*.log"},
		IgnoreDigest:       "",
		EmbeddingProvider:  "",
		EmbeddingModel:     "",
		EmbeddingDimension: 0,
		VectorBackend:      "",
		Hybrid:             false,
	}

	first, _, err := merkle.Capture(context.Background(), root, config)
	if err != nil {
		t.Fatalf("first Capture returned error: %v", err)
	}
	if first.HasFile("skip.log") {
		t.Fatalf("ignored file skip.log was captured into the snapshot")
	}
	if !first.HasFile("keep.go") {
		t.Fatalf("keep.go was not captured")
	}
	second, _, err := merkle.Capture(context.Background(), root, config)
	if err != nil {
		t.Fatalf("second Capture returned error: %v", err)
	}
	if diff := merkle.DiffSnapshots(first, second); !diff.Empty() {
		t.Fatalf("tree with an ignored file did not converge: %#v", diff)
	}
}

func writeFilesetFixture(t *testing.T, root string) []filesetFixtureEntry {
	t.Helper()

	writeFixtureFile(t, root, "small.go", []byte("package p\nfunc small() {}\n"))
	writeFixtureFile(t, root, "oversize.go", []byte(strings.Repeat("a", 65)))
	writeFixtureFile(t, root, "invalid.go", []byte{'p', 'a', 'c', 'k', 'a', 'g', 'e', ' ', 0xff, '\n'})
	writeFixtureFile(t, root, "data.bin", []byte{0x00, 0x01, 0xff, 0xfe, 0x00, 'x'})

	nestedDir := filepath.Join(root, "nested")
	if err := os.Mkdir(nestedDir, 0o755); err != nil {
		t.Fatalf("Mkdir(%q) returned error: %v", nestedDir, err)
	}
	writeFixtureFile(t, root, "nested/nested.go", []byte("package nested\nfunc ok() {}\n"))

	entries := []filesetFixtureEntry{
		{relativePath: "small.go", wantCaptured: true, wantSkipped: false, wantRemoved: false},
		{relativePath: "oversize.go", wantCaptured: false, wantSkipped: true, wantRemoved: false},
		{relativePath: "invalid.go", wantCaptured: false, wantSkipped: true, wantRemoved: false},
		{relativePath: "data.bin", wantCaptured: false, wantSkipped: true, wantRemoved: false},
		{relativePath: "nested", wantCaptured: false, wantSkipped: false, wantRemoved: true},
		{relativePath: "nested/nested.go", wantCaptured: true, wantSkipped: false, wantRemoved: false},
	}

	symlinkPath := filepath.Join(root, "linked-dir.go")
	if err := os.Symlink(nestedDir, symlinkPath); err != nil {
		t.Logf("skipping symlink-to-directory assertion because Symlink returned error: %v", err)
		return entries
	}
	entries = append(entries, filesetFixtureEntry{
		relativePath: "linked-dir.go",
		wantCaptured: false,
		wantSkipped:  false,
		wantRemoved:  true,
	})
	return entries
}

func writeFixtureFile(t *testing.T, root string, relativePath string, content []byte) {
	t.Helper()
	path := filepath.Join(root, relativePath)
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("WriteFile(%q) returned error: %v", relativePath, err)
	}
}

func sortedSnapshotPaths(snapshot merkle.Snapshot) []string {
	paths := make([]string, 0, len(snapshot.Files))
	for path := range snapshot.Files {
		paths = append(paths, path)
	}
	slices.Sort(paths)
	return paths
}
