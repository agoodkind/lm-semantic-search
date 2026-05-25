package tssnapshot

import (
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/claude-context-go/internal/model"
)

const sampleSnapshot = `{
  "formatVersion": "v2",
  "codebases": {
    "/Users/example/Sites/repo-alpha": {
      "status": "indexed",
      "indexedFiles": 42,
      "totalChunks": 314,
      "indexStatus": "completed",
      "requestSplitter": "ast",
      "lastUpdated": "2026-05-24T11:28:01.806Z"
    },
    "/Users/example/Sites/repo-beta": {
      "status": "indexfailed",
      "errorMessage": "embed failed",
      "lastAttemptedPercentage": 18,
      "requestSplitter": "ast",
      "lastUpdated": "2026-05-17T22:20:17.897Z"
    }
  },
  "lastUpdated": "2026-05-25T02:14:06.870Z"
}`

func TestLoadReturnsParsedEntries(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "mcp-codebase-snapshot.json")
	if err := os.WriteFile(path, []byte(sampleSnapshot), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	entries, err := Load(path)
	if err != nil {
		t.Fatalf("Load returned error: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("entries count = %d, want 2", len(entries))
	}
	alpha, found := entries["/Users/example/Sites/repo-alpha"]
	if !found {
		t.Fatal("repo-alpha entry missing")
	}
	if alpha.Status != string(StatusIndexed) {
		t.Fatalf("alpha.Status = %q", alpha.Status)
	}
	if alpha.IndexedFiles != 42 || alpha.TotalChunks != 314 {
		t.Fatalf("alpha counts = %d files / %d chunks", alpha.IndexedFiles, alpha.TotalChunks)
	}
}

func TestLoadMissingFileReturnsNil(t *testing.T) {
	t.Parallel()

	entries, err := Load(filepath.Join(t.TempDir(), "does-not-exist.json"))
	if err != nil {
		t.Fatalf("Load returned error for missing file: %v", err)
	}
	if entries != nil {
		t.Fatalf("entries = %#v, want nil for missing file", entries)
	}
}

func TestSynthesizeIndexed(t *testing.T) {
	t.Parallel()

	entry := Entry{
		Status:          string(StatusIndexed),
		IndexedFiles:    877,
		TotalChunks:     9139,
		IndexStatus:     "completed",
		RequestSplitter: "ast",
		LastUpdated:     "2026-05-24T11:28:01.806Z",
	}
	codebase := Synthesize("/Users/agoodkind/Sites/clyde-dev/clyde", entry, true)

	if codebase.Status != model.CodebaseStatusIndexed {
		t.Fatalf("status = %q", codebase.Status)
	}
	if codebase.CollectionName != "hybrid_code_chunks_1827ceba" {
		t.Fatalf("collection name = %q, want hybrid_code_chunks_1827ceba (md5 of clyde path)", codebase.CollectionName)
	}
	if codebase.LastSuccessfulRun == nil {
		t.Fatal("LastSuccessfulRun nil")
	}
	if codebase.LastSuccessfulRun.IndexedFiles != 877 {
		t.Fatalf("indexed files = %d", codebase.LastSuccessfulRun.IndexedFiles)
	}
}

func TestSynthesizeIndexFailed(t *testing.T) {
	t.Parallel()

	entry := Entry{
		Status:                  string(StatusIndexFailed),
		ErrorMessage:            "embed failed",
		LastAttemptedPercentage: 18,
		LastUpdated:             "2026-05-17T22:20:17.897Z",
	}
	codebase := Synthesize("/Users/example/Sites/clyde-research", entry, true)

	if codebase.Status != model.CodebaseStatusFailed {
		t.Fatalf("status = %q", codebase.Status)
	}
	if codebase.LastFailedRun == nil {
		t.Fatal("LastFailedRun nil")
	}
	if codebase.LastFailedRun.Message != "embed failed" {
		t.Fatalf("LastFailedRun.Message = %q", codebase.LastFailedRun.Message)
	}
}

func TestPathHonorsContextRoot(t *testing.T) {
	t.Parallel()

	contextRoot := "/tmp/example/.context"
	got, err := Path(contextRoot)
	if err != nil {
		t.Fatalf("Path returned error: %v", err)
	}
	want := filepath.Join(contextRoot, "mcp-codebase-snapshot.json")
	if got != want {
		t.Fatalf("Path(%q) = %q, want %q", contextRoot, got, want)
	}
}
