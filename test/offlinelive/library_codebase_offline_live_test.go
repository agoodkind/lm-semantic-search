//go:build offlinelive

package offlinelive

import (
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
)

// TestLibraryCodebaseOfflineSearchUsesTheEmbeddedPool indexes the offline
// fixture with the library codebase store and the real ONNX model. The daemon
// writes the codebase catalog and the exact embedded vector pool under its
// state root, and SearchCode returns the target function from the library.
func TestLibraryCodebaseOfflineSearchUsesTheEmbeddedPool(t *testing.T) {
	t.Setenv("CLAUDE_CONTEXT_CODEBASE_STORE", string(config.CodebaseStoreLibrary))
	harness := newHarness(t)
	if harness.config.CodebaseStore != config.CodebaseStoreLibrary {
		t.Fatalf("codebase store = %q, want %q", harness.config.CodebaseStore, config.CodebaseStoreLibrary)
	}

	requireCompleted(t, harness.indexFixture())

	libraryRoot := filepath.Join(harness.config.StateRoot, "library", "codebase")
	if _, err := os.Stat(filepath.Join(libraryRoot, "catalog.sqlite")); err != nil {
		t.Fatalf("library codebase catalog: %v", err)
	}
	poolEntries, err := os.ReadDir(filepath.Join(libraryRoot, "pool"))
	if err != nil {
		t.Fatalf("read embedded vector pool: %v", err)
	}
	if len(poolEntries) == 0 {
		t.Fatal("the embedded vector pool is empty after indexing")
	}

	searchResponse := harness.search(fixtureQuery, searchResultLimit)
	if !containsTargetResult(searchResponse.GetResults(), targetRelativePath, targetFunctionName) {
		t.Fatalf(
			"library offline search did not return %s from %q in the top %d results:\n%s",
			targetFunctionName,
			targetRelativePath,
			searchResultLimit,
			searchResponse.GetDisplayText(),
		)
	}
}
