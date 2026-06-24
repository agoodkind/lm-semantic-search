package semantic

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestNoMmapExtraParamOnCreateIndex guards the Milvus 2.6 regression that broke
// every new-collection build. createCollection once passed mmap.enabled into the
// dense AutoIndex create call via WithExtraParam; Milvus 2.6 rejects any extra
// param on AutoIndex with "only metric type can be passed when use AutoIndex".
// mmap on the dense index must be enabled after creation through
// AlterIndexProperties (see mmap.go), never as a create-time index param.
//
// This scans the package source so a reintroduction fails here, in a fast unit
// test, rather than only against a live Milvus instance. It looks for the
// mmap-enable key being passed to a create-index extra param in any
// non-test source file of the package.
func TestNoMmapExtraParamOnCreateIndex(t *testing.T) {
	t.Parallel()

	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}

	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		source, readErr := os.ReadFile(filepath.Clean(name))
		if readErr != nil {
			t.Fatalf("read %s: %v", name, readErr)
		}
		text := string(source)
		// WithExtraParam carries index build params; on an AutoIndex only the
		// metric type is allowed, so the mmap key must never appear as one.
		if strings.Contains(text, "WithExtraParam(mmapEnabledKey") || strings.Contains(text, `WithExtraParam("mmap.enabled"`) {
			t.Fatalf("%s sets mmap as a create-index extra param; Milvus 2.6 rejects mmap on AutoIndex. Enable mmap after creation via AlterIndexProperties (mmap.go) instead.", name)
		}
	}
}
