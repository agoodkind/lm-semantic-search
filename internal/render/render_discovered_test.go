package render

import (
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/view"
)

// TestListIndexesDiscoveredRowShowsReuseForecast proves the codebase list adds a
// reuse-forecast suffix for a discovered worktree with eligible siblings, so the
// row reads as a cheap pending build rather than a blank entry.
func TestListIndexesDiscoveredRowShowsReuseForecast(t *testing.T) {
	t.Parallel()
	out := ListIndexes([]view.CodebaseRowView{{
		ID:                "cb",
		CanonicalPath:     "/x",
		Display:           displayDiscovered,
		ReuseSiblingCount: 2,
	}})
	if !strings.Contains(out, "♻️ reuses 2 sibling collections") {
		t.Fatalf("discovered list row missing the reuse-forecast suffix; got %q", out)
	}
}

// TestListIndexesDiscoveredRowWithoutSiblingsHasNoForecast proves a discovered
// row with no eligible sibling carries no reuse suffix, so the forecast appears
// only when a reuse source exists.
func TestListIndexesDiscoveredRowWithoutSiblingsHasNoForecast(t *testing.T) {
	t.Parallel()
	out := ListIndexes([]view.CodebaseRowView{{
		ID:                "cb",
		CanonicalPath:     "/x",
		Display:           displayDiscovered,
		ReuseSiblingCount: 0,
	}})
	if strings.Contains(out, "reuses") {
		t.Fatalf("discovered list row with no siblings should carry no reuse suffix; got %q", out)
	}
}
