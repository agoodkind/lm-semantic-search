package render

import (
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/view"
)

const reuseRowMarker = "♻️"

func TestListIndexesRowShowsDeclaredSource(t *testing.T) {
	t.Parallel()
	out := ListIndexes([]view.CodebaseRowView{{
		ID:                 "cb",
		CanonicalPath:      "/x",
		Display:            displayDiscovered,
		WorktreeSourcePath: "/source/worktree",
	}})
	if !strings.Contains(out, reuseRowMarker) || !strings.Contains(out, "/source/worktree") {
		t.Fatalf("The list row %q must show the reuse marker and the source path.", out)
	}
}

func TestListIndexesRowShowsReuseUnavailableReason(t *testing.T) {
	t.Parallel()
	reasons := []view.WorktreeReuseReason{
		view.WorktreeReuseNoSourceRecorded,
		view.WorktreeReuseSourceNotRegistered,
		view.WorktreeReuseSourceNotIndexed,
		view.WorktreeReuseSourceEmbeddingConfigurationDiffers,
	}
	seen := map[string]view.WorktreeReuseReason{}
	for _, reason := range reasons {
		out := ListIndexes([]view.CodebaseRowView{{
			ID:                     "cb",
			CanonicalPath:          "/x",
			Display:                displayDiscovered,
			WorktreeSourcePath:     "/source/worktree",
			ReuseUnavailableReason: reason,
		}})
		if !strings.Contains(out, reuseRowMarker) {
			t.Fatalf("The list row for reason %q must show the reuse marker in rendered output %q.", reason, out)
		}
		if strings.Contains(out, "/source/worktree") {
			t.Fatalf("The list row for reason %q must omit the source path in rendered output %q.", reason, out)
		}
		if earlier, repeated := seen[out]; repeated {
			t.Fatalf("Reasons %q and %q rendered the same list row %q.", earlier, reason, out)
		}
		seen[out] = reason
	}
}

func TestListIndexesRowWithoutSourceOrReasonHasNoReuseSuffix(t *testing.T) {
	t.Parallel()
	out := ListIndexes([]view.CodebaseRowView{{
		ID:            "cb",
		CanonicalPath: "/x",
		Display:       displayDiscovered,
	}})
	if strings.Contains(out, reuseRowMarker) {
		t.Fatalf("The list row %q must have no reuse suffix.", out)
	}
}
