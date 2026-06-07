package daemon

import "testing"

// TestDisplayVocabularyCoversEveryStatus proves every displayStatus constant has
// a non-empty glyph and label, so a new status cannot be added without giving it
// both, and the CLI and MCP surfaces stay in sync with the daemon vocabulary.
func TestDisplayVocabularyCoversEveryStatus(t *testing.T) {
	t.Parallel()
	statuses := []displayStatus{
		displayPreparing,
		displayIndexing,
		displayWaiting,
		displayIndexed,
		displayStale,
		displayFailed,
		displayMissing,
	}
	for _, status := range statuses {
		if glyph := glyphForDisplay(status); glyph == "" {
			t.Errorf("glyphForDisplay(%q) = empty, want a glyph", status)
		}
		if label := labelForDisplay(status); label == "" {
			t.Errorf("labelForDisplay(%q) = empty, want a label", status)
		}
	}
}

// TestDisplayVocabularyKnownTokens locks the glyph and label for the waiting
// status, the token both clients must render identically.
func TestDisplayVocabularyKnownTokens(t *testing.T) {
	t.Parallel()
	if got := glyphForDisplay(displayWaiting); got != "⋯" {
		t.Errorf("glyphForDisplay(waiting) = %q, want ⋯", got)
	}
	if got := labelForDisplay(displayWaiting); got != "waiting" {
		t.Errorf("labelForDisplay(waiting) = %q, want waiting", got)
	}
}
