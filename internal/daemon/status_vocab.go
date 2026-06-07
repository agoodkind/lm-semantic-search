package daemon

import "goodkind.io/lm-semantic-search/internal/status"

// glyphForDisplay returns the shape token for a display status from the single
// status vocabulary, so every surface renders the same glyph rather than each
// client owning its own map.
func glyphForDisplay(display displayStatus) string {
	return status.GlyphFor(display)
}

// labelForDisplay returns the human word for a display status from the single
// status vocabulary, so the list and detail surfaces share one label set.
func labelForDisplay(display displayStatus) string {
	return status.LabelFor(display)
}
