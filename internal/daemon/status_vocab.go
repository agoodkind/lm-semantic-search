package daemon

// glyphForDisplay returns the shape token for a display status, so every surface
// renders the same glyph from one place rather than each client owning its own
// map. The glyphs keep the states distinguishable without color, per the
// don't-rely-on-color-alone guideline. The switch is exhaustive over every
// displayStatus; the default is a safe fallback for an unrecognized value.
func glyphForDisplay(status displayStatus) string {
	switch status {
	case displayPreparing:
		return "◌"
	case displayIndexing:
		return "◐"
	case displayWaiting:
		return "⋯"
	case displayIndexed:
		return "●"
	case displayStale:
		return "△"
	case displayFailed:
		return "✗"
	case displayMissing:
		return "⊘"
	default:
		return "•"
	}
}

// labelForDisplay returns the human word for a display status, so the list and
// detail surfaces share one label vocabulary. The switch is exhaustive over
// every displayStatus; the default returns the raw token for an unrecognized
// value.
func labelForDisplay(status displayStatus) string {
	switch status {
	case displayPreparing:
		return "preparing"
	case displayIndexing:
		return "indexing"
	case displayWaiting:
		return "waiting"
	case displayIndexed:
		return "indexed"
	case displayStale:
		return "stale"
	case displayFailed:
		return "failed"
	case displayMissing:
		return "missing"
	default:
		return string(status)
	}
}
