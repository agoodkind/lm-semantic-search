// Package status is the single source of truth for the daemon's status
// calculation. It turns a normalized [Inputs] snapshot into one resolved
// [Surface] descriptor: the display status, its glyph and label, the
// dependency-health banner headline, and any search state note. Rendering lives
// downstream in the daemon and only formats the fields this package resolves, so
// no surface re-derives status from raw records. The status calculation stays
// independent of the daemon, the gRPC boundary, and the render layer; only
// status.go needs internal/model, for the codebase lifecycle status type.
package status

// glyphByDisplay maps a display status to its shape token. The glyphs keep the
// states distinguishable without color, per the don't-rely-on-color-alone
// guideline.
var glyphByDisplay = map[Display]string{
	DisplayPreparing:  "◌",
	DisplayIndexing:   "◐",
	DisplayWaiting:    "⋯",
	DisplayIndexed:    "●",
	DisplayStale:      "△",
	DisplayFailed:     "✗",
	DisplayMissing:    "⊘",
	DisplayDiscovered: "⊙",
}

// labelByDisplay maps a display status to its human word.
var labelByDisplay = map[Display]string{
	DisplayPreparing:  "preparing",
	DisplayIndexing:   "indexing",
	DisplayWaiting:    "waiting",
	DisplayIndexed:    "indexed",
	DisplayStale:      "stale",
	DisplayFailed:     "failed",
	DisplayMissing:    "missing",
	DisplayDiscovered: "discovered",
}

// bannerHeadlineByMode maps a degraded dependency mode to its one-line banner
// headline. Healthy has no headline. The default generic headline covers any
// future mode that has no explicit entry.
const genericDegradedHeadline = "A shared dependency is degraded. Indexing is paused until it recovers."

var bannerHeadlineByMode = map[DependencyMode]string{
	EmbedderUnreachable: "Embedding server unreachable. Indexing is paused and resumes automatically when it returns.",
	EmbedderRejected:    "Embedding server is rejecting requests. Indexing is paused until the embedding config is fixed.",
	EmbedderBusy:        "Embedding server is at capacity. Indexing is paused and retries automatically when it frees up.",
	StoreUnavailable:    "Vector store unavailable. Search and indexing are paused until it returns.",
}

// stateNoteBySearch maps a search outcome to the read-only note a surface shows
// alongside results when results alone do not explain the state. Most outcomes
// carry no note.
var stateNoteBySearch = map[SearchOutcome]string{
	SearchRepairing: "⚠️ Search is temporarily unavailable because the semantic collection is missing. The daemon is handling automatic rebuild in the background.",
}

// GlyphFor returns the shape token for a display status, with a safe fallback for
// an unrecognized value.
func GlyphFor(display Display) string {
	if glyph, ok := glyphByDisplay[display]; ok {
		return glyph
	}
	return "•"
}

// LabelFor returns the human word for a display status, falling back to the raw
// token for an unrecognized value.
func LabelFor(display Display) string {
	if label, ok := labelByDisplay[display]; ok {
		return label
	}
	return string(display)
}

// BannerHeadlineFor returns the banner headline for a dependency mode. A healthy
// mode returns the empty string; a degraded mode with no explicit entry returns
// the generic headline.
func BannerHeadlineFor(mode DependencyMode) string {
	if !mode.Degraded() {
		return ""
	}
	if headline, ok := bannerHeadlineByMode[mode]; ok {
		return headline
	}
	return genericDegradedHeadline
}

// StateNoteFor returns the read-only search state note for an outcome, or empty
// when the outcome needs none.
func StateNoteFor(outcome SearchOutcome) string {
	return stateNoteBySearch[outcome]
}
