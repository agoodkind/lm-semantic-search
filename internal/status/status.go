package status

import "goodkind.io/lm-semantic-search/internal/model"

// Display is the user-facing status a codebase presents. It is derived, never
// persisted: the registry keeps the lifecycle model.CodebaseStatus, and this
// adds the presentation folds (live job phase, degraded dependency) on top.
type Display string

// Display values.
const (
	DisplayPreparing Display = "preparing"
	DisplayIndexing  Display = "indexing"
	DisplayIndexed   Display = "indexed"
	DisplayWaiting   Display = "waiting"
	DisplayStale     Display = "stale"
	DisplayFailed    Display = "failed"
	DisplayMissing   Display = "missing"
)

// DependencyMode names a degraded shared-dependency condition. The empty mode is
// healthy. Each non-empty mode selects one banner variant.
type DependencyMode string

// DependencyMode values.
const (
	Healthy             DependencyMode = ""
	EmbedderUnreachable DependencyMode = "embedder_unreachable"
	EmbedderRejected    DependencyMode = "embedder_rejected"
	EmbedderBusy        DependencyMode = "embedder_busy"
	StoreUnavailable    DependencyMode = "store_unavailable"
)

// Degraded reports whether the mode is any non-healthy condition.
func (mode DependencyMode) Degraded() bool {
	return mode != Healthy
}

// SearchOutcome names the resolved shape of a search call so the state note is
// chosen from one vocabulary rather than written inline at the call site.
type SearchOutcome string

// SearchOutcome values.
const (
	SearchNone      SearchOutcome = ""
	SearchOK        SearchOutcome = "ok"
	SearchEmpty     SearchOutcome = "empty"
	SearchRepairing SearchOutcome = "repairing"
	SearchNotReady  SearchOutcome = "not_ready"
	SearchMissing   SearchOutcome = "missing"
	SearchFallback  SearchOutcome = "fallback"
)

// Inputs is the normalized snapshot the daemon builds for a codebase. It carries
// only the facts the calculation needs, already reduced from the raw records, so
// the resolver stays a pure function the table tests can exhaust.
type Inputs struct {
	// Status is the persisted codebase lifecycle status.
	Status model.CodebaseStatus
	// HasActiveJob reports whether a live job owns this codebase right now.
	HasActiveJob bool
	// JobScopeKnown reports whether the live job has measured its file scope, so
	// it reads as indexing rather than preparing.
	JobScopeKnown bool
	// BackgroundSyncReconcile reports whether the live job is a background sync
	// over an already-indexed codebase, which keeps reading indexed.
	BackgroundSyncReconcile bool
	// Dependency is the daemon's shared-dependency health mode.
	Dependency DependencyMode
	// Search is the resolved search outcome, or SearchNone outside a search call.
	Search SearchOutcome
}

// Surface is the fully resolved view model. Every field is decided here so the
// render layer only formats them; no renderer re-inspects the raw records.
type Surface struct {
	// Display is the resolved presentation status.
	Display Display
	// Glyph is the shape token for Display.
	Glyph string
	// Label is the human word for Display.
	Label string
	// BannerPresent reports whether a degraded-dependency banner should show.
	BannerPresent bool
	// BannerHeadline is the one-line headline for the degraded mode, empty when
	// healthy.
	BannerHeadline string
	// StateNote is the read-only search note for the outcome, empty when none.
	StateNote string
}

// displayRule is one row of the declarative display table: a predicate over the
// normalized inputs and the display it yields. The first matching rule wins.
type displayRule struct {
	when func(Inputs) bool
	then Display
}

// displayRules is the ordered base-display decision table. It is read top to
// bottom; the first rule whose predicate holds decides the base display. The
// live-job rules come first because a live job overrides the persisted status.
// A codebase that matches no rule (NotIndexed or Indexing with no live job, an
// interrupted build the background pass re-queues) defaults to preparing.
var displayRules = []displayRule{
	{func(in Inputs) bool { return in.HasActiveJob && in.BackgroundSyncReconcile }, DisplayIndexed},
	{func(in Inputs) bool { return in.HasActiveJob && in.JobScopeKnown }, DisplayIndexing},
	{func(in Inputs) bool { return in.HasActiveJob }, DisplayPreparing},
	{func(in Inputs) bool { return in.Status == model.CodebaseStatusIndexed }, DisplayIndexed},
	{func(in Inputs) bool { return in.Status == model.CodebaseStatusStale }, DisplayStale},
	{func(in Inputs) bool { return in.Status == model.CodebaseStatusFailed }, DisplayFailed},
	{func(in Inputs) bool { return in.Status == model.CodebaseStatusMissing }, DisplayMissing},
}

// baseDisplay applies the display table and falls back to preparing.
func baseDisplay(in Inputs) Display {
	for _, rule := range displayRules {
		if rule.when(in) {
			return rule.then
		}
	}
	return DisplayPreparing
}

// ResolveDisplay computes the display status, applying the degraded-dependency
// fold on top of the base table. When a shared dependency is degraded a codebase
// that is not embedding right now cannot be searched, so a base of preparing or
// indexed folds to waiting; the banner names the specific cause. A live scoped
// job stays indexing because it is embedding now, and stale, failed, or missing
// stay as they are because the dependency does not change a local terminal state.
func ResolveDisplay(in Inputs) Display {
	base := baseDisplay(in)
	if in.Dependency.Degraded() && (base == DisplayPreparing || base == DisplayIndexed) {
		return DisplayWaiting
	}
	return base
}

// Resolve turns the normalized inputs into the fully resolved surface.
func Resolve(in Inputs) Surface {
	display := ResolveDisplay(in)
	return Surface{
		Display:        display,
		Glyph:          GlyphFor(display),
		Label:          LabelFor(display),
		BannerPresent:  in.Dependency.Degraded(),
		BannerHeadline: BannerHeadlineFor(in.Dependency),
		StateNote:      StateNoteFor(in.Search),
	}
}
