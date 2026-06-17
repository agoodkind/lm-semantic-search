package status

import (
	"strings"

	"goodkind.io/lm-semantic-search/internal/model"
)

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
	// DisplayDiscovered is a worktree registered by a read but not yet built. It
	// has no active job and is not searchable yet; the deferred build will move it
	// to indexing. It reads distinctly from indexing (which implies live work) and
	// from not_indexed (which the SOT never emits).
	DisplayDiscovered Display = "discovered"
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
	// SearchableEligible reports whether the queried path is in-scope indexed, so
	// search could serve it when the backend is up. It is the per-path precondition
	// the searchable fold combines with the dependency health.
	SearchableEligible bool
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
	// Searchable reports whether a search can serve the path right now: it is
	// in-scope indexed and the shared backend is not degraded. The wire
	// `searchable` field and any "ready to search" surface read this one value.
	Searchable bool
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
	{func(in Inputs) bool { return in.Status == model.CodebaseStatusDiscovered }, DisplayDiscovered},
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

// ResolveSearchable reports whether a path can serve a search right now. A path
// is searchable only when it is in-scope indexed (SearchableEligible) AND the
// shared backend is not degraded, so a store or embedder outage flips it false
// even while the on-disk classification stays indexed. This is the single place
// the searchable fold lives, so the wire `searchable` field and the display
// status both derive from one resolution and cannot disagree.
func ResolveSearchable(in Inputs) bool {
	return in.SearchableEligible && !in.Dependency.Degraded()
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
		Searchable:     ResolveSearchable(in),
	}
}

// jobStateWordByState maps a persisted job state to its human word. The
// cancelling and cancelled states use the American spellings the surfaces show.
var jobStateWordByState = map[model.JobState]string{
	model.JobStateQueued:     "queued",
	model.JobStateRunning:    "running",
	model.JobStateCancelling: "canceling",
	model.JobStateCompleted:  "completed",
	model.JobStateFailed:     "failed",
	model.JobStateCancelled:  "canceled",
}

// JobStateLabelFor returns the human word for a job state, falling back to the
// raw token for an unrecognized value. It is the only place a job state becomes
// a word, so every job surface reads the same vocabulary.
func JobStateLabelFor(state model.JobState) string {
	if word, ok := jobStateWordByState[state]; ok {
		return word
	}
	return string(state)
}

// JobInputs is the normalized snapshot for one job's presentation. Like Inputs
// for a codebase, it carries only the facts the resolver needs, already reduced
// from the raw record, so the render layer never inspects a model.Job to decide
// how the job reads.
type JobInputs struct {
	// State is the persisted job state.
	State model.JobState
	// Retryable reports whether the job's error is self-healing, which makes a
	// failure read as a transient stop rather than a hard failure.
	Retryable bool
	// ErrorMessage is the sanitized, client-safe message for the job's error, or
	// empty when the job carries no error.
	ErrorMessage string
	// Dependency is the daemon's shared-dependency health mode.
	Dependency DependencyMode
	// SupersededByJobID is the id of the immediate next terminal job for this
	// job's codebase, or empty when this job is the latest. A failed job with a
	// successor is superseded.
	SupersededByJobID string
}

// JobSupersededCountLabel is the summary-tally word for a failed job overtaken by
// a later terminal job, read from the one vocabulary instead of a renderer
// hard-coding the phrase.
const JobSupersededCountLabel = "superseded"

// JobSurface is the fully resolved presentation of one job. Every field is
// decided here so the render layer only formats them; no renderer re-derives a
// state label or an error echo from the raw job record.
type JobSurface struct {
	// StateLabel is the comma-joined tag list for the job: the state word, then
	// "retryable" when the failure is self-healing, then "superseded by <id>"
	// when a later terminal job overtook it.
	StateLabel string
	// ErrorLine is the message a surface shows beneath the job, or empty when the
	// job has no error or the dependency banner already carries the cause.
	ErrorLine string
	// Superseded reports a failed job overtaken by a later terminal job for the
	// same codebase. The job-list summary tallies these apart from current
	// failures.
	Superseded bool
	// SupersededByJobID is the successor job id when Superseded, else empty.
	SupersededByJobID string
}

// ResolveJob turns the normalized job inputs into the resolved surface. The
// state label is a comma-joined tag list: the state word, then "retryable" when
// the error is self-healing, then "superseded by <id>" when a later terminal job
// overtook this failure. A retryable failure that coincides with a degraded
// dependency suppresses the error line, because the banner already names that
// shared-infrastructure cause and a per-job echo would only repeat it; every
// other error still shows.
func ResolveJob(in JobInputs) JobSurface {
	superseded := in.State == model.JobStateFailed && in.SupersededByJobID != ""
	tags := []string{JobStateLabelFor(in.State)}
	if in.Retryable {
		tags = append(tags, "retryable")
	}
	if superseded {
		tags = append(tags, "superseded by "+in.SupersededByJobID)
	}
	errorLine := ""
	if in.ErrorMessage != "" && (!in.Dependency.Degraded() || !in.Retryable) {
		errorLine = in.ErrorMessage
	}
	supersededBy := ""
	if superseded {
		supersededBy = in.SupersededByJobID
	}
	return JobSurface{
		StateLabel:        strings.Join(tags, ", "),
		ErrorLine:         errorLine,
		Superseded:        superseded,
		SupersededByJobID: supersededBy,
	}
}
