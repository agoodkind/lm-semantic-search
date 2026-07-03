// Package view holds the typed view models the render layer formats. Every
// field is plain data already resolved at the daemon boundary, so a renderer
// never decides presentation from a raw record. The render layer imports this
// package and never internal/model; that import wall is the choke point.
package view

// Display is the resolved codebase presentation status word.
type Display string

// JobSurface is the resolved presentation of one job.
type JobSurface struct {
	StateLabel        string
	ErrorLine         string
	Superseded        bool
	SupersededByJobID string
}

// FailureSurface is the resolved failure detail for a codebase.
type FailureSurface struct {
	HasFailure    bool
	Message       string
	FailedAtLabel string
	JobID         string
	TraceID       string
}

// QuarantineSurface is the resolved detail for a codebase whose destructive
// sync is paused after a suspicious large disappearance.
type QuarantineSurface struct {
	HasQuarantine      bool
	Reason             string
	FirstObservedLabel string
	LastObservedLabel  string
	ObservationCount   int32
	MissingCount       int32
	TotalCount         int32
	Trigger            string
}

// StatusNarrative is the boundary-owned, display-ready body for a non-template
// codebase status (failed, missing, stale, quarantined). The daemon boundary
// builds each line so the render layer only joins them; render never synthesizes
// status prose from a raw record. The state itself is carried by
// GetIndexView.Display, so the narrative holds only the pre-rendered lines.
type StatusNarrative struct {
	Lines []string
}

// RunMode names what kind of pass a job is making.
type RunMode string

// RunMode values.
const (
	RunModeFirstBuild    RunMode = "first_build"
	RunModeChanged       RunMode = "changed"
	RunModeForcedReindex RunMode = "forced_reindex"
	RunModeResuming      RunMode = "resuming"
)

// OutcomeKind is the semantic identity of one outcome row. It carries no
// presentation: the render layer maps a kind to its glyph and label, so the
// kind is the single wire-safe representation shared by text, the TUI, and JSON.
type OutcomeKind string

// Outcome kinds, in the fixed display order ResolveBreakdown emits them.
const (
	KindEmbedded   OutcomeKind = "embedded"
	KindUnchanged  OutcomeKind = "unchanged"
	KindRemoved    OutcomeKind = "removed"
	KindPending    OutcomeKind = "pending"
	KindOversize   OutcomeKind = "oversize"
	KindUnreadable OutcomeKind = "unreadable"
	KindAdded      OutcomeKind = "added"
	KindReused     OutcomeKind = "reused"
)

// OutcomeRow is one child line in an outcome tree: a semantic kind and a count.
// The render layer derives the glyph and label from the kind. Its fields are
// unexported so a row cannot be hand-assembled as a literal outside this
// package; NewOutcomeRow is the only constructor, which keeps the breakdown
// vocabulary funnelled through ResolveBreakdown and pbconv.
type OutcomeRow struct {
	kind  OutcomeKind
	count int32
}

// NewOutcomeRow builds one outcome row. It is the only way to construct a row
// outside this package (the proto rebuild in pbconv uses it).
func NewOutcomeRow(kind OutcomeKind, count int32) OutcomeRow {
	return OutcomeRow{kind: kind, count: count}
}

// Kind returns the row's semantic kind.
func (row OutcomeRow) Kind() OutcomeKind {
	return row.kind
}

// Count returns the row's count.
func (row OutcomeRow) Count() int32 {
	return row.count
}

// OutcomeBreakdown is the resolved file-and-chunk outcome tree shared by every
// status surface. It is built once at the daemon boundary (resolveOutcomeBreakdown)
// so the compact job views and the codebase status templates render a
// byte-identical tree, which is the structural guard against diverging status
// vocabularies. The file rows always sum to Processed, and a zero bucket is
// omitted, except the embedded file row and the added chunk row, which always
// render.
type OutcomeBreakdown struct {
	// ScopeLabel types the denominator, for example "changed files" or
	// "files (full build)".
	ScopeLabel string
	// Processed is the N in "N of M ... processed"; it equals the sum of the
	// FileRows counts.
	Processed int32
	// ScopeTotal is the M denominator.
	ScopeTotal int32
	// FileRows are the per-outcome file children in fixed order: embedded,
	// unchanged, removed, pending, too-large, error. Empty when no scope is
	// measured yet.
	FileRows []OutcomeRow
	// ChunksTotal is the whole-collection chunk count for the chunk tree header.
	ChunksTotal int32
	// ChunkRows are the chunk children: added always, reused on reuse-capable
	// passes and on first builds that loaded or served reuse. Empty when there
	// is no chunk activity to report.
	ChunkRows []OutcomeRow
}

// ProgressCounts is the raw counter input to ResolveBreakdown. It is a plain
// struct, not a model type, so internal/view keeps its no-model import wall
// while still owning the one resolver.
type ProgressCounts struct {
	RunMode                string
	Unit                   string
	FilesTotal             int32
	FilesProcessed         int32
	FilesAdded             int32
	FilesModified          int32
	FilesRemoved           int32
	FilesEmbedded          int32
	FilesSkippedOversize   int32
	FilesSkippedUnreadable int32
	FilesPending           int32
	ChunksTotal            int32
	ChunksProcessed        int32
	ChunksReused           int32
	ChunksEmbedded         int32
	ChunksGenerated        int32
	ReuseVectorsLoaded     int32
}

// ResolveBreakdown is the single source of truth for the outcome tree. Every
// surface (CLI text, the TUI, the status templates, and the wire breakdown)
// projects from the value it returns, so a status can never diverge between
// commands. The file rows partition the processed set and always sum to
// Processed; a zero bucket is omitted except embedded and the added chunk row.
func ResolveBreakdown(counts ProgressCounts) OutcomeBreakdown {
	unit := counts.Unit
	if unit == "" {
		unit = "file"
	}
	embedded := counts.FilesEmbedded
	oversize := counts.FilesSkippedOversize
	unreadable := counts.FilesSkippedUnreadable
	pending := counts.FilesPending
	removed := counts.FilesRemoved
	unchanged := max(counts.FilesProcessed-embedded-oversize-unreadable-pending, 0)

	// The changed set is known from the diff before the embed loop reports a
	// FilesTotal, so the denominator and the scope gate fold it in. This keeps
	// the file tree showing "0 of N processed" during the pre-embed window.
	changedSet := counts.FilesAdded + counts.FilesModified + counts.FilesRemoved
	hasFileScope := counts.FilesTotal > 0 || counts.FilesProcessed > 0 || removed > 0 || changedSet > 0

	processed := embedded + unchanged + removed + pending + oversize + unreadable
	scopeTotal := max(counts.FilesTotal+removed, changedSet)
	chunksEmbedded := counts.ChunksEmbedded
	if chunksEmbedded == 0 && counts.ChunksGenerated > 0 {
		chunksEmbedded = counts.ChunksGenerated
	}
	chunksProcessed := max(counts.ChunksProcessed, counts.ChunksReused+chunksEmbedded)
	chunksTotal := max(counts.ChunksTotal, chunksProcessed)
	reuse := resolveReusePresentation(counts.RunMode, counts.ChunksReused, counts.ReuseVectorsLoaded)
	hasChunks := hasFileScope || chunksTotal > 0 || chunksProcessed > 0 || reuse.hasReuse

	return OutcomeBreakdown{
		ScopeLabel:  scopeLabelForRunMode(counts.RunMode, unit, scopeTotal, reuse.seededFirstBuild),
		Processed:   processed,
		ScopeTotal:  scopeTotal,
		FileRows:    breakdownFileRows(hasFileScope, embedded, unchanged, removed, pending, oversize, unreadable),
		ChunksTotal: chunksTotal,
		ChunkRows:   breakdownChunkRows(hasChunks, chunksEmbedded, counts.ChunksReused, reuse.showReuseRow),
	}
}

// breakdownFileRows builds the file children in fixed order, omitting a zero
// bucket except embedded, which always renders.
func breakdownFileRows(hasScope bool, embedded, unchanged, removed, pending, oversize, unreadable int32) []OutcomeRow {
	if !hasScope {
		return nil
	}
	rows := []OutcomeRow{NewOutcomeRow(KindEmbedded, embedded)}
	if unchanged > 0 {
		rows = append(rows, NewOutcomeRow(KindUnchanged, unchanged))
	}
	if removed > 0 {
		rows = append(rows, NewOutcomeRow(KindRemoved, removed))
	}
	if pending > 0 {
		rows = append(rows, NewOutcomeRow(KindPending, pending))
	}
	if oversize > 0 {
		rows = append(rows, NewOutcomeRow(KindOversize, oversize))
	}
	if unreadable > 0 {
		rows = append(rows, NewOutcomeRow(KindUnreadable, unreadable))
	}
	return rows
}

// breakdownChunkRows builds the chunk children: added always, reused when the
// run mode or reuse counters make that bucket meaningful.
func breakdownChunkRows(hasChunks bool, added, reused int32, showReused bool) []OutcomeRow {
	if !hasChunks {
		return nil
	}
	rows := []OutcomeRow{NewOutcomeRow(KindAdded, added)}
	if showReused {
		rows = append(rows, NewOutcomeRow(KindReused, reused))
	}
	return rows
}

// ZeroBreakdown returns an empty breakdown. It is the one blank constructor, so
// no caller hand-writes an OutcomeBreakdown literal for the empty case and the
// construction-site guard can lock every literal to this package and pbconv.
func ZeroBreakdown() OutcomeBreakdown {
	return OutcomeBreakdown{
		ScopeLabel:  "",
		Processed:   0,
		ScopeTotal:  0,
		FileRows:    nil,
		ChunksTotal: 0,
		ChunkRows:   nil,
	}
}

type reusePresentation struct {
	hasReuse         bool
	seededFirstBuild bool
	showReuseRow     bool
}

// resolveReusePresentation reports how reuse should appear in the shared
// breakdown. A first build can now load sibling vectors, so first-build reuse is
// driven by the counters instead of the run mode alone.
func resolveReusePresentation(runMode string, chunksReused int32, reuseVectorsLoaded int32) reusePresentation {
	hasReuse := chunksReused > 0 || reuseVectorsLoaded > 0
	switch RunMode(runMode) {
	case RunModeChanged, RunModeResuming, RunModeForcedReindex:
		return reusePresentation{
			hasReuse:         hasReuse,
			seededFirstBuild: false,
			showReuseRow:     true,
		}
	case RunModeFirstBuild:
		return reusePresentation{
			hasReuse:         hasReuse,
			seededFirstBuild: reuseVectorsLoaded > 0,
			showReuseRow:     hasReuse,
		}
	default:
		return reusePresentation{
			hasReuse:         hasReuse,
			seededFirstBuild: false,
			showReuseRow:     hasReuse,
		}
	}
}

// scopeLabelForRunMode types the "N of M ..." denominator from the run mode and
// unit, for example "changed files" or "documents (full build)".
func scopeLabelForRunMode(runMode string, unit string, total int32, seededFirstBuild bool) string {
	plural := unit
	if total != 1 {
		plural = unit + "s"
	}
	switch RunMode(runMode) {
	case RunModeFirstBuild:
		if seededFirstBuild {
			return plural + " (first build, reusing prior vectors)"
		}
		return plural + " (full build)"
	case RunModeForcedReindex:
		return plural + " (forced reindex)"
	case RunModeResuming, RunModeChanged:
		return "changed " + plural
	default:
		return plural
	}
}

// ProgressSurface is the resolved progress view for the compact job surfaces.
type ProgressSurface struct {
	// Heading names the pass in plain words, empty for terminal entries.
	Heading string
	// Breakdown is the shared file-and-chunk outcome tree, rendered identically
	// here and in the codebase status templates.
	Breakdown OutcomeBreakdown
	// ScopeLine is the classification line with its own unit, for example
	// "Changed since last sync: 1,004 conversations added · 7 modified".
	// Empty when the run classified nothing.
	ScopeLine string
	// PercentLabel is the progress figure ("23.5%") or the preparing label
	// when the scope is not measured yet.
	PercentLabel string
}

// TimingView is the resolved timing block for a job.
type TimingView struct {
	StartedLabel   string
	UpdatedLabel   string
	CompletedLabel string
	DurationLabel  string
	DurationWord   string
}

// JobEntryView is one fully resolved job for the list and detail surfaces.
type JobEntryView struct {
	ID            string
	CanonicalPath string
	Operation     string
	PhaseLabel    string
	Surface       JobSurface
	Progress      ProgressSurface
	Timing        TimingView
}

// ListSummary is the resolved job list tally.
type ListSummary struct {
	Total      int
	Queued     int
	Running    int
	Canceling  int
	Completed  int
	Failed     int
	Superseded int
	Canceled   int
}

// StatusView is the codebase status template view.
type StatusView struct {
	Name            string
	UpdatedAt       string
	PrepareLabel    string
	WaitLabel       string
	Heading         string
	Percent         int32
	FilesInCodebase int32
	FilesChanged    int32
	FilesUnchanged  int32
	// Breakdown is the shared outcome tree; the status template emits it as one
	// block via BreakdownBlock, so it stays identical to the compact job view.
	Breakdown OutcomeBreakdown
	// ReuseForecastLine is the pre-rendered reuse forecast for a discovered
	// (not-yet-built) worktree, for example "reuses embeddings from 1 indexed
	// sibling worktree". Empty for every other state. It is built from git
	// topology and the registry with no vector-store call, so the status read that
	// produces it stays cheap.
	ReuseForecastLine string
	Files             int32
	Chunks            int32
	SkippedLine       string
	SyncNote          string
	GraphLine         string
	HasStats          bool
}

// BannerView is the dependency health banner.
type BannerView struct {
	Headline string
	Detail   string
}

// SearchResultView is one reduced search hit.
type SearchResultView struct {
	RelativePath string
	StartLine    int32
	EndLine      int32
	Language     string
	Score        float64
	Content      string
}

// SearchView is the code search response view.
type SearchView struct {
	RequestedPath          string
	Query                  string
	CodebaseName           string
	CodebasePath           string
	Results                []SearchResultView
	StateNote              string
	InFlight               bool
	InFlightStatus         StatusView
	InFlightTemplateName   string
	InFlightPercent        int32
	InFlightBackgroundSync bool
	Degraded               bool
	ResolutionLines        []string
}

// ConversationSearchView is the conversation search response view.
type ConversationSearchView struct {
	CollectionID string
	Query        string
	Results      []ConversationResultView
	StateNote    string
}

// ConversationResultView is one reduced conversation hit.
type ConversationResultView struct {
	ConversationID string
	MessageIndex   int32
	Role           string
	TimestampUnix  int64
	Score          float64
	Content        string
}

// GetIndexView is the resolved codebase status response.
type GetIndexView struct {
	Tracked            bool
	RequestedPath      string
	CanonicalPath      string
	Display            Display
	TemplateName       string
	Status             StatusView
	Failure            FailureSurface
	Quarantine         QuarantineSurface
	Narrative          StatusNarrative
	WaitLabel          string
	ClassificationLine string
	ResolutionLines    []string
	CoverageLine       string
	DescendantsHint    string
	SyncNote           string
}

// StartIndexView is the start acknowledgment.
type StartIndexView struct {
	RequestedPath      string
	CanonicalPath      string
	CodebaseID         string
	JobID              string
	SplitterType       string
	Deduplicated       bool
	OverlapsCodebaseID string
	MergeNote          string
}

// MutationAckView covers clear, cancel, sync, and conversation acks. Exactly
// one Kind renders per call.
type MutationAckView struct {
	Kind            string
	Path            string
	JobID           string
	StateLabel      string
	AlreadyTerminal bool
	Deduplicated    bool
	CollectionID    string
	CollectionName  string
	CodebaseID      string
	ConversationID  string
	DocumentCount   int
	NeededCount     int
	TotalCount      int
}

// MutationAckView kinds.
const (
	AckClear                = "clear"
	AckCancel               = "cancel"
	AckSync                 = "sync"
	AckRegisterConversation = "register_conversation"
	AckUpsertConversation   = "upsert_conversation"
	AckDeleteConversation   = "delete_conversation"
	AckManifest             = "manifest"
)

// DoctorView is the doctor response view.
type DoctorView struct {
	Diagnostics []string
	Dropped     []string
	Quarantined []string
}

// CodebaseRowView is one row of the codebase list.
type CodebaseRowView struct {
	ID            string
	CanonicalPath string
	Display       Display
	// ReuseSiblingCount surfaces a discovered worktree's reuse forecast in the
	// list, so a deferred build reads as cheap rather than a blank pending row. It
	// is zero for codebases that are not discovered worktrees.
	ReuseSiblingCount int32
	// Active reports whether the codebase has a live indexing job, so the list
	// renders its breakdown tree inline. Breakdown is the same value the status
	// surface shows; it is empty when Active is false.
	Active    bool
	Breakdown OutcomeBreakdown
}
