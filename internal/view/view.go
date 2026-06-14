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

// RunMode names what kind of pass a job is making.
type RunMode string

// RunMode values.
const (
	RunModeFirstBuild    RunMode = "first_build"
	RunModeChanged       RunMode = "changed"
	RunModeForcedReindex RunMode = "forced_reindex"
	RunModeResuming      RunMode = "resuming"
)

// OutcomeRow is one already-resolved child line in an outcome tree: a glyph, a
// count, and a label. The renderer joins them under a tree connector. The lead
// glyph carries the status before the count: ➕/⏭️ normal, 🗑️ removed,
// ⏳ pending (transient, will retry), 📏 skipped (deliberate policy), ⚠️ error.
type OutcomeRow struct {
	Glyph string
	Count int32
	Label string
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
	// ChunkRows are the chunk children: added always, reused on a reuse-capable
	// pass (shown even at zero), omitted for a first build. Empty when there is
	// no chunk activity to report.
	ChunkRows []OutcomeRow
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
	Breakdown   OutcomeBreakdown
	Files       int32
	Chunks      int32
	SkippedLine string
	SyncNote    string
	HasStats    bool
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
}

// CodebaseRowView is one row of the codebase list.
type CodebaseRowView struct {
	ID            string
	CanonicalPath string
	Display       Display
}
