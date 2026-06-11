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

// ProgressSurface is the resolved progress view. Every number carries its
// label so a bare unlabeled total can never render.
type ProgressSurface struct {
	// Heading names the pass in plain words, empty for terminal entries.
	Heading string
	// HasScope reports whether the run has measured its work scope.
	HasScope bool
	// Checked of ScopeTotal items walked so far; ScopeLabel types the
	// denominator (for example "changed documents" or "files (full build)").
	Checked    int32
	ScopeTotal int32
	ScopeLabel string
	// CheckVerb is "checked" for a pass that fast-forwards through unchanged
	// work and "embedded" for a pass that embeds everything it walks.
	CheckVerb string
	// Embedded and AlreadyIndexed split Checked into real work and
	// pass-throughs. Shown only when CheckVerb is "checked".
	Embedded       int32
	AlreadyIndexed int32
	// Chunk counts: this run, reused from prior vectors, and the collection
	// total. ChunksInCollection of zero means unknown and the segment is
	// omitted rather than rendered as a shrunken corpus.
	ChunksThisRun      int32
	ChunksReused       int32
	ChunksInCollection int32
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
	Name                   string
	UpdatedAt              string
	PrepareLabel           string
	WaitLabel              string
	Heading                string
	Percent                int32
	FilesProcessed         int32
	FilesTotal             int32
	FilesInCodebase        int32
	FilesChanged           int32
	FilesUnchanged         int32
	FilesReEmbedded        int32
	FilesRemoved           int32
	FilesSkippedOversize   int32
	FilesSkippedUnreadable int32
	FilesProcessedChanged  int32
	ChunksAdded            int32
	ChunksReused           int32
	ChunksEmbeddedThisRun  int32
	ChunksTotal            int32
	Files                  int32
	Chunks                 int32
	SkippedLine            string
	SyncNote               string
	HasStats               bool
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
