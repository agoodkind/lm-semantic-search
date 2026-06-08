// Package model defines the daemon's persisted and in-memory domain types.
package model

import (
	"time"

	"goodkind.io/lm-semantic-search/internal/discovery"
)

// CodebaseStatus captures the lifecycle state of one tracked codebase.
type CodebaseStatus string

const (
	// CodebaseStatusNotIndexed means the codebase has no active index.
	CodebaseStatusNotIndexed CodebaseStatus = "not_indexed"
	// CodebaseStatusIndexing means the codebase currently has an active job.
	CodebaseStatusIndexing CodebaseStatus = "indexing"
	// CodebaseStatusIndexed means the codebase has a completed index.
	CodebaseStatusIndexed CodebaseStatus = "indexed"
	// CodebaseStatusFailed means the last index attempt failed.
	CodebaseStatusFailed CodebaseStatus = "failed"
	// CodebaseStatusStale means the index metadata is known to be stale.
	CodebaseStatusStale CodebaseStatus = "stale"
	// CodebaseStatusMissing means the codebase's source directory is absent. It
	// is distinct from failed (a build error) and never auto-retries; the index
	// is kept in case the directory returns, and is removed only by an explicit
	// clear or the removed-worktree auto-clean.
	CodebaseStatusMissing CodebaseStatus = "missing"
)

// CodebaseKind distinguishes filesystem code indexes from virtual document
// collections. The empty value is accepted as code for older registry entries
// written before this field existed.
type CodebaseKind string

const (
	// CodebaseKindCode is a filesystem-backed source code collection.
	CodebaseKindCode CodebaseKind = "code"
	// CodebaseKindDocument is a virtual document collection with no source
	// directory on disk.
	CodebaseKindDocument CodebaseKind = "document"
)

// JobState captures the lifecycle state of one daemon job.
type JobState string

const (
	// JobStateQueued means the job was accepted but not started.
	JobStateQueued JobState = "queued"
	// JobStateRunning means the job is actively running.
	JobStateRunning JobState = "running"
	// JobStateCancelling means the job is winding down after cancellation.
	JobStateCancelling JobState = "cancelling"
	// JobStateCompleted means the job finished successfully.
	JobStateCompleted JobState = "completed"
	// JobStateFailed means the job ended in failure.
	JobStateFailed JobState = "failed"
	// JobStateCancelled means the job was cancelled.
	JobStateCancelled JobState = "cancelled"
)

// ClientInfo identifies the caller that initiated a daemon request.
type ClientInfo struct {
	Name string `json:"name"`
	PID  int32  `json:"pid,omitempty"`
}

// IndexConfig records the effective configuration of one indexing request.
type IndexConfig struct {
	SplitterType       string   `json:"splitter_type"`
	SplitterChunkSize  int32    `json:"splitter_chunk_size"`
	SplitterOverlap    int32    `json:"splitter_overlap"`
	Extensions         []string `json:"extensions,omitempty"`
	IgnorePatterns     []string `json:"ignore_patterns,omitempty"`
	IgnoreDigest       string   `json:"ignore_digest"`
	EmbeddingProvider  string   `json:"embedding_provider,omitempty"`
	EmbeddingModel     string   `json:"embedding_model,omitempty"`
	EmbeddingDimension int32    `json:"embedding_dimension,omitempty"`
	VectorBackend      string   `json:"vector_backend,omitempty"`
	Hybrid             bool     `json:"hybrid"`
}

// Progress records daemon-visible structured progress for a job.
type Progress struct {
	Phase          string  `json:"phase"`
	PhasePercent   float64 `json:"phase_percent"`
	OverallPercent float64 `json:"overall_percent"`
	FilesTotal     int32   `json:"files_total"`
	FilesProcessed int32   `json:"files_processed"`
	// FilesAdded, FilesModified, and FilesRemoved record a delta sync's change
	// breakdown so status output can show the magnitude of a reconcile (for
	// example after a large merge). They stay zero for a full reindex.
	FilesAdded    int32 `json:"files_added"`
	FilesModified int32 `json:"files_modified"`
	FilesRemoved  int32 `json:"files_removed"`
	// FilesInCodebase is the total file count in the current snapshot, so an
	// incremental run can report unchanged as this total less the changed set.
	FilesInCodebase int32 `json:"files_in_codebase"`
	// FilesEmbedded counts files this run actually re-embedded, so an
	// incremental run can report unchanged (FilesProcessed - FilesEmbedded)
	// separately from re-embedded. It stays zero until embedding begins.
	FilesEmbedded int32 `json:"files_embedded"`
	// FilesSkippedOversize and FilesSkippedUnreadable count changed files the
	// indexer declined to embed: past the size cap, or not valid UTF-8.
	FilesSkippedOversize   int32 `json:"files_skipped_oversize"`
	FilesSkippedUnreadable int32 `json:"files_skipped_unreadable"`
	// ChunksTotal is the live whole-collection chunk count, populated at render
	// time for an in-flight incremental run so status can show the running total
	// rather than only the per-run additions. Zero means not populated.
	ChunksTotal int32 `json:"chunks_total"`
	// ChunksReused counts chunks this run served from an already-embedded vector
	// (a merge-down child or sibling worktree) instead of calling the embedder,
	// so a surface can show total = reused + embedded and make the reuse-vs-redo
	// split visible. ChunksGenerated stays the embedded-this-run total.
	ChunksReused              int32     `json:"chunks_reused"`
	ChunksGenerated           int32     `json:"chunks_generated"`
	EmbeddingBatchesTotal     int32     `json:"embedding_batches_total"`
	EmbeddingBatchesCompleted int32     `json:"embedding_batches_completed"`
	CollectionRowsWritten     int32     `json:"collection_rows_written"`
	LastEventAt               time.Time `json:"last_event_at"`
	HeartbeatAt               time.Time `json:"heartbeat_at"`
}

// JobError records job-level failure details. TraceID and JobID tie the
// failure to the daemon's structured logs so an operator can grep for the
// full context behind a reported error.
type JobError struct {
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
	TraceID   string `json:"trace_id,omitempty"`
	JobID     string `json:"job_id,omitempty"`
}

// IndexRunSummary records the last successful indexing run for a codebase.
//
// SkippedFiles names files the indexer refused to embed because their bytes
// are not valid UTF-8 and would fail the Milvus gRPC marshal. The list is
// persisted in the registry so the operator can audit it without re-walking
// the daemon logs.
type IndexRunSummary struct {
	IndexedFiles int32     `json:"indexed_files"`
	TotalChunks  int32     `json:"total_chunks"`
	Status       string    `json:"status"`
	CompletedAt  time.Time `json:"completed_at"`
	SkippedFiles []string  `json:"skipped_files,omitempty"`
}

// IndexRunFailure records the last failed indexing run for a codebase.
// TraceID and JobID tie the failure to the daemon's structured logs so the
// reported error resolves to the full context by a log lookup.
type IndexRunFailure struct {
	Message                 string    `json:"message"`
	LastAttemptedPercentage int32     `json:"last_attempted_percentage"`
	FailedAt                time.Time `json:"failed_at"`
	TraceID                 string    `json:"trace_id,omitempty"`
	JobID                   string    `json:"job_id,omitempty"`
}

// Codebase records one canonical indexed codebase.
//
// ResolvedIgnoreRules is the runtime cache of the ignore rules that apply to
// this codebase. The discovery package computes it at registration and on
// periodic sync; the field is not persisted because it is derived from disk
// (built-in defaults, nested .gitignore files, repo ignore files, global
// ~/.context/.contextignore, and user overrides).
//
// InodeTrackingDisabled records that the root filesystem reported unstable
// inodes at registration time. When true, convergence falls back to
// path-only file tracking instead of (device, inode, contentHash).
type Codebase struct {
	ID                    string           `json:"id"`
	Kind                  CodebaseKind     `json:"kind,omitempty"`
	CanonicalPath         string           `json:"canonical_path"`
	Status                CodebaseStatus   `json:"status"`
	ActiveJobID           string           `json:"active_job_id,omitempty"`
	LastSuccessfulRun     *IndexRunSummary `json:"last_successful_run,omitempty"`
	LastFailedRun         *IndexRunFailure `json:"last_failed_run,omitempty"`
	EffectiveConfig       IndexConfig      `json:"effective_config"`
	CollectionName        string           `json:"collection_name,omitempty"`
	LegacyCollectionNames []string         `json:"legacy_collection_names,omitempty"`
	MerkleSnapshotPath    string           `json:"merkle_snapshot_path,omitempty"`
	// WorktreeCommonDir is the shared git common dir when this codebase's root
	// is a linked git worktree, else empty. It lets the daemon recognize a
	// removed worktree (git deleted its admin entry) after the directory is gone
	// and auto-clean the disposable index.
	WorktreeCommonDir     string                `json:"worktree_common_dir,omitempty"`
	InodeTrackingDisabled bool                  `json:"inode_tracking_disabled,omitempty"`
	ResolvedIgnoreRules   discovery.IgnoreRules `json:"-"`
	UpdatedAt             time.Time             `json:"updated_at"`
}

// Job records one daemon job and its latest known state.
type Job struct {
	ID            string     `json:"id"`
	CodebaseID    string     `json:"codebase_id"`
	RequestedPath string     `json:"requested_path"`
	CanonicalPath string     `json:"canonical_path"`
	Client        ClientInfo `json:"client"`
	Operation     string     `json:"operation"`
	State         JobState   `json:"state"`
	// Forced records that the caller passed force=true on the index request, so a
	// trigger-aware heading can tell a forced reindex apart from a first build or
	// a changed-files sync, which otherwise share the same operation.
	Forced      bool        `json:"forced"`
	Progress    Progress    `json:"progress"`
	Config      IndexConfig `json:"config"`
	StartedAt   time.Time   `json:"started_at"`
	UpdatedAt   time.Time   `json:"updated_at"`
	CompletedAt *time.Time  `json:"completed_at,omitempty"`
	Error       *JobError   `json:"error,omitempty"`
}

// RegistryFile is the durable JSON representation of tracked codebases.
type RegistryFile struct {
	Codebases []Codebase `json:"codebases"`
	UpdatedAt time.Time  `json:"updated_at"`
}

// JobEvent is one append-only journal entry for a job mutation.
type JobEvent struct {
	Event      string    `json:"event"`
	OccurredAt time.Time `json:"occurred_at"`
	Job        Job       `json:"job"`
}

// StoredChunk is one persisted search chunk for a codebase.
type StoredChunk struct {
	Content        string `json:"content"`
	RelativePath   string `json:"relative_path"`
	StartLine      int32  `json:"start_line"`
	EndLine        int32  `json:"end_line"`
	Language       string `json:"language"`
	FileExtension  string `json:"file_extension"`
	ConversationID string `json:"conversation_id"`
	// ParentConversationID names the conversation this chunk's conversation
	// forked from, so a fork can be grouped with its parent. Empty for code
	// chunks and for conversations with no parent.
	ParentConversationID string `json:"parent_conversation_id"`
	MessageIndex         int32  `json:"message_index"`
	Role                 string `json:"role"`
	TimestampUnix        int64  `json:"timestamp_unix"`
}

// ConversationDocument is one caller-provided conversation message chunk.
type ConversationDocument struct {
	ConversationID string `json:"conversation_id"`
	// ParentConversationID names the conversation this one forked from, carried
	// into chunk metadata so forks group with their parent. Empty when absent.
	ParentConversationID string `json:"parent_conversation_id"`
	MessageIndex         int32  `json:"message_index"`
	Role                 string `json:"role"`
	TimestampUnix        int64  `json:"timestamp_unix"`
	Text                 string `json:"text"`
}

// PathClassificationKind reports the daemon's verdict about one queried path.
type PathClassificationKind string

const (
	// PathClassificationUnspecified is the zero value; callers should treat
	// it as out-of-scope or unknown.
	PathClassificationUnspecified PathClassificationKind = ""
	// PathClassificationInScopeIndexed means the path is covered by a
	// codebase, survives ignore rules, and has a chunk row in the codebase
	// collection.
	PathClassificationInScopeIndexed PathClassificationKind = "in_scope_indexed"
	// PathClassificationInScopeExcluded means the path is covered but
	// excluded by the codebase's resolved ignore rules.
	PathClassificationInScopeExcluded PathClassificationKind = "in_scope_excluded"
	// PathClassificationInScopeUnindexed means the path is covered and not
	// excluded but has no chunk row (not yet indexed or removed).
	PathClassificationInScopeUnindexed PathClassificationKind = "in_scope_unindexed"
	// PathClassificationOutOfScope means no tracked codebase covers the
	// path's canonical form.
	PathClassificationOutOfScope PathClassificationKind = "out_of_scope"
)

// PathClassification carries the daemon's verdict about one queried path.
// CoveringCodebaseID names the longest-prefix covering codebase, when any.
// ExcludedByPattern and ExcludedByGitignore name the rule that excluded
// the path when Kind is PathClassificationInScopeExcluded.
type PathClassification struct {
	Kind                PathClassificationKind
	ExcludedByPattern   string
	ExcludedByGitignore string
	CoveringCodebaseID  string
}
