# Presentation Choke Point Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every information surface through one resolver layer that produces typed view models, then put the render layer behind a compile-time wall so it cannot read raw model records.

**Architecture:** Boundary resolvers in `internal/daemon` are the only presentation code that reads `model.*`. They produce view models in a new `internal/view` package. All formatters move to a new `internal/render` package that imports `view` but not `model`, so laundering becomes a compile error. Riders: the boot-resume kind-skip bug, the ghost registry record, URI rejection in path canonicalization, and a uniform envelope.

**Tech Stack:** Go, `internal/status` (vocabulary + resolution rules), new `internal/view` and `internal/render`, `go/ast` guard tests.

**Spec:** `docs/superpowers/specs/2026-06-10-presentation-choke-point-design.md`

**Conventions for every task:** run `go test ./internal/...` after each code step; `make lint` and `make build` run in the final task and after any task that the executor suspects changed lint surface. Commit after each task with the message given. The plan never uses em or en dashes in code or prose because the repo prose gate blocks them.

---

### Task 1: Riders: resume kind-skip, URI rejection, ghost record cleanup

These fix the active recurring bug (a boot-resume pass re-launches the conversation codebase as a filesystem index of `/chat:/clyde-conversations` on every daemon restart) and remove the corrupt record it created.

**Files:**
- Modify: `internal/daemon/manager_resume.go`
- Modify: `internal/daemon/manager_paths.go` (`canonicalizePath`)
- Modify: `internal/daemon/manager.go` (registry load repair; find the function that loads the registry into `manager.codebases`, it is in the `NewManager` path)
- Test: `internal/daemon/manager_resume_test.go` (create if absent), `internal/daemon/manager_paths_test.go` (create if absent)

- [ ] **Step 1: Write the failing tests**

In `internal/daemon/manager_resume_test.go` (create with package `daemon` and imports `context`, `testing`, `goodkind.io/lm-semantic-search/internal/model`):

```go
// A document (conversation) codebase left mid-index must never be re-launched
// by the boot resume pass: its path is a chat:// URI, not a directory, and the
// conversation trigger path owns its recovery.
func TestResumeOrphanedJobsSkipsDocumentCodebases(t *testing.T) {
	manager, _, _ := newTestManager(t)
	manager.config.ResumeIndexingOnBoot = true

	codebase := newCodebaseRecord("chat:///clyde-conversations")
	codebase.Kind = model.CodebaseKindDocument
	codebase.Status = model.CodebaseStatusIndexing
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	manager.ResumeOrphanedJobs(context.Background())

	manager.mu.Lock()
	jobCount := len(manager.jobs)
	manager.mu.Unlock()
	if jobCount != 0 {
		t.Fatalf("resume launched %d job(s) for a document codebase, want 0", jobCount)
	}
}
```

In `internal/daemon/manager_paths_test.go` (create with package `daemon`, imports `strings`, `testing`):

```go
// A URI-shaped argument must be rejected, not resolved as a filesystem path.
// filepath.Abs on "chat:///x" with cwd "/" produced the ghost path
// "/chat:/x" that broke the boot resume pass.
func TestCanonicalizePathRejectsURISchemes(t *testing.T) {
	t.Parallel()
	for _, arg := range []string{"chat:///clyde-conversations", "https://example.com/repo", "file:///tmp/x"} {
		_, err := canonicalizePath(arg)
		if err == nil {
			t.Fatalf("canonicalizePath(%q) succeeded, want a rejection", arg)
		}
		if !strings.Contains(err.Error(), "URI") {
			t.Fatalf("canonicalizePath(%q) error %q does not name the URI cause", arg, err)
		}
	}
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `go test ./internal/daemon/ -run 'TestResumeOrphanedJobsSkipsDocumentCodebases|TestCanonicalizePathRejectsURISchemes' -v`
Expected: both FAIL (resume launches a job; canonicalizePath resolves the URI).

- [ ] **Step 3: Implement the three fixes**

3a. In `internal/daemon/manager_resume.go`, inside the `for _, codebase := range manager.codebases` loop in `ResumeOrphanedJobs`, add the kind skip as the first check:

```go
		if codebase.Kind == model.CodebaseKindDocument {
			// A conversation codebase recovers through its own ingest trigger;
			// its chat:// path is not a directory the index runner can walk.
			continue
		}
		if codebase.Status != model.CodebaseStatusIndexing {
			continue
		}
```

3b. In `internal/daemon/manager_paths.go`, at the top of `canonicalizePath` after the empty-path check, add:

```go
	if strings.Contains(requestedPath, "://") {
		return "", fmt.Errorf("path %q looks like a URI; pass a filesystem directory instead", requestedPath)
	}
```

(`strings` and `fmt` are already imported in that file.)

3c. In `internal/daemon/manager.go`, in the registry-load path of `NewManager` (immediately after the loaded codebases map is populated), add a repair pass plus its helper at file scope:

```go
	dropGhostURICodebases(manager.codebases)
```

```go
// dropGhostURICodebases removes code-kind records whose canonical path is a
// filesystem-mangled URI (for example "/chat:/clyde-conversations"), which the
// pre-fix boot resume pass created by running filepath.Abs on a chat:// path.
// A legitimate conversation codebase keeps its scheme intact and its kind set
// to document, so it never matches.
func dropGhostURICodebases(codebases map[string]model.Codebase) {
	for id, codebase := range codebases {
		if codebase.Kind == model.CodebaseKindDocument {
			continue
		}
		segments := strings.SplitN(strings.TrimPrefix(codebase.CanonicalPath, "/"), "/", 2)
		if len(segments) > 0 && strings.HasSuffix(segments[0], ":") {
			slog.Warn("dropping ghost URI codebase record", "codebase_id", id, "path", codebase.CanonicalPath)
			delete(codebases, id)
		}
	}
}
```

Add a test in `internal/daemon/manager_paths_test.go`:

```go
func TestDropGhostURICodebases(t *testing.T) {
	t.Parallel()
	ghost := newCodebaseRecord("/chat:/clyde-conversations")
	real := newCodebaseRecord("/Users/x/repo")
	conversation := newCodebaseRecord("chat:///clyde-conversations")
	conversation.Kind = model.CodebaseKindDocument
	codebases := map[string]model.Codebase{ghost.ID: ghost, real.ID: real, conversation.ID: conversation}

	dropGhostURICodebases(codebases)

	if _, ok := codebases[ghost.ID]; ok {
		t.Fatal("ghost URI record survived the repair pass")
	}
	if _, ok := codebases[real.ID]; !ok {
		t.Fatal("real filesystem record was dropped")
	}
	if _, ok := codebases[conversation.ID]; !ok {
		t.Fatal("document codebase was dropped")
	}
}
```

(Add `model` to the imports of that test file.)

- [ ] **Step 4: Run the tests**

Run: `go test ./internal/daemon/ -run 'TestResumeOrphanedJobsSkipsDocumentCodebases|TestCanonicalizePathRejectsURISchemes|TestDropGhostURICodebases' -v`
Expected: all PASS. Also run `go test ./internal/daemon/` to confirm no regression (conversation registration must still work because it never calls `canonicalizePath` on the chat URI; if any existing test fails on the URI rejection, that call site is a bug to route through the conversation path, not a reason to weaken the check).

- [ ] **Step 5: Commit**

```bash
git add internal/daemon/manager_resume.go internal/daemon/manager_paths.go internal/daemon/manager.go internal/daemon/manager_resume_test.go internal/daemon/manager_paths_test.go
git commit -m "Skip document codebases in boot resume, reject URI paths, drop ghost URI records"
```

---

### Task 2: Create internal/view with all view models

Pure data shapes only. No function in this package reads `model.*` except type references in field types are forbidden too: fields are plain strings, ints, bools, slices of view types.

**Files:**
- Create: `internal/view/view.go`
- Create: `internal/view/view_test.go`

- [ ] **Step 1: Write the package**

`internal/view/view.go`:

```go
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
	HasFailure bool
	Message    string
	FailedAtLabel string
	JobID      string
	TraceID    string
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
	Score        float32
	Content      string
}

// SearchView is the code search response view.
type SearchView struct {
	RequestedPath   string
	Query           string
	CodebaseName    string
	Results         []SearchResultView
	StateNote       string
	InFlight        bool
	InFlightStatus  StatusView
	InFlightPercent int32
	Degraded        bool
	ResolutionLines []string
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
	Score          float32
	Content        string
}

// GetIndexView is the resolved codebase status response.
type GetIndexView struct {
	Tracked         bool
	RequestedPath   string
	CanonicalPath   string
	Display         Display
	TemplateName    string
	Status          StatusView
	Failure         FailureSurface
	WaitLabel       string
	ClassificationLine string
	ResolutionLines []string
	DescendantsHint string
	SyncNote        string
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
```

`internal/view/view_test.go`:

```go
package view

import "testing"

// The view package must stay pure data: this test locks the absence of an
// internal/model dependency at the package level. The render wall depends on
// view being importable without model.
func TestViewHasNoModelDependency(t *testing.T) {
	// Compile-time property: if any view type referenced internal/model this
	// package would import it and the import-list assertion in the render
	// package tests would fail. This test exists to document the invariant.
	_ = ProgressSurface{}
	_ = JobEntryView{}
	_ = GetIndexView{}
}
```

- [ ] **Step 2: Compile and test**

Run: `go test ./internal/view/`
Expected: PASS (compiles, no model import).

- [ ] **Step 3: Commit**

```bash
git add internal/view/
git commit -m "Add internal/view package with all presentation view models"
```

---

### Task 3: RunMode, scope unit, and live totals plumbing

The data the new `ProgressSurface` needs but the model does not yet carry.

**Files:**
- Modify: `internal/model/types.go` (add `RunMode`, `ScopeUnit` to `Progress`; add `LiveFileTotal`, `LiveChunkTotal` to `Codebase`)
- Modify: `internal/daemon/manager_delta.go` (set RunMode in the bootstrap and delta paths)
- Modify: `internal/daemon/manager_jobs_state.go` (carry live totals onto the codebase during progress and completion)
- Modify: `internal/daemon/manager_conversations.go` (set ScopeUnit "conversation" where Unit "document" is set)
- Test: `internal/daemon/manager_runmode_test.go` (create)

- [ ] **Step 1: Add the model fields**

In `internal/model/types.go`, add to `Progress` (next to `Unit`):

```go
	// RunMode names the kind of pass: "first_build", "changed",
	// "forced_reindex", or "resuming". Set when the run plan is decided so
	// surfaces can label the denominator and name a resume.
	RunMode string `json:"runMode,omitempty"`
	// ScopeUnit is the unit of the added/modified/removed classification when
	// it differs from Unit (a conversation manifest classifies conversations
	// while delivery counts documents). Empty means same as Unit.
	ScopeUnit string `json:"scopeUnit,omitempty"`
```

Add to `Codebase`:

```go
	// LiveFileTotal and LiveChunkTotal track the latest known corpus size,
	// updated during runs rather than only at completion, so a failed run does
	// not leave the totals frozen at the last success.
	LiveFileTotal  int32 `json:"liveFileTotal,omitempty"`
	LiveChunkTotal int32 `json:"liveChunkTotal,omitempty"`
```

- [ ] **Step 2: Set RunMode at plan time**

In `internal/daemon/manager_delta.go`:

2a. In `runBootstrap`, immediately after `plan := manager.planBootstrap(ctx, job, codebase.ID)` and its `plan.handled` return, add:

```go
	runMode := model.RunModeFirstBuild
	if len(plan.seedSnapshot.Files) > 0 {
		runMode = model.RunModeResuming
	}
	if job.Forced && codebase.LastSuccessfulRun != nil {
		runMode = model.RunModeForcedReindex
	}
	manager.setJobRunMode(job.ID, runMode)
```

2b. In `runDeltaSync`, at the point the delta is accepted (right after `planSyncDiff` succeeds and before the per-file loop), add:

```go
	manager.setJobRunMode(job.ID, model.RunModeChanged)
```

2c. Add the constants to `internal/model/types.go`:

```go
// RunMode values for Progress.RunMode.
const (
	RunModeFirstBuild    = "first_build"
	RunModeChanged       = "changed"
	RunModeForcedReindex = "forced_reindex"
	RunModeResuming      = "resuming"
)
```

2d. Add the setter in `internal/daemon/manager_jobs_state.go` next to `setJobDeltaCounts`:

```go
// setJobRunMode records the kind of pass a run is making, decided once when
// the plan is chosen, so surfaces can label denominators and name a resume.
func (manager *Manager) setJobRunMode(jobID string, runMode string) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	job, found := manager.jobs[jobID]
	if !found {
		return
	}
	job.Progress.RunMode = runMode
	manager.jobs[jobID] = job
}
```

- [ ] **Step 3: Carry live totals**

In `internal/daemon/manager_jobs_state.go`:

3a. In `updateJobProgress`, after the job fields are copied, add (inside the same lock):

```go
	if codebase, ok := manager.codebases[job.CodebaseID]; ok {
		changed := false
		if progress.FilesInCodebase > 0 && codebase.LiveFileTotal != progress.FilesInCodebase {
			codebase.LiveFileTotal = progress.FilesInCodebase
			changed = true
		}
		liveChunks := max(progress.ChunksTotal, progress.ChunksReused+progress.ChunksGenerated)
		if liveChunks > codebase.LiveChunkTotal {
			codebase.LiveChunkTotal = liveChunks
			changed = true
		}
		if changed {
			manager.codebases[job.CodebaseID] = codebase
		}
	}
```

(No registry write here; the in-memory record serves reads and `updateJobCompleted` already persists.)

3b. In `updateJobCompleted`, where `codebase.LastSuccessfulRun` is set, also set:

```go
	codebase.LiveFileTotal = result.IndexedFiles
	codebase.LiveChunkTotal = result.TotalChunks
```

- [ ] **Step 4: Conversation scope unit**

In `internal/daemon/manager_conversations.go`, find each `Unit: "document"` (or `progress.Unit = "document"`) assignment and set alongside it:

```go
	ScopeUnit: "conversation",
```

(or `progress.ScopeUnit = "conversation"` for field-style assignment; the compiler finds every literal that now needs the field if you add it to a fully-keyed struct literal flagged by exhaustruct.)

- [ ] **Step 5: Test**

`internal/daemon/manager_runmode_test.go`:

```go
package daemon

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

func TestSetJobRunMode(t *testing.T) {
	manager, _, _ := newTestManager(t)
	job := model.Job{ID: "job-rm", State: model.JobStateRunning}
	manager.mu.Lock()
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	manager.setJobRunMode(job.ID, model.RunModeResuming)

	got, _ := manager.GetJob(job.ID)
	if got.Progress.RunMode != model.RunModeResuming {
		t.Fatalf("RunMode = %q, want %q", got.Progress.RunMode, model.RunModeResuming)
	}
}
```

Run: `go test ./internal/daemon/ -run TestSetJobRunMode -v` then `go test ./internal/daemon/ ./internal/model/...`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/model/types.go internal/daemon/manager_delta.go internal/daemon/manager_jobs_state.go internal/daemon/manager_conversations.go internal/daemon/manager_runmode_test.go
git commit -m "Carry run mode, scope unit, and live corpus totals through job progress"
```

---

### Task 4: Boundary resolvers produce the view models

All resolution concentrates in `internal/daemon/status_present.go` (and a new `internal/daemon/present_progress.go` to keep files focused). After this task the resolvers exist and are tested; render functions still have their old signatures.

**Files:**
- Create: `internal/daemon/present_progress.go`
- Create: `internal/daemon/present_progress_test.go`
- Modify: `internal/daemon/status_present.go` (return view types from existing resolvers)

- [ ] **Step 1: Convert existing resolvers to view types**

In `internal/daemon/status_present.go`:

1a. `resolveJobSurface` returns `view.JobSurface` by converting the `status.JobSurface`:

```go
func resolveJobSurface(job model.Job, pipelineDegraded bool, supersededByJobID string) view.JobSurface {
	dependency := status.Healthy
	if pipelineDegraded {
		dependency = status.EmbedderBusy
	}
	retryable := false
	errorMessage := ""
	if job.Error != nil {
		retryable = job.Error.Retryable
		errorMessage = strings.TrimSpace(job.Error.Message)
	}
	resolved := status.ResolveJob(status.JobInputs{
		State:             job.State,
		Retryable:         retryable,
		ErrorMessage:      errorMessage,
		Dependency:        dependency,
		SupersededByJobID: supersededByJobID,
	})
	return view.JobSurface{
		StateLabel:        resolved.StateLabel,
		ErrorLine:         resolved.ErrorLine,
		Superseded:        resolved.Superseded,
		SupersededByJobID: resolved.SupersededByJobID,
	}
}
```

1b. `resolveCodebaseFailure` returns `view.FailureSurface`, formatting the time at the boundary:

```go
func resolveCodebaseFailure(codebase model.Codebase) view.FailureSurface {
	if codebase.LastFailedRun == nil {
		return view.FailureSurface{HasFailure: false, Message: "", FailedAtLabel: "", JobID: "", TraceID: ""}
	}
	return view.FailureSurface{
		HasFailure:    true,
		Message:       codebase.LastFailedRun.Message,
		FailedAtLabel: formatLocalTime(codebase.LastFailedRun.FailedAt),
		JobID:         codebase.LastFailedRun.JobID,
		TraceID:       codebase.LastFailedRun.TraceID,
	}
}
```

(`formatLocalTime` stays reachable from the daemon package until the render move; after the move it lives in render and the boundary uses its own copy named `formatLocalTime` kept in `internal/daemon/present_progress.go`. Define it there in this task so the boundary never depends on render helpers:)

```go
// formatBoundaryTime renders a timestamp for view labels in daemon local time.
func formatBoundaryTime(value time.Time) string {
	if value.IsZero() {
		return ""
	}
	local := value.Local()
	return local.Format("1/2/2006, 3:04:05 PM MST")
}
```

Use `formatBoundaryTime` in `resolveCodebaseFailure` and everywhere this task formats times.

- [ ] **Step 2: Write the progress resolver and its helpers**

`internal/daemon/present_progress.go`:

```go
package daemon

import (
	"fmt"
	"strings"
	"time"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/view"
)

// formatCount renders an integer with thousands separators so large corpus
// numbers stay readable ("33,240").
func formatCount(value int32) string {
	digits := fmt.Sprintf("%d", value)
	if len(digits) <= 3 {
		return digits
	}
	var out strings.Builder
	lead := len(digits) % 3
	if lead > 0 {
		out.WriteString(digits[:lead])
		if len(digits) > lead {
			out.WriteString(",")
		}
	}
	for i := lead; i < len(digits); i += 3 {
		out.WriteString(digits[i : i+3])
		if i+3 < len(digits) {
			out.WriteString(",")
		}
	}
	return out.String()
}

// progressHeading names the pass for an active job, empty otherwise.
func progressHeading(job model.Job) string {
	if job.State != model.JobStateQueued && job.State != model.JobStateRunning && job.State != model.JobStateCancelling {
		return ""
	}
	switch job.Progress.RunMode {
	case model.RunModeResuming:
		return "Resuming after restart: checking changed work, embedding only what's new"
	case model.RunModeFirstBuild:
		return "Building initial index"
	case model.RunModeForcedReindex:
		return "Forced reindex"
	case model.RunModeChanged:
		return "Indexing new changes"
	default:
		return ""
	}
}

// scopeLabelFor types the denominator from the run mode and unit.
func scopeLabelFor(runMode string, unit string, total int32) string {
	plural := unit
	if total != 1 {
		plural = unit + "s"
	}
	switch runMode {
	case model.RunModeFirstBuild:
		return plural + " (full build)"
	case model.RunModeForcedReindex:
		return plural + " (forced reindex)"
	case model.RunModeResuming, model.RunModeChanged:
		return "changed " + plural
	default:
		return plural
	}
}

// checkVerbFor is "checked" for fast-forward passes and "embedded" otherwise.
func checkVerbFor(runMode string) string {
	switch runMode {
	case model.RunModeResuming, model.RunModeChanged:
		return "checked"
	default:
		return "embedded"
	}
}

// resolveProgressSurface reduces a job's progress into the typed view. It is
// the only reader of Progress fields for presentation.
func resolveProgressSurface(job model.Job) view.ProgressSurface {
	progress := job.Progress
	unit := progress.Unit
	if unit == "" {
		unit = "file"
	}
	scopeUnit := progress.ScopeUnit
	if scopeUnit == "" {
		scopeUnit = unit
	}

	active := job.State == model.JobStateQueued || job.State == model.JobStateRunning || job.State == model.JobStateCancelling
	hasScope := progress.FilesTotal > 0

	percentLabel := fmt.Sprintf("%.1f%%", progress.OverallPercent)
	if active && !jobScopeKnown(progress) {
		if jobOperation(job.Operation) == jobOperationSync {
			percentLabel = "Changes detected, preparing to index"
		} else {
			percentLabel = "Preparing to index"
		}
	}

	removedAndSkipped := progress.FilesRemoved + progress.FilesSkippedOversize + progress.FilesSkippedUnreadable
	alreadyIndexed := progress.FilesProcessed - progress.FilesEmbedded - removedAndSkipped
	if alreadyIndexed < 0 {
		alreadyIndexed = 0
	}

	scopeLine := ""
	if progress.FilesAdded > 0 || progress.FilesModified > 0 || progress.FilesRemoved > 0 {
		parts := []string{}
		if progress.FilesAdded > 0 {
			parts = append(parts, fmt.Sprintf("%s %s added", formatCount(progress.FilesAdded), scopeUnit+pluralSuffix(progress.FilesAdded)))
		}
		if progress.FilesModified > 0 {
			parts = append(parts, fmt.Sprintf("%s modified", formatCount(progress.FilesModified)))
		}
		if progress.FilesRemoved > 0 {
			parts = append(parts, fmt.Sprintf("%s removed", formatCount(progress.FilesRemoved)))
		}
		scopeLine = "Changed since last sync: " + strings.Join(parts, " · ")
	}

	return view.ProgressSurface{
		Heading:            progressHeading(job),
		HasScope:           hasScope,
		Checked:            progress.FilesProcessed,
		ScopeTotal:         progress.FilesTotal,
		ScopeLabel:         scopeLabelFor(progress.RunMode, unit, progress.FilesTotal),
		CheckVerb:          checkVerbFor(progress.RunMode),
		Embedded:           progress.FilesEmbedded,
		AlreadyIndexed:     alreadyIndexed,
		ChunksThisRun:      progress.ChunksGenerated,
		ChunksReused:       progress.ChunksReused,
		ChunksInCollection: max(progress.ChunksTotal, progress.ChunksReused+progress.ChunksGenerated),
		ScopeLine:          scopeLine,
		PercentLabel:       percentLabel,
	}
}

// pluralSuffix returns "s" for counts other than one.
func pluralSuffix(count int32) string {
	if count == 1 {
		return ""
	}
	return "s"
}

// resolveTimingView formats a job's timing block.
func resolveTimingView(job model.Job) view.TimingView {
	timing := view.TimingView{
		StartedLabel:   formatBoundaryTime(job.StartedAt),
		UpdatedLabel:   formatBoundaryTime(job.UpdatedAt),
		CompletedLabel: "",
		DurationLabel:  "",
		DurationWord:   "Elapsed",
	}
	if job.CompletedAt != nil {
		timing.CompletedLabel = formatBoundaryTime(*job.CompletedAt)
		timing.DurationWord = "Duration"
	}
	end := job.UpdatedAt
	if job.CompletedAt != nil && !job.CompletedAt.IsZero() {
		end = *job.CompletedAt
	}
	if !job.StartedAt.IsZero() && !end.IsZero() && !end.Before(job.StartedAt) {
		duration := end.Sub(job.StartedAt).Round(time.Second)
		if duration <= 0 {
			timing.DurationLabel = "0s"
		} else {
			timing.DurationLabel = duration.String()
		}
	}
	return timing
}

// resolveJobEntry assembles the full job view for list and detail surfaces.
func resolveJobEntry(job model.Job, pipelineDegraded bool, supersededByJobID string) view.JobEntryView {
	return view.JobEntryView{
		ID:            job.ID,
		CanonicalPath: job.CanonicalPath,
		Operation:     job.Operation,
		PhaseLabel:    displayJobPhase(job.Progress.Phase),
		Surface:       resolveJobSurface(job, pipelineDegraded, supersededByJobID),
		Progress:      resolveProgressSurface(job),
		Timing:        resolveTimingView(job),
	}
}

// resolveListSummary tallies the job list.
func resolveListSummary(jobs []model.Job, pipelineDegraded bool) view.ListSummary {
	successors := buildJobSuccessors(jobs)
	summary := view.ListSummary{Total: len(jobs), Queued: 0, Running: 0, Canceling: 0, Completed: 0, Failed: 0, Superseded: 0, Canceled: 0}
	for _, job := range jobs {
		switch job.State {
		case model.JobStateQueued:
			summary.Queued++
		case model.JobStateRunning:
			summary.Running++
		case model.JobStateCancelling:
			summary.Canceling++
		case model.JobStateCompleted:
			summary.Completed++
		case model.JobStateFailed:
			if resolveJobSurface(job, pipelineDegraded, successors[job.ID]).Superseded {
				summary.Superseded++
			} else {
				summary.Failed++
			}
		case model.JobStateCancelled:
			summary.Canceled++
		}
	}
	return summary
}
```

(`displayJobPhase`, `jobScopeKnown`, `jobOperation`, `buildJobSuccessors` already exist in the daemon package. `max` is the Go builtin.)

- [ ] **Step 3: Test the resolvers**

`internal/daemon/present_progress_test.go`:

```go
package daemon

import (
	"strings"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/model"
)

func TestFormatCount(t *testing.T) {
	t.Parallel()
	cases := map[int32]string{0: "0", 29: "29", 1011: "1,011", 33240: "33,240", 124754: "124,754"}
	for in, want := range cases {
		if got := formatCount(in); got != want {
			t.Errorf("formatCount(%d) = %q, want %q", in, got, want)
		}
	}
}

func TestResolveProgressSurfaceResumingIngest(t *testing.T) {
	t.Parallel()
	job := model.Job{
		ID: "job-p", State: model.JobStateRunning, Operation: "conversation_ingest",
		Progress: model.Progress{
			RunMode: model.RunModeResuming, Unit: "document", ScopeUnit: "conversation",
			OverallPercent: 23.5, FilesTotal: 1011, FilesProcessed: 238, FilesEmbedded: 12,
			FilesAdded: 1004, FilesModified: 7,
			ChunksGenerated: 29, ChunksTotal: 33240,
		},
	}
	got := resolveProgressSurface(job)
	if got.Heading == "" || !strings.Contains(got.Heading, "Resuming after restart") {
		t.Fatalf("heading = %q, want the resume heading", got.Heading)
	}
	if got.ScopeLabel != "changed documents" {
		t.Fatalf("scope label = %q, want %q", got.ScopeLabel, "changed documents")
	}
	if got.CheckVerb != "checked" {
		t.Fatalf("check verb = %q, want checked", got.CheckVerb)
	}
	if got.AlreadyIndexed != 226 {
		t.Fatalf("already indexed = %d, want 226 (238 checked minus 12 embedded)", got.AlreadyIndexed)
	}
	if got.ChunksInCollection != 33240 {
		t.Fatalf("collection total = %d, want 33240", got.ChunksInCollection)
	}
	if !strings.Contains(got.ScopeLine, "1,004 conversations added · 7 modified") {
		t.Fatalf("scope line = %q, want the typed classification", got.ScopeLine)
	}
}

func TestResolveProgressSurfaceFirstBuild(t *testing.T) {
	t.Parallel()
	job := model.Job{
		ID: "job-fb", State: model.JobStateRunning, Operation: "index",
		Progress: model.Progress{RunMode: model.RunModeFirstBuild, FilesTotal: 100, FilesProcessed: 10, FilesEmbedded: 10, OverallPercent: 10},
	}
	got := resolveProgressSurface(job)
	if got.ScopeLabel != "files (full build)" {
		t.Fatalf("scope label = %q, want %q", got.ScopeLabel, "files (full build)")
	}
	if got.CheckVerb != "embedded" {
		t.Fatalf("check verb = %q, want embedded for a full build", got.CheckVerb)
	}
}

func TestResolveListSummarySplitsSuperseded(t *testing.T) {
	t.Parallel()
	older := time.Now().Add(-2 * time.Minute)
	newer := time.Now().Add(-1 * time.Minute)
	jobs := []model.Job{
		{ID: "a1", CodebaseID: "A", State: model.JobStateFailed, StartedAt: older, CompletedAt: &older, Error: &model.JobError{Message: "x", Retryable: true}},
		{ID: "a2", CodebaseID: "A", State: model.JobStateFailed, StartedAt: older, CompletedAt: &newer, Error: &model.JobError{Message: "y", Retryable: false}},
		{ID: "b1", CodebaseID: "B", State: model.JobStateCompleted, StartedAt: older, CompletedAt: &newer},
	}
	got := resolveListSummary(jobs, false)
	if got.Failed != 1 || got.Superseded != 1 || got.Completed != 1 {
		t.Fatalf("summary = %+v, want 1 failed, 1 superseded, 1 completed", got)
	}
}
```

Run: `go test ./internal/daemon/ -run 'TestFormatCount|TestResolveProgressSurface|TestResolveListSummary' -v`
Expected: PASS. (The repo compiles throughout because the old render paths still use their old helpers; the view-returning conversions in Step 1 require updating the call sites in render.go in the same change: `surface.StateLabel` etc. keep the same field names, so `renderGetJob`, `renderJobListEntry`, `renderListJobs`, `renderHistoricalFailure`, `renderStaleStatus`, `renderFailureDiagnostics`, and `applyJobDisplayTokens` continue to compile against `view.JobSurface`/`view.FailureSurface` with one edit: `renderStaleStatus` reads `failure.FailedAtLabel` instead of calling `formatLocalTime(failure.FailedAt)`.)

- [ ] **Step 4: Run the full daemon tests and commit**

Run: `go test ./internal/daemon/ ./internal/status/ ./internal/view/`
Expected: PASS.

```bash
git add internal/daemon/present_progress.go internal/daemon/present_progress_test.go internal/daemon/status_present.go internal/daemon/render.go
git commit -m "Resolve job, progress, timing, and list views at the daemon boundary"
```

---

### Task 5: Job surfaces format the resolved views

`renderGetJob`, `renderListJobs`, and `renderJobListEntry` take view models. The display gains the heading, the typed denominator, the embedded/already-indexed split, the collection total, and the typed scope line.

**Files:**
- Modify: `internal/daemon/render.go`
- Modify: `internal/daemon/grpc_server.go` (call sites)
- Modify: `internal/daemon/render_test.go`, `internal/daemon/banner_test.go` (expectations)
- Modify: `internal/daemon/render_guard_test.go` (forbid Progress reads in render)

- [ ] **Step 1: Rewrite the three renderers**

In `internal/daemon/render.go`, replace `renderGetJob`, `renderListJobs`, `renderJobListEntry`, `renderJobTimingLines`, `formatJobDuration`, `jobProgressDisplay`, and `renderReconcileMagnitude` with view-driven versions:

```go
func renderGetJob(entry view.JobEntryView, found bool) string {
	if !found {
		return "Job not found."
	}
	lines := []string{
		"🧾 Job " + entry.ID,
		"📁 Codebase: " + entry.CanonicalPath,
		"⚙️ Operation: " + entry.Operation,
		"🚦 State: " + entry.Surface.StateLabel,
		"🔧 Phase: " + entry.PhaseLabel,
		"📊 Progress: " + entry.Progress.PercentLabel,
	}
	lines = append(lines, renderTimingLines(entry.Timing)...)
	lines = append(lines, renderProgressLines(entry.Progress)...)
	if entry.Surface.ErrorLine != "" {
		lines = append(lines, "🧯 Error: "+entry.Surface.ErrorLine)
	}
	return strings.Join(lines, "\n")
}

// renderProgressLines renders the resolved progress: heading, the typed
// denominator with the work split, the chunk line with the collection total,
// and the typed classification line. Each line renders only when its data is
// present so terminal acks stay compact.
func renderProgressLines(progress view.ProgressSurface) []string {
	lines := make([]string, 0, 4)
	if progress.Heading != "" {
		lines = append(lines, "  "+progress.Heading)
	}
	if progress.HasScope {
		main := fmt.Sprintf("  📄 %s of %s %s %s",
			formatCountString(progress.Checked), formatCountString(progress.ScopeTotal), progress.ScopeLabel, progress.CheckVerb)
		if progress.CheckVerb == "checked" {
			main += fmt.Sprintf(" · %s embedded · %s already indexed",
				formatCountString(progress.Embedded), formatCountString(progress.AlreadyIndexed))
		}
		lines = append(lines, main)
	}
	if progress.ChunksThisRun > 0 || progress.ChunksInCollection > 0 {
		chunkLine := fmt.Sprintf("  🧩 %s chunks added this run", formatCountString(progress.ChunksThisRun))
		if progress.ChunksReused > 0 {
			chunkLine += fmt.Sprintf(" · %s reused", formatCountString(progress.ChunksReused))
		}
		if progress.ChunksInCollection > 0 {
			chunkLine += fmt.Sprintf(" · %s in collection", formatCountString(progress.ChunksInCollection))
		}
		lines = append(lines, chunkLine)
	}
	if progress.ScopeLine != "" {
		lines = append(lines, "  "+progress.ScopeLine)
	}
	return lines
}

func renderTimingLines(timing view.TimingView) []string {
	lines := []string{
		"  Started: " + timing.StartedLabel,
		"  Updated: " + timing.UpdatedLabel,
	}
	if timing.CompletedLabel != "" {
		lines = append(lines, "  Completed: "+timing.CompletedLabel)
	}
	if timing.DurationLabel != "" {
		lines = append(lines, "  "+timing.DurationWord+": "+timing.DurationLabel)
	}
	return lines
}

func renderListJobs(summary view.ListSummary, active []view.JobEntryView, terminal []view.JobEntryView) string {
	if summary.Total == 0 {
		return "No tracked jobs."
	}
	lines := make([]string, 0, 32)
	lines = append(lines, fmt.Sprintf("Tracked jobs: %d total", summary.Total))
	lines = append(lines, fmt.Sprintf("Active: %d queued, %d running, %d canceling", summary.Queued, summary.Running, summary.Canceling))
	lines = append(lines, fmt.Sprintf("Terminal: %d completed, %d failed, %d superseded, %d canceled",
		summary.Completed, summary.Failed, summary.Superseded, summary.Canceled))
	if len(active) == 0 {
		lines = append(lines, "", "No active jobs.")
	} else {
		lines = append(lines, "", "Active jobs:")
		for _, entry := range active {
			lines = append(lines, renderJobListEntry(entry)...)
		}
	}
	const recentTerminalLimit = 8
	if len(terminal) == 0 {
		return strings.Join(lines, "\n")
	}
	lines = append(lines, "")
	if len(terminal) > recentTerminalLimit {
		lines = append(lines, fmt.Sprintf("Recent terminal jobs: showing %d of %d", recentTerminalLimit, len(terminal)))
		for _, entry := range terminal[:recentTerminalLimit] {
			lines = append(lines, renderJobListEntry(entry)...)
		}
		lines = append(lines, "Use `job get JOB_ID` or `--json` for full history.")
		return strings.Join(lines, "\n")
	}
	lines = append(lines, fmt.Sprintf("Terminal jobs: %d", len(terminal)))
	for _, entry := range terminal {
		lines = append(lines, renderJobListEntry(entry)...)
	}
	return strings.Join(lines, "\n")
}

func renderJobListEntry(entry view.JobEntryView) []string {
	lines := []string{fmt.Sprintf("- %s [%s · %s] %s %s",
		entry.ID, entry.Surface.StateLabel, entry.Progress.PercentLabel, entry.Operation, entry.CanonicalPath)}
	lines = append(lines, renderTimingLines(entry.Timing)...)
	lines = append(lines, renderProgressLines(entry.Progress)...)
	if entry.Surface.ErrorLine != "" {
		lines = append(lines, "  Error: "+entry.Surface.ErrorLine)
	}
	return lines
}

// formatCountString is the render-side alias for thousands formatting; the
// value arrives pre-resolved, only the digit grouping happens here.
func formatCountString(value int32) string {
	return formatCount(value)
}
```

Delete `jobProgressDisplay`, `renderReconcileMagnitude`, `renderJobTimingLines`, `formatJobDuration` (replaced above). `renderCancelJob` keeps its current form for now (Task 7 moves it behind `MutationAckView`).

- [ ] **Step 2: Rewire the gRPC call sites**

In `internal/daemon/grpc_server.go`:

GetJob:

```go
	health := server.manager.DependencyHealth()
	successorID := server.manager.JobSuccessorID(job)
	entry := resolveJobEntry(job, health.Degraded(), successorID)
	return &pb.GetJobResponse{
		Job:              toJobWithTokens(job, health.Degraded(), successorID),
		DependencyHealth: toDependencyHealth(health),
		DisplayText:      server.envelopeText(ctx, health, renderGetJob(entry, true), "job_id", job.ID, "codebase_id", job.CodebaseID),
	}, nil
```

ListJobs:

```go
	jobs := server.manager.ListJobs(request.GetCodebaseId())
	health := server.manager.DependencyHealth()
	successors := buildJobSuccessors(jobs)
	summary := resolveListSummary(jobs, health.Degraded())
	activeEntries := make([]view.JobEntryView, 0, len(jobs))
	terminalEntries := make([]view.JobEntryView, 0, len(jobs))
	response := &pb.ListJobsResponse{Jobs: make([]*pb.Job, 0, len(jobs))}
	for _, job := range jobs {
		response.Jobs = append(response.Jobs, toJobWithTokens(job, health.Degraded(), successors[job.ID]))
		entry := resolveJobEntry(job, health.Degraded(), successors[job.ID])
		if isTerminalJobState(job.State) {
			terminalEntries = append(terminalEntries, entry)
		} else {
			activeEntries = append(activeEntries, entry)
		}
	}
	response.DependencyHealth = toDependencyHealth(health)
	response.DisplayText = server.envelopeText(ctx, health, renderListJobs(summary, activeEntries, terminalEntries), "codebase_id", request.GetCodebaseId())
	return response, nil
```

(Add `view` to the grpc_server.go imports.)

- [ ] **Step 3: Extend the render guard**

In `internal/daemon/render_guard_test.go`, extend the forbidden set: any selector on an identifier `progress`, `job`, or `codebase` reading the fields `FilesTotal`, `FilesProcessed`, `FilesAdded`, `FilesModified`, `FilesRemoved`, `FilesEmbedded`, `ChunksGenerated`, `ChunksReused`, `ChunksTotal`, `OverallPercent`, `Unit`, `StartedAt`, `UpdatedAt`, `CompletedAt`, `LastSuccessfulRun` is a violation in `render*.go`. Add to the `ast.Inspect` switch in `scanRenderFileForLaundering`:

```go
		case *ast.SelectorExpr:
			// existing checks stay; add:
			if isIdent(typed.X, "progress") || isIdent(typed.X, "job") || isIdent(typed.X, "codebase") {
				switch typed.Sel.Name {
				case "FilesTotal", "FilesProcessed", "FilesAdded", "FilesModified", "FilesRemoved",
					"FilesEmbedded", "ChunksGenerated", "ChunksReused", "ChunksTotal", "OverallPercent",
					"Unit", "StartedAt", "UpdatedAt", "CompletedAt", "LastSuccessfulRun":
					report(typed, "reads raw "+typed.Sel.Name+"; use the resolved view models")
				}
			}
```

Note: `renderIndexingActive`, `renderIndexedDetail`, `renderIndexedWithSync`, `backgroundSyncNote`, `isBackgroundSyncReconcile`, `headingFor`, `renderIndexedDescendantsHint`, `renderStartIndex`, and `renderSearch` still read raw fields at this point. They are converted in Tasks 6 and 7. To keep this task green, the guard addition lands in Task 7 Step 4 instead if it fires here; the executor moves this step to Task 7 in that case and notes it in the commit message. Run the guard now to find out:

Run: `go test ./internal/daemon/ -run TestRenderLayerDoesNotLaunderJobStatus -v`
If it fails on functions Task 6 and 7 convert, defer this guard extension to Task 7 Step 4.

- [ ] **Step 4: Update test expectations**

In `internal/daemon/render_test.go` and `internal/daemon/banner_test.go`, the compiler flags every old-signature call. Mechanical conversion for each:

- `renderGetJob(job, degraded, successor)` becomes `renderGetJob(resolveJobEntry(*job, degraded, successor), true)` for non-nil jobs, and the nil-job case becomes `renderGetJob(view.JobEntryView{}, false)`.
- `renderListJobs(jobs, degraded)` becomes the resolver pipeline used by the handler; add a small test helper in `render_test.go`:

```go
func renderListJobsForTest(jobs []model.Job, degraded bool) string {
	successors := buildJobSuccessors(jobs)
	summary := resolveListSummary(jobs, degraded)
	active := make([]view.JobEntryView, 0, len(jobs))
	terminal := make([]view.JobEntryView, 0, len(jobs))
	for _, job := range jobs {
		entry := resolveJobEntry(job, degraded, successors[job.ID])
		if isTerminalJobState(job.State) {
			terminal = append(terminal, entry)
		} else {
			active = append(active, entry)
		}
	}
	return renderListJobs(summary, active, terminal)
}
```

and point the list tests at it. Update string expectations that changed shape:

- `"📄 7 of 58 files · 🧩 84 chunks"` becomes `"📄 7 of 58 files embedded"` and a separate `"🧩 84 chunks added this run"` check (no RunMode set in fixtures means the default scope label is the bare unit and the verb is "embedded").
- `"Added 12 · Modified 30 · Removed 5"` becomes `"Changed since last sync: 12 files added · 30 modified · 5 removed"`.
- Percent and preparing-label expectations move from `Progress:` lines unchanged (PercentLabel carries them).

- [ ] **Step 5: Run, fix, commit**

Run: `go test ./internal/daemon/ ./internal/status/ ./internal/view/`
Expected: PASS after expectation updates.

```bash
git add internal/daemon/render.go internal/daemon/grpc_server.go internal/daemon/render_test.go internal/daemon/banner_test.go internal/daemon/render_guard_test.go
git commit -m "Render job surfaces from resolved view models with typed denominators and collection totals"
```

---### Task 6: Codebase status, search, acks, and doctor format resolved views

Converts the remaining raw readers: the statusView builder moves to the boundary, search results reduce at the boundary, the two stray grpc_server compositions move behind views, and every ack gets a view.

**Files:**
- Modify: `internal/daemon/status_render.go` (statusView type replaced by `view.StatusView`)
- Modify: `internal/daemon/render.go`
- Modify: `internal/daemon/status_present.go` (new resolvers)
- Modify: `internal/daemon/grpc_server.go`
- Test: `internal/daemon/present_views_test.go` (create)

- [ ] **Step 1: Move the statusView build to the boundary**

In `internal/daemon/status_present.go` add (this is the body of today's `renderIndexingActive` field-filling, relocated):

```go
// resolveStatusView builds the template view for an active or ready codebase.
// It is the relocated body of the render-side builder, so the templates keep
// their exact output. templateName selects among preparing, building,
// incremental, ready, waiting.
func resolveStatusView(codebase model.Codebase, activeJob *model.Job, display displayStatus, waitLabel string) (view.StatusView, string) {
	statusView := view.StatusView{
		Name:      filepath.Base(codebase.CanonicalPath),
		UpdatedAt: formatBoundaryStatusTime(codebase.UpdatedAt),
	}
	switch display {
	case displayWaiting:
		statusView.WaitLabel = waitLabel
		return statusView, "waiting.md.tmpl"
	case displayIndexed:
		if codebase.LastSuccessfulRun != nil {
			statusView.HasStats = true
			statusView.Files = codebase.LastSuccessfulRun.IndexedFiles
			statusView.Chunks = codebase.LastSuccessfulRun.TotalChunks
			statusView.SkippedLine = renderSkippedFiles(codebase.LastSuccessfulRun.SkippedFiles)
		}
		if activeJob != nil && isBackgroundSyncReconcile(&codebase, activeJob) {
			statusView.SyncNote = backgroundSyncNote(activeJob.Progress)
		}
		return statusView, "ready.md.tmpl"
	}
	statusView.PrepareLabel = prepareLabel(activeJob)
	embedding := false
	if activeJob != nil {
		progress := activeJob.Progress
		if !progress.LastEventAt.IsZero() {
			statusView.UpdatedAt = formatBoundaryStatusTime(progress.LastEventAt)
		}
		changed := progress.FilesAdded + progress.FilesModified + progress.FilesRemoved
		statusView.Percent = int32(progress.OverallPercent + 0.5)
		statusView.FilesProcessed = progress.FilesProcessed
		statusView.FilesTotal = progress.FilesTotal
		statusView.FilesInCodebase = progress.FilesInCodebase
		statusView.FilesChanged = changed
		statusView.FilesUnchanged = max(progress.FilesInCodebase-changed, 0)
		statusView.FilesReEmbedded = progress.FilesEmbedded
		statusView.FilesRemoved = progress.FilesRemoved
		statusView.FilesSkippedOversize = progress.FilesSkippedOversize
		statusView.FilesSkippedUnreadable = progress.FilesSkippedUnreadable
		statusView.FilesProcessedChanged = progress.FilesEmbedded + progress.FilesRemoved + progress.FilesSkippedOversize + progress.FilesSkippedUnreadable
		statusView.Heading = headingFor(codebase, activeJob)
		statusView.ChunksAdded = progress.ChunksGenerated
		statusView.ChunksReused = progress.ChunksReused
		statusView.ChunksEmbeddedThisRun = progress.ChunksGenerated
		statusView.ChunksTotal = max(progress.ChunksTotal, progress.ChunksReused+progress.ChunksGenerated)
		if statusView.ChunksTotal == 0 {
			statusView.ChunksTotal = codebase.LiveChunkTotal
		}
		if statusView.ChunksTotal == 0 && codebase.LastSuccessfulRun != nil {
			statusView.ChunksTotal = codebase.LastSuccessfulRun.TotalChunks
		}
		embedding = progress.FilesTotal > 0 || progress.FilesInCodebase > 0
	}
	if !embedding {
		return statusView, "preparing.md.tmpl"
	}
	if activeJob != nil && jobOperation(activeJob.Operation) == jobOperationIndex {
		return statusView, "building.md.tmpl"
	}
	return statusView, "incremental.md.tmpl"
}
```

Move `headingFor`, `prepareLabel`, `isBackgroundSyncReconcile`, `backgroundSyncNote`, `renderSkippedFiles` from render.go into status_present.go unchanged (they read model fields, so they are boundary code; `renderSkippedFiles` returns a string consumed by the view, rename it `skippedFilesLine`). Add `formatBoundaryStatusTime` next to `formatBoundaryTime` with the existing `formatStatusTime` body.

`renderIndexingActive`, `renderIndexedDetail`, `renderIndexedWithSync`, `renderWaiting` in render.go collapse to:

```go
func renderStatusBody(statusView view.StatusView, templateName string) string {
	return renderStatusTemplate(templateName, statusView)
}
```

and `renderGetIndexBody` takes a `view.GetIndexView`:

```go
func renderGetIndexBody(getIndex view.GetIndexView) string {
	if !getIndex.Tracked {
		return fmt.Sprintf("❌ Codebase '%s' is not indexed. Please use the index_codebase tool to index it first.", getIndex.RequestedPath)
	}
	switch getIndex.Display {
	case view.Display(displayFailed):
		return renderHistoricalFailure(getIndex.CanonicalPath, getIndex.Failure)
	case view.Display(displayMissing):
		return renderMissingStatus(getIndex.CanonicalPath)
	case view.Display(displayStale):
		return renderStaleStatus(getIndex.CanonicalPath, getIndex.Failure)
	default:
		return renderStatusBody(getIndex.Status, getIndex.TemplateName)
	}
}
```

`renderHistoricalFailure`, `renderStaleStatus`, `renderMissingStatus` change their first parameter from `*model.Codebase` to `canonicalPath string` (they only used the path; `renderStaleStatus` already takes the failure view and now reads `failure.FailedAtLabel`).

The boundary assembles `view.GetIndexView` in a new resolver in status_present.go:

```go
// resolveGetIndexView assembles the full codebase status response view.
func (manager *Manager) resolveGetIndexView(requestedPath string, tracked bool, codebase *model.Codebase, activeJob *model.Job, health dependencyHealth, classification *model.PathClassification, descendants []model.Codebase) view.GetIndexView {
	getIndex := view.GetIndexView{Tracked: tracked, RequestedPath: requestedPath}
	getIndex.ClassificationLine = classificationLine(classification)
	getIndex.ResolutionLines = pathResolutionLines(requestedPath)
	getIndex.DescendantsHint = descendantsHint(requestedPath, descendants)
	if !tracked || codebase == nil {
		return getIndex
	}
	getIndex.CanonicalPath = codebase.CanonicalPath
	display := computeDisplayStatus(*codebase, activeJob, health.Degraded())
	getIndex.Display = view.Display(display)
	getIndex.Failure = resolveCodebaseFailure(*codebase)
	statusView, templateName := resolveStatusView(*codebase, activeJob, display, waitingLabel(health.Mode))
	getIndex.Status = statusView
	getIndex.TemplateName = templateName
	return getIndex
}
```

`classificationLine`, `pathResolutionLines` (with its gitworktree and symlink reads), `descendantsHint` move from render.go to status_present.go with their current bodies (renamed from `renderClassificationLine`, `pathResolutionLines`, `renderIndexedDescendantsHint`; they return strings the view carries). `waitingLabel` moves too.

- [ ] **Step 2: Reduce search results at the boundary**

In status_present.go:

```go
// resolveSearchResults reduces stored chunks to the view shape.
func resolveSearchResults(chunks []model.StoredChunk) []view.SearchResultView {
	results := make([]view.SearchResultView, 0, len(chunks))
	for _, chunk := range chunks {
		results = append(results, view.SearchResultView{
			RelativePath: chunk.RelativePath,
			StartLine:    chunk.StartLine,
			EndLine:      chunk.EndLine,
			Language:     chunk.Language,
			Score:        chunk.Score,
			Content:      chunk.Content,
		})
	}
	return results
}
```

(If `StoredChunk` field names differ, use the names from `internal/model`; the compiler verifies.) Conversation results reduce the same way into `view.ConversationResultView`. `renderSearch` and `renderConversationSearch` change their view structs to the `view.SearchView` and `view.ConversationSearchView` shapes, with the boundary (grpc_server handlers) constructing them. The in-flight status section inside `renderSearch` uses `InFlightStatus`/`TemplateName` resolved at the boundary the same way as GetIndex.

- [ ] **Step 3: Acks and the stray compositions**

In status_present.go:

```go
// resolveStartIndexView assembles the start acknowledgment including the merge
// note (relocated from grpc_server.startIndexMergeNote).
func (server *GRPCServer) resolveStartIndexView(requestedPath string, codebase model.Codebase, job model.Job, deduplicated bool, overlapsCodebaseID string) view.StartIndexView {
	return view.StartIndexView{
		RequestedPath:      requestedPath,
		CanonicalPath:      codebase.CanonicalPath,
		CodebaseID:         codebase.ID,
		JobID:              job.ID,
		SplitterType:       job.Config.SplitterType,
		Deduplicated:       deduplicated,
		OverlapsCodebaseID: overlapsCodebaseID,
		MergeNote:          server.startIndexMergeNote(requestedPath, codebase),
	}
}
```

`renderStartIndex` formats `view.StartIndexView` only (same output text as today, fields substituted). The `SyncConversationManifest` inline `fmt.Sprintf` at grpc_server.go:538 becomes:

```go
	ack := view.MutationAckView{Kind: view.AckManifest, CollectionID: collectionID, NeededCount: len(needed), TotalCount: total}
	displayText := appendCorrelationRef(renderMutationAck(ack), ctx, ...)
```

with one renderer covering all acks:

```go
func renderMutationAck(ack view.MutationAckView) string {
	switch ack.Kind {
	case view.AckClear:
		return fmt.Sprintf("Cleared index for '%s'", ack.Path)
	case view.AckCancel:
		if ack.AlreadyTerminal {
			return fmt.Sprintf("Indexing job %s is already %s", ack.JobID, ack.StateLabel)
		}
		return "Canceled indexing job " + ack.JobID
	case view.AckSync:
		if ack.Deduplicated {
			return fmt.Sprintf("Sync request deduplicated onto active job %s for '%s'", ack.JobID, ack.Path)
		}
		return fmt.Sprintf("Started sync job %s for '%s'", ack.JobID, ack.Path)
	case view.AckRegisterConversation:
		return fmt.Sprintf("Registered conversation collection '%s' as %s (collection %s)", ack.CollectionID, ack.CodebaseID, ack.CollectionName)
	case view.AckUpsertConversation:
		return fmt.Sprintf("Started conversation ingest job %s for collection '%s' with %d %s.", ack.JobID, ack.CollectionID, ack.DocumentCount, pluralWord("document", ack.DocumentCount))
	case view.AckDeleteConversation:
		return fmt.Sprintf("Started conversation delete job %s for conversation '%s' in collection '%s'.", ack.JobID, ack.ConversationID, ack.CollectionID)
	case view.AckManifest:
		return fmt.Sprintf("Conversation collection '%s' needs %d of %d %s.", ack.CollectionID, ack.NeededCount, ack.TotalCount, pluralWord("conversation", ack.TotalCount))
	default:
		return ""
	}
}
```

IMPORTANT: before replacing each old renderer (`renderClearIndex`, `renderCancelJob`, `renderSyncIndex`, `renderRegisterConversationCollection`, `renderUpsertConversationDocuments`, `renderDeleteConversation`), read its current body and copy its exact wording into the matching case so no output text changes except where this plan says so. The texts above are reconstructions; the source of truth is the current function body. `pluralWord` is the existing `plural` helper renamed if needed. The boundary builds the ack (cancel resolves `StateLabel` via `status.JobStateLabelFor(job.State)` in the handler).

`renderDoctor` and `renderDroppedSection` merge into `renderDoctor(doctor view.DoctorView) string` with their current bodies appended; the handler builds `view.DoctorView{Diagnostics: diagnostics, Dropped: server.manager.DroppedCodebases()}`.

- [ ] **Step 4: Tests**

`internal/daemon/present_views_test.go`:

```go
package daemon

import (
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/view"
)

func TestResolveStatusViewFallsBackToLiveChunkTotal(t *testing.T) {
	t.Parallel()
	codebase := model.Codebase{CanonicalPath: "/repo", LiveChunkTotal: 33240}
	job := model.Job{State: model.JobStateRunning, Operation: "sync", Progress: model.Progress{FilesInCodebase: 100, FilesModified: 2}}
	statusView, templateName := resolveStatusView(codebase, &job, displayIndexing, "")
	if statusView.ChunksTotal != 33240 {
		t.Fatalf("ChunksTotal = %d, want the live total 33240", statusView.ChunksTotal)
	}
	if templateName != "incremental.md.tmpl" {
		t.Fatalf("template = %q, want incremental", templateName)
	}
}

func TestRenderMutationAckManifest(t *testing.T) {
	t.Parallel()
	out := renderMutationAck(view.MutationAckView{Kind: view.AckManifest, CollectionID: "clyde-conversations", NeededCount: 11, TotalCount: 1011})
	if !strings.Contains(out, "needs 11 of 1011") {
		t.Fatalf("manifest ack = %q, want the needed-of-total counts", out)
	}
}
```

Run: `go test ./internal/daemon/ ./internal/view/ ./internal/status/`
Expected: PASS after the mechanical call-site updates the compiler demands (every old call the conversion touched).

- [ ] **Step 5: Commit**

```bash
git add internal/daemon/ internal/view/
git commit -m "Resolve status, search, ack, and doctor views at the boundary"
```

---

### Task 7: The compile-time wall: move render to internal/render

After Task 6 every render function formats view types or plain strings. The move is now mechanical.

**Files:**
- Create: `internal/render/` (moved files)
- Move: `internal/daemon/render.go` to `internal/render/render.go`
- Move: `internal/daemon/status_render.go` to `internal/render/status_render.go`
- Move: `internal/daemon/banner.go` formatting half to `internal/render/banner.go`
- Move: `internal/daemon/templates/` to `internal/render/templates/`
- Create: `internal/render/imports_test.go`
- Modify: `internal/daemon/render_guard_test.go` (shrink), `internal/daemon/grpc_server.go` (call render.*)

- [ ] **Step 1: Move the files**

```bash
mkdir -p internal/render
git mv internal/daemon/render.go internal/render/render.go
git mv internal/daemon/status_render.go internal/render/status_render.go
git mv internal/daemon/templates internal/render/templates
```

Change `package daemon` to `package render` in the moved files. Export every function grpc_server calls (`RenderGetJob`, `RenderListJobs`, `RenderGetIndexBody`, `RenderSearch`, `RenderConversationSearch`, `RenderMutationAck`, `RenderStartIndex`, `RenderDoctor`, `RenderStatusBody`, `RenderHealthBanner`, timing/progress helpers stay unexported). Update the embed directive path in status_render.go (`//go:embed templates/status/*.md.tmpl` still resolves relative to the new file location).

`banner.go` splits: `bannerViewFor` (reads config and health) stays in the daemon as `resolveBannerView` returning `view.BannerView`; the template execution moves to render as `RenderHealthBanner(banner view.BannerView) string` (empty Headline returns ""). `envelopeText` in grpc_server calls `render.RenderHealthBanner(server.resolveBannerView(health))`.

Remove `goodkind.io/lm-semantic-search/internal/model` from every import block in `internal/render`. The compiler now reports any function the earlier tasks missed; convert each the same way (boundary resolver plus view field) rather than re-importing model.

- [ ] **Step 2: The import wall test**

`internal/render/imports_test.go`:

```go
package render

import (
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// The render package is behind a compile-time wall: it formats view models and
// must never read raw records. The compiler enforces this because the package
// does not import internal/model; this test makes the rule explicit and fails
// with a named violation if anyone re-adds the import.
func TestRenderImportsNoModel(t *testing.T) {
	t.Parallel()
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	fset := token.NewFileSet()
	for _, file := range files {
		parsed, parseErr := parser.ParseFile(fset, file, nil, parser.ImportsOnly)
		if parseErr != nil {
			t.Fatalf("parse %s: %v", file, parseErr)
		}
		for _, imported := range parsed.Imports {
			path := strings.Trim(imported.Path.Value, `"`)
			if strings.HasSuffix(path, "internal/model") || strings.HasSuffix(path, "internal/daemon") {
				t.Fatalf("%s imports %s; the render wall forbids raw record access", file, path)
			}
		}
	}
}
```

- [ ] **Step 3: Rewire the daemon**

Add `render "goodkind.io/lm-semantic-search/internal/render"` to grpc_server.go imports and prefix every moved call. Helpers the boundary still owns (`formatBoundaryTime`, `resolveStatusView`, all resolvers, `plural`) stay in the daemon. Run the compiler repeatedly until clean:

Run: `go build ./internal/... 2>&1` via `make build` at the end; during iteration `go vet ./internal/render ./internal/daemon` is faster.

- [ ] **Step 4: Shrink the old guard, extend coverage**

`internal/daemon/render_guard_test.go` is now pointless for render files (they moved); repurpose it to guard the boundary's only remaining presentation files if desired, or delete it and keep the wall test plus the cmd guard (Task 8). Apply the deferred Progress-field guard from Task 5 Step 3 here if it was deferred: it now guards nothing in daemon render files (gone), so delete the deferral note and rely on the import wall.

- [ ] **Step 5: Run everything and commit**

Run: `go test ./...`
Expected: all packages PASS, including `internal/render`.

```bash
git add internal/render/ internal/daemon/ docs/
git commit -m "Move the render layer behind a compile-time wall that forbids model imports"
```

---

### Task 8: Uniform envelope, TUI fallback removal, cmd guard

**Files:**
- Modify: `internal/daemon/grpc_server.go` (7 RPCs gain the envelope)
- Modify: `cmd/lm-semantic-search/codebase_list_tui.go`
- Create: `cmd/lm-semantic-search/display_guard_test.go`
- Test: `internal/daemon/banner_test.go` (envelope coverage)

- [ ] **Step 1: Uniform envelope**

In each of StartIndex, ClearIndex, CancelJob, SyncIndex, RegisterConversationCollection, SyncConversationManifest, UpsertConversationDocuments, DeleteConversation handlers, replace `appendCorrelationRef(body, ctx, ...)` with:

```go
	health := server.manager.DependencyHealth()
	displayText := server.envelopeText(ctx, health, body, extras...)
```

(keeping each handler's existing extras). Add a test in banner_test.go:

```go
// Every text-bearing mutation RPC shows the degraded banner, not only the read
// surfaces: a StartIndex during an outage must carry the same warning.
func TestStartIndexShowsBannerWhenDegraded(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	manager.runner = fakeRunner{}
	manager.mu.Lock()
	manager.health = dependencyHealth{Mode: dependencyEmbedderUnreachable, Since: clock.Now(), LastHealthyAt: clock.Now()}
	manager.mu.Unlock()
	server := NewGRPCServer(manager, nil)
	resp, err := server.StartIndex(context.Background(), &pb.StartIndexRequest{Path: repoPath})
	if err != nil {
		t.Fatalf("StartIndex returned error: %v", err)
	}
	if !strings.Contains(resp.GetDisplayText(), "🟥") {
		t.Fatalf("StartIndex display text lacks the degraded banner:\n%s", resp.GetDisplayText())
	}
}
```

- [ ] **Step 2: TUI fallbacks out**

In `cmd/lm-semantic-search/codebase_list_tui.go` `renderRow` (lines 345-374): delete the `GetStatus()` fallback branch and the label fallback; the row uses `GetDisplayStatus()`, `GetGlyphToken()`, `GetStatusLabel()` unconditionally (empty renders empty, which only happens against a daemon older than this change, and the CLI ships with the daemon). The `statusColors` map re-keys on the resolved display values only (`preparing`, `indexing`, `waiting`, `indexed`, `stale`, `failed`, `missing`); delete the `"not_indexed"` entry.

- [ ] **Step 3: cmd guard**

`cmd/lm-semantic-search/display_guard_test.go`:

```go
package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// The CLI renders only resolved display fields from the daemon. Reading the
// raw lifecycle fields for display is how the TUI forked its own status
// vocabulary once; this guard makes that a test failure.
func TestCLIDisplayDoesNotReadRawStatusFields(t *testing.T) {
	t.Parallel()
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	fset := token.NewFileSet()
	var violations []string
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		parsed, parseErr := parser.ParseFile(fset, file, nil, 0)
		if parseErr != nil {
			t.Fatalf("parse %s: %v", file, parseErr)
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			selector, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			if selector.Sel.Name == "GetStatus" || selector.Sel.Name == "GetState" {
				position := fset.Position(node.Pos())
				violations = append(violations, file+":"+position.String()+": calls "+selector.Sel.Name)
			}
			return true
		})
	}
	if len(violations) > 0 {
		t.Fatalf("CLI reads raw status fields for display:\n%s", strings.Join(violations, "\n"))
	}
}
```

- [ ] **Step 4: Run and commit**

Run: `go test ./cmd/... ./internal/daemon/`
Expected: PASS (the guard passes because Step 2 removed the only raw reads).

```bash
git add internal/daemon/grpc_server.go internal/daemon/banner_test.go cmd/lm-semantic-search/
git commit -m "Show the banner on every text RPC and remove the TUI's raw status fallbacks"
```

---

### Task 9: Full verification

- [ ] **Step 1:** `go test ./...` (expect all ok; cold-build daemon timeout under `./...` is a known contention effect; re-run `go test ./internal/daemon/ -count=1` to confirm in isolation if it appears).
- [ ] **Step 2:** `make lint` (fix any exhaustruct findings by fully keying new struct literals; fix De Morgan or exhaustive-switch findings as the linters direct; never silence).
- [ ] **Step 3:** `make build` (expect codesign ok).
- [ ] **Step 4:** Live smoke: `make deploy`, then `lm-semantic-search job list` against the running daemon. Verify: a running job shows a heading, a typed denominator, the embedded/already-indexed split, and a chunks line with the collection total; the ghost `/chat:/clyde-conversations` record is gone from `codebase list`; no `daemon-resume` job for a `chat:` path appears after a daemon restart.
- [ ] **Step 5:** Commit any verification fixes:

```bash
git add -A
git commit -m "Finalize presentation choke point: lint, build, and live smoke clean"
```

---

## Self-Review

**Spec coverage:** wall (Tasks 2, 7), boundary resolvers (Tasks 4, 6), ProgressSurface semantics and corpus totals (Tasks 3, 4, 5), typed denominator and resume wording (Tasks 3, 4, 5), uniform envelope (Task 8), TUI and cmd guard (Task 8), riders (Task 1), search reduction (Task 6), stray grpc compositions (Task 6), import wall test (Task 7), migration order matches the spec with riders pulled first because they fix an active boot bug.

**Placeholder scan:** none. Task 6 Step 3 instructs copying current ack wordings from the live function bodies rather than trusting reconstructions; that is an instruction to preserve exact behavior, with complete fallback text provided.

**Type consistency:** `view.JobSurface`/`view.FailureSurface` field names match the existing `status.JobSurface`/`codebaseFailureView` consumers (`StateLabel`, `ErrorLine`, `Superseded`, `SupersededByJobID`, `HasFailure`, `Message`, `JobID`, `TraceID`; `FailedAtLabel` replaces `FailedAt` with one named call-site change). `resolveJobEntry(job, bool, string) view.JobEntryView` is used identically in Tasks 4, 5. `formatCount` defined in Task 4, aliased in render in Task 5, and moves with render in Task 7 (the alias body moves; the boundary keeps its own copy since both packages need digit grouping; if lint flags duplication, export `view.FormatCount` instead and use it from both sides). `resolveStatusView(codebase, activeJob, display, waitLabel) (view.StatusView, string)` is used in Tasks 6 and 7. RunMode constants defined once in `internal/model` (Task 3) and consumed in Task 4.
