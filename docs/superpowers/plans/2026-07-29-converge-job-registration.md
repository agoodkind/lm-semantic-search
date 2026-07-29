# Converge job registration implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `lm-semantic-search job list`, `job get`, and `job cancel` reach the indexing the daemon does when a watched file changes.

**Architecture:** The watcher already runs its work through `Manager.ConvergePaths`. That call registers nothing, so no job record exists. This adds a job around it, using the same job store, journal, and cancel registry the periodic sync already uses. The work stays path-scoped; only registration is added.

**Tech Stack:** Go 1.26, gRPC, the daemon's existing job store in `internal/daemon`.

## Global Constraints

Every task inherits these. They come from `AGENTS.md` in the repository root and from `docs/superpowers/specs/2026-07-29-converge-job-registration-design.md`.

- Do not add a second registration mechanism. Reuse the job store, the journal, and the cancel registry that `StartIndex` and `SyncIndex` already use.
- Do not route the watcher through `SyncIndex`. That diffs a whole codebase; a converge reads only the paths it was handed.
- No `any`, `interface{}`, `map[string]any`, or `[]any` in production code. Use named types.
- A counter must mean what its name says. A converge does not measure embedding, so it reports no embedded files and therefore never clears a degraded dependency banner.
- Run `make check` and `make test` before committing. Every `make` needs `GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile`.
- Targeted `go test` needs `PKG_CONFIG_PATH=<worktree>/.make/cgo/darwin-arm64/lib/pkgconfig` and `CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path'` exported.
- Commit with `git commit -S`. Subject line in imperative mood, no trailing period, with `Co-authored-by: Claude <noreply@anthropic.com>` after a blank line.

## File structure

| File | Responsibility after this plan |
| --- | --- |
| `internal/daemon/converge.go` | Converges named paths, reports what it converged, and stops between paths when its context is cancelled. |
| `internal/daemon/background_sync.go` | Runs the watcher's converge, and now registers it as a job first. |
| `internal/daemon/converge_test.go` | Covers the outcome counts and the cancellation stop. |
| `internal/daemon/background_sync_job_test.go` | Covers the job appearing, resolving, and cancelling. |
| `test/offlinelive/status_live_test.go` | Proves the three job commands reach a real converge on a running daemon. |

---

### Task 1: ConvergePaths reports what it converged and stops when cancelled

`ConvergePaths` returns only an error today, so a caller cannot tell how many paths it handled. It also walks its whole path list regardless of its context, so a cancel cannot stop it. Both are needed before a job can wrap it.

**Files:**
- Modify: `internal/daemon/converge.go:28`
- Create: `internal/daemon/converge_test.go`
- Modify: `internal/daemon/background_sync.go:485` (call site)
- Modify: `internal/daemon/admission_test.go:242`, `internal/daemon/converge_concurrency_test.go:646`, `internal/daemon/quarantine_test.go:57`, `internal/daemon/quarantine_test.go:111`, `internal/daemon/reuse_realstack_test.go:305`, `internal/daemon/reuse_realstack_test.go:355`, `internal/daemon/reuse_realstack_test.go:406` (call sites)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ConvergeOutcome` with fields `PathsGiven int32` and `PathsConverged int32`, and the new signature `func (manager *Manager) ConvergePaths(ctx context.Context, codebaseID string, relativePaths []string) (ConvergeOutcome, error)`.

- [ ] **Step 1: Write the failing test for the outcome counts**

Create `internal/daemon/converge_test.go`:

```go
package daemon

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

// TestConvergePathsReportsWhatItConverged proves the caller learns how many of
// the paths it handed over actually reached the index. A job built around this
// call reports that count as its scope, so a wrong count is a wrong status.
func TestConvergePathsReportsWhatItConverged(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
	manager.semantic = &fakeSemantic{
		reindex: func(context.Context, string, []model.StoredChunk, []string) error { return nil },
	}

	present := filepath.Join(repoPath, "present.go")
	if err := os.WriteFile(present, []byte("package main\n\nfunc Present() {}\n"), 0o644); err != nil {
		t.Fatalf("write the present file: %v", err)
	}

	outcome, err := manager.ConvergePaths(context.Background(), codebase.ID, []string{"present.go", "absent.go"})
	if err != nil {
		t.Fatalf("ConvergePaths returned error: %v", err)
	}
	if outcome.PathsGiven != 2 {
		t.Fatalf("PathsGiven = %d, want 2", outcome.PathsGiven)
	}
	if outcome.PathsConverged != 1 {
		t.Fatalf("PathsConverged = %d, want 1; only present.go exists on disk", outcome.PathsConverged)
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd ~/.worktrees/-Users-agoodkind-Sites-lm-semantic-search/status-command
export PKG_CONFIG_PATH="$PWD/.make/cgo/darwin-arm64/lib/pkgconfig"
export CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path'
go test ./internal/daemon/ -run TestConvergePathsReportsWhatItConverged -count=1
```

Expected: a compile failure saying `ConvergePaths` returns one value, not two.

- [ ] **Step 3: Add the outcome type and change the signature**

In `internal/daemon/converge.go`, above `ConvergePaths`, add:

```go
// ConvergeOutcome reports what one ConvergePaths call handled, so a caller can
// state the size of the work rather than guess it. PathsConverged counts the
// paths that changed the index, which is fewer than PathsGiven whenever a path
// was already current, was deleted before the call ran, or the call stopped
// early on cancellation.
//
// It carries no embedded-chunk count. ConvergePaths does not measure embedding,
// and a count that meant "paths seen" would be read as "work embedded" by the
// dependency-health gate in updateJobCompleted.
type ConvergeOutcome struct {
	PathsGiven     int32
	PathsConverged int32
}
```

Change the signature and every early return. The function currently returns `(err error)`; make it `(outcome ConvergeOutcome, err error)`. Each existing `return nil` becomes `return outcome, nil`, and the snapshot-write failure becomes `return outcome, fmt.Errorf(...)`.

Set `outcome.PathsGiven` immediately after `orderedPaths` is built:

```go
	orderedPaths := orderPathsByPresence(codebase.CanonicalPath, relativePaths)
	outcome.PathsGiven = int32(len(orderedPaths))
```

Count converged paths in the existing loop:

```go
	changed := false
	for _, relativePath := range orderedPaths {
		if converged := manager.convergeOnePath(ctx, codebase, relativePath, &snapshot, admission); converged {
			changed = true
			outcome.PathsConverged++
		}
	}
```

- [ ] **Step 4: Update the seven call sites**

The one production call site in `internal/daemon/background_sync.go:485` becomes:

```go
	if _, err := syncer.manager.ConvergePaths(ctx, codebaseID, relativePaths); err != nil {
		slog.ErrorContext(ctx, "watcher.converge_failed", "component", "daemon", "subcomponent", "watcher", "codebase_id", codebaseID, "err", err)
	}
```

Each of the six test call sites changes from `err := manager.ConvergePaths(...)` to `_, err := manager.ConvergePaths(...)`. The files and lines are listed under **Files** above.

- [ ] **Step 5: Run the test to verify it passes**

Run:

```bash
go test ./internal/daemon/ -run TestConvergePathsReportsWhatItConverged -count=1
```

Expected: PASS.

- [ ] **Step 6: Write the failing test for cancellation**

Append to `internal/daemon/converge_test.go`:

```go
// TestConvergePathsStopsBetweenPathsOnCancel proves a cancelled converge stops
// rather than finishing its list. The paths it did not reach keep their previous
// index entries, which the periodic sync repairs on its next pass.
func TestConvergePathsStopsBetweenPathsOnCancel(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel as soon as the first path reaches the store, so the second path is
	// never read.
	manager.semantic = &fakeSemantic{
		reindex: func(context.Context, string, []model.StoredChunk, []string) error {
			cancel()
			return nil
		},
	}

	names := []string{"first.go", "second.go"}
	for _, name := range names {
		body := "package main\n\nfunc " + name[:len(name)-3] + "() {}\n"
		if err := os.WriteFile(filepath.Join(repoPath, name), []byte(body), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	outcome, err := manager.ConvergePaths(ctx, codebase.ID, names)
	if err != nil {
		t.Fatalf("ConvergePaths returned error: %v", err)
	}
	if outcome.PathsGiven != 2 {
		t.Fatalf("PathsGiven = %d, want 2", outcome.PathsGiven)
	}
	if outcome.PathsConverged >= 2 {
		t.Fatalf("PathsConverged = %d, want fewer than 2; the cancel did not stop the loop", outcome.PathsConverged)
	}
}
```

- [ ] **Step 7: Run it to verify it fails**

Run:

```bash
go test ./internal/daemon/ -run TestConvergePathsStopsBetweenPathsOnCancel -count=1
```

Expected: FAIL reporting `PathsConverged = 2`, because the loop ignores the context.

- [ ] **Step 8: Check for cancellation before each path**

In `internal/daemon/converge.go`, add the check at the top of the loop body:

```go
	changed := false
	for _, relativePath := range orderedPaths {
		// A cancel stops the walk here rather than mid-path, so the snapshot
		// written below covers exactly the paths that reached the index. The
		// paths not reached become drift, which the periodic sync repairs.
		if ctx.Err() != nil {
			break
		}
		if converged := manager.convergeOnePath(ctx, codebase, relativePath, &snapshot, admission); converged {
			changed = true
			outcome.PathsConverged++
		}
	}
```

The existing snapshot write below the loop stays unchanged. It already runs only when `changed` is true, so a converge stopped before its first path writes nothing.

- [ ] **Step 9: Run both tests to verify they pass**

Run:

```bash
go test ./internal/daemon/ -run 'TestConvergePaths' -count=1
```

Expected: PASS for both.

- [ ] **Step 10: Run the full package and commit**

Run:

```bash
go test ./internal/daemon/ -count=1
make check GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile
```

Expected: both pass. Then:

```bash
git add internal/daemon/converge.go internal/daemon/converge_test.go \
  internal/daemon/background_sync.go internal/daemon/admission_test.go \
  internal/daemon/converge_concurrency_test.go internal/daemon/quarantine_test.go \
  internal/daemon/reuse_realstack_test.go
git commit -S -m "Report what a converge handled and stop it between paths

ConvergePaths returned only an error, so a caller could not state how much
work it did. It also walked its whole path list regardless of its context,
so nothing could stop it.

Return a ConvergeOutcome carrying the paths given and the paths converged,
and check for cancellation before each path. A stopped converge writes the
snapshot for the paths it reached; the rest become drift the periodic sync
repairs.

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 2: The watcher registers its converge as a job

The watcher builds a `watch-<codebase-id>-<unix>` string as a logging label and calls `ConvergePaths` directly. Nothing writes a job record, so `job list` reports zero active while the daemon indexes.

**Files:**
- Modify: `internal/daemon/background_sync.go:436-488`
- Create: `internal/daemon/background_sync_job_test.go`

**Interfaces:**
- Consumes: `ConvergeOutcome` and the two-value `ConvergePaths` from Task 1.
- Produces: `func (syncer *BackgroundSync) registerConvergeJob(ctx context.Context, codebaseID string, pathCount int) (convergeRegistration, bool)`, where `convergeRegistration` holds the job, the context the converge must run under, and a `release` function that cancels it and drops it from the registry.

- [ ] **Step 1: Write the failing test**

Create `internal/daemon/background_sync_job_test.go`:

```go
package daemon

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
)

// TestWatcherConvergeAppearsInJobList proves file-change indexing carries a job
// record, so job list reports it, job get resolves it, and job cancel can reach
// it. Before this, the work ran under a logging label that no command could
// address.
func TestWatcherConvergeAppearsInJobList(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	// Hold the converge inside the store call so the assertions run while the
	// job is live rather than after it finished.
	running := make(chan struct{})
	release := make(chan struct{})
	manager.semantic = &fakeSemantic{
		reindex: func(context.Context, string, []model.StoredChunk, []string) error {
			close(running)
			<-release
			return nil
		},
	}

	if err := os.WriteFile(filepath.Join(repoPath, "edited.go"), []byte("package main\n\nfunc Edited() {}\n"), 0o644); err != nil {
		t.Fatalf("write the edited file: %v", err)
	}

	syncer := NewBackgroundSync(config.Config{}, manager)
	go syncer.convergeViaWatcher(context.Background(), codebase.ID, []string{"edited.go"})
	<-running

	jobs := manager.ListJobs(codebase.ID)
	var converge *model.Job
	for index := range jobs {
		if jobs[index].Operation == "converge" {
			converge = &jobs[index]
			break
		}
	}
	if converge == nil {
		t.Fatalf("no converge job in the list of %d jobs for this codebase", len(jobs))
	}
	if converge.State != model.JobStateRunning {
		t.Fatalf("converge job state = %q, want running", converge.State)
	}
	if _, found := manager.GetJob(converge.ID); !found {
		t.Fatalf("GetJob(%q) found nothing, so job get would return NotFound", converge.ID)
	}

	close(release)
}
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
go test ./internal/daemon/ -run TestWatcherConvergeAppearsInJobList -count=1
```

Expected: FAIL with "no converge job in the list of 0 jobs for this codebase".

- [ ] **Step 3: Add the registration helper**

In `internal/daemon/background_sync.go`, add above `convergeViaWatcher`:

```go
// registerConvergeJob creates the job record for one watcher converge and
// stores its cancel function under the job id, which is what lets CancelJob
// reach the running work. It reuses newQueuedJob and the manager's cancel
// registry rather than adding a second mechanism.
//
// The job carries operation "converge" and a scope of the path count, because a
// converge reads only the paths it was handed rather than the whole codebase.
// Registration failing is not fatal: the converge still runs, unregistered,
// which is the behavior that existed before it registered at all.
// convergeRegistration carries what one registered converge needs: the job
// record, the context to run under, and the cleanup to run when it ends. The
// context and the registry entry come from one cancel function, so nothing can
// store one cancel and run under another.
type convergeRegistration struct {
	job     model.Job
	ctx     context.Context
	release func()
}

func (syncer *BackgroundSync) registerConvergeJob(
	ctx context.Context,
	codebaseID string,
	pathCount int,
) (convergeRegistration, bool) {
	syncer.manager.mu.Lock()

	codebase, found := syncer.manager.codebases[codebaseID]
	if !found {
		syncer.manager.mu.Unlock()
		return convergeRegistration{}, false
	}

	job := newQueuedJob(
		codebaseID,
		codebase.CanonicalPath,
		codebase.CanonicalPath,
		model.ClientInfo{Name: "daemon-watcher", PID: 0},
		"converge",
		false,
		codebase.EffectiveConfig,
		model.AdmissionBudget{},
		clock.Now(),
	)
	job.Progress.FilesTotal = int32(pathCount)
	job.Progress.ScopeUnit = "path"

	if err := syncer.manager.appendJobLocked("job_queued", job); err != nil {
		syncer.manager.mu.Unlock()
		slog.ErrorContext(ctx, "watcher.job_register_failed", "component", "daemon", "subcomponent", "watcher", "codebase_id", codebaseID, "err", err)
		return convergeRegistration{}, false
	}

	// One cancel function serves both purposes: it derives the context the
	// converge runs under, and it is what CancelJob calls. Deriving a second
	// context anywhere would leave CancelJob cancelling something the converge
	// does not observe.
	jobCtx, cancel := context.WithCancel(ctx)
	syncer.manager.cancels[job.ID] = cancel
	syncer.manager.mu.Unlock()

	release := func() {
		cancel()
		syncer.manager.mu.Lock()
		delete(syncer.manager.cancels, job.ID)
		syncer.manager.mu.Unlock()
	}
	return convergeRegistration{job: job, ctx: jobCtx, release: release}, true
}
```

- [ ] **Step 4: Wire the registration into convergeViaWatcher**

Replace the correlation block and the final call in `convergeViaWatcher`. The registration happens after the defer-for-first-build check and after `beginConverge` succeeds, so a converge that never runs creates no job.

The block from `syncer.markConvergeRunning(codebaseID)` to the end of the function becomes:

```go
	syncer.markConvergeRunning(codebaseID)

	registration, registered := syncer.registerConvergeJob(ctx, codebaseID, len(relativePaths))
	if registered {
		// Run under the job's context, which is the one CancelJob cancels.
		ctx = registration.ctx
		defer registration.release()
		ctx = spans.Attach(ctx, correlation.IdentityAttribute{Key: "job_id", Value: registration.job.ID})
		syncer.manager.updateJobRunning(registration.job)
	}

	outcome, err := syncer.manager.ConvergePaths(ctx, codebaseID, relativePaths)
	switch {
	case err != nil:
		slog.ErrorContext(ctx, "watcher.converge_failed", "component", "daemon", "subcomponent", "watcher", "codebase_id", codebaseID, "err", err)
		if registered {
			syncer.manager.updateJobFailed(context.WithoutCancel(ctx), registration.job.ID, err)
		}
	case !registered:
		// The converge ran unregistered, which is the behavior that existed
		// before registration and is preserved so a registration failure does
		// not also lose the work.
	case ctx.Err() != nil:
		syncer.manager.updateJobCancelled(context.WithoutCancel(ctx), registration.job.ID)
	default:
		// FilesEmbedded stays zero: a converge does not measure embedding, and
		// updateJobCompleted reads that field as evidence the store was
		// reachable when it decides whether to clear a degraded banner.
		syncer.manager.updateJobCompleted(ctx, registration.job.ID, indexer.Result{
			IndexedFiles: outcome.PathsConverged,
		})
	}
```

Replace the correlation attribute so the label is the real job id rather than a fabricated string. Move the correlation construction to after registration:

```go
	corr := correlation.New("").WithIdentityAttributes(
		correlation.IdentityAttribute{Key: "origin", Value: "watcher"},
		correlation.IdentityAttribute{Key: "codebase_id", Value: codebaseID},
	)
	ctx = correlation.WithContext(ctx, corr)
```

Then after `registered` is known, attach the job id:

```go
	if registered {
		ctx = spans.Attach(ctx, correlation.IdentityAttribute{Key: "job_id", Value: job.ID})
	}
```

Add `"goodkind.io/lm-semantic-search/internal/indexer"` and `"goodkind.io/lm-semantic-search/internal/spans"` to the import block if they are absent, and remove `"fmt"` if the removed `fmt.Sprintf` was its only use.

- [ ] **Step 5: Run the test to verify it passes**

Run:

```bash
go test ./internal/daemon/ -run TestWatcherConvergeAppearsInJobList -count=1
```

Expected: PASS.

- [ ] **Step 6: Write the failing test for cancel**

Append to `internal/daemon/background_sync_job_test.go`:

```go
// TestWatcherConvergeStopsWhenCancelled proves job cancel reaches the running
// converge, which is the third of the three commands the operator could not use.
func TestWatcherConvergeStopsWhenCancelled(t *testing.T) {
	t.Parallel()

	manager, _, repoPath := newTestManager(t)
	indexConfig := defaultIndexConfig()
	indexConfig.IgnoreDigest = digestIndexConfig(indexConfig)
	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = indexConfig
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	running := make(chan struct{})
	var once sync.Once
	manager.semantic = &fakeSemantic{
		reindex: func(ctx context.Context, _ string, _ []model.StoredChunk, _ []string) error {
			once.Do(func() { close(running) })
			<-ctx.Done()
			return ctx.Err()
		},
	}

	for _, name := range []string{"a.go", "b.go"} {
		if err := os.WriteFile(filepath.Join(repoPath, name), []byte("package main\n"), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	syncer := NewBackgroundSync(config.Config{}, manager)
	finished := make(chan struct{})
	go func() {
		syncer.convergeViaWatcher(context.Background(), codebase.ID, []string{"a.go", "b.go"})
		close(finished)
	}()
	<-running

	jobs := manager.ListJobs(codebase.ID)
	var jobID string
	for _, job := range jobs {
		if job.Operation == "converge" {
			jobID = job.ID
			break
		}
	}
	if jobID == "" {
		t.Fatal("no converge job to cancel")
	}

	if _, err := manager.CancelJob(context.Background(), jobID); err != nil {
		t.Fatalf("CancelJob returned error: %v", err)
	}

	select {
	case <-finished:
	case <-time.After(10 * time.Second):
		t.Fatal("the converge did not stop within ten seconds of the cancel")
	}

	cancelled, found := manager.GetJob(jobID)
	if !found {
		t.Fatalf("GetJob(%q) found nothing after the cancel", jobID)
	}
	if cancelled.State != model.JobStateCancelled {
		t.Fatalf("job state = %q, want cancelled", cancelled.State)
	}
}
```

Add `"sync"` and `"time"` to that file's imports.

- [ ] **Step 7: Run it to verify it fails, then passes**

Run:

```bash
go test ./internal/daemon/ -run TestWatcherConvergeStopsWhenCancelled -count=1
```

If it fails because the cancel does not reach the converge, the wiring in Step 4 stored the wrong cancel function. The one stored under `job.ID` must be the one whose context `ConvergePaths` receives.

Expected after correct wiring: PASS.

- [ ] **Step 8: Run the package under the race detector**

Run:

```bash
go test -race ./internal/daemon/ -run 'TestWatcherConverge|TestConvergePaths' -count=1
```

Expected: PASS with no race reports. The converge writes job state from its own goroutine while `ListJobs` reads it from the test goroutine, so this is the check that matters.

- [ ] **Step 9: Run the full gate and commit**

Run:

```bash
go test ./internal/daemon/ -count=1
make check GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile
make test GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile
```

Expected: all pass. Then:

```bash
git add internal/daemon/background_sync.go internal/daemon/background_sync_job_test.go
git commit -S -m "Register the watcher's converge as a job

Saving a file made the daemon index it under a correlation label that no
command could address, so job list reported nothing active, job get on that
label returned NotFound, and nothing could cancel the work.

Create a job before the converge runs, carrying operation converge and the
path count as its scope, and store its cancel function under the job id so
CancelJob reaches it. The work stays path-scoped; only registration is
added.

The job reports no embedded files, because a converge does not measure
embedding and that count is read as evidence a degraded dependency
recovered.

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 3: Prove the three commands reach a real converge on a running daemon

The unit tests drive `convergeViaWatcher` directly. The reported defect was observed through the CLI against a live daemon, so the close needs a check at that boundary.

**Files:**
- Modify: `test/offlinelive/status_live_test.go`

**Interfaces:**
- Consumes: the job registration from Task 2.
- Produces: nothing later tasks use.

- [ ] **Step 1: Write the failing live test**

Append to `test/offlinelive/status_live_test.go`:

```go
// TestWatcherConvergeIsAddressableByTheJobCommands proves the three commands an
// operator uses reach file-change indexing on a running daemon.
//
// The reported defect was observed exactly here: job list reported zero active
// while the daemon indexed a saved file, and job get on the identifier in the
// logs returned NotFound.
func TestWatcherConvergeIsAddressableByTheJobCommands(t *testing.T) {
	harness := newHarness(t)

	job := harness.indexFixture()
	requireCompleted(t, job)

	indexed := harness.indexStatus()
	codebaseID := indexed.GetCodebase().GetId()
	if codebaseID == "" {
		t.Fatal("the fixture codebase has no id")
	}

	// Change a file the daemon watches, which is what a save does.
	edited := filepath.Join(harness.fixturePath, "watched_edit.go")
	body := "package fixture\n\n// Edited exists to raise a watcher event.\nfunc Edited() string { return \"edited\" }\n"
	if err := os.WriteFile(edited, []byte(body), 0o644); err != nil {
		t.Fatalf("write the edited file: %v", err)
	}
	t.Cleanup(func() { _ = os.Remove(edited) })

	// Poll job list until a converge appears, which is the assertion: before
	// this change it never did.
	deadline := time.Now().Add(90 * time.Second)
	var convergeID string
	for time.Now().Before(deadline) {
		listed, err := harness.client.ListJobs(context.Background(), &pb.ListJobsRequest{CodebaseId: codebaseID})
		if err != nil {
			t.Fatalf("ListJobs returned error: %v", err)
		}
		for _, candidate := range listed.GetJobs() {
			if candidate.GetOperation() == "converge" {
				convergeID = candidate.GetId()
				break
			}
		}
		if convergeID != "" {
			break
		}
		time.Sleep(250 * time.Millisecond)
	}
	if convergeID == "" {
		t.Fatal("no converge job appeared in job list within ninety seconds of the file change")
	}

	got, err := harness.client.GetJob(context.Background(), &pb.GetJobRequest{JobId: convergeID})
	if err != nil {
		t.Fatalf("GetJob(%q) returned error, so job get would fail for the operator: %v", convergeID, err)
	}
	if got.GetJob().GetId() != convergeID {
		t.Fatalf("GetJob returned job %q, want %q", got.GetJob().GetId(), convergeID)
	}
}
```

Add `"os"`, `"path/filepath"`, and `"time"` to that file's imports if they are absent.

- [ ] **Step 2: Run it to verify it fails on the pre-change build**

Task 2 is already committed by this point, so the working tree is clean and stashing would stash nothing. Run the new test against a throwaway checkout of `origin/main` instead, which mutates nothing in the working worktree:

```bash
probe="$HOME/.worktrees/-Users-agoodkind-Sites-lm-semantic-search/converge-probe"
git worktree add --detach "$probe" origin/main
cp test/offlinelive/status_live_test.go "$probe/test/offlinelive/status_live_test.go"
( cd "$probe" \
  && export PKG_CONFIG_PATH="$probe/.make/cgo/darwin-arm64/lib/pkgconfig" \
  && export CGO_LDFLAGS_ALLOW='-Wl,-rpath,@loader_path' \
  && go test -tags offlinelive ./test/offlinelive/ -run TestWatcherConvergeIsAddressableByTheJobCommands -count=1 )
git worktree remove --force "$probe"
```

Expected: FAIL with "no converge job appeared in job list". That is the reported defect, reproduced through the same commands the operator used, against code that predates the fix.

If the probe worktree has no built cgo prerequisites, run `make build GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile` inside it first. If that is more setup than the evidence is worth, skip this step and say so in the commit body: Task 1 and Task 2 each already proved their own layer red before green, and nothing but the new registration path creates a job whose operation is `converge`, so the assertion cannot pass on the pre-change build by construction.

Nothing else may run while this executes. The harness isolation guard runs `pgrep -f lm-semantic-search-daemon`, which matches any command line containing that string, including a linter invoked on repository paths, and reports it as a disappeared production daemon.

- [ ] **Step 3: Run it against the change**

Run:

```bash
go test -tags offlinelive ./test/offlinelive/ -run TestWatcherConvergeIsAddressableByTheJobCommands -count=1
```

Expected: PASS.

If it fails because the watcher never fired, confirm `FileWatcherEnabled` is true in the harness config. The offline harness may disable it; if so, call `syncer.convergeViaWatcher` through the daemon's exported surface instead of relying on a filesystem event, and say so in the test comment.

- [ ] **Step 4: Run the whole live suite and commit**

Run:

```bash
make offline-live GO_MK_DEV_DIR=/Users/agoodkind/Sites/go-makefile
```

Expected: PASS. Then:

```bash
git add test/offlinelive/status_live_test.go
git commit -S -m "Prove the job commands reach a watcher converge on a live daemon

The defect was reported through the command line against a running daemon,
so the close is checked there. Change a watched file, then require a
converge job to appear in job list and resolve through job get.

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

## Not in this plan

Moving the journal append off the manager lock. `appendJobLocked` holds `manager.mu` across a file open, encode, and close, which serializes every manager operation behind a disk write. That is a performance property of every job path rather than something this change needs, and the three job commands reach a converge without it. It belongs with the journal work in LMS-1.

Journal retention and compaction, tracked as LMS-1.

## Self-review

**Spec coverage.** Registration is Task 2. Path-scoped work preserved is Task 1 Step 4, which keeps `ConvergePaths` rather than routing through `SyncIndex`. Cancel between paths is Task 1 Steps 6 through 8. The operator-facing claims are Task 3. The journal-off-lock section of the spec is deliberately excluded and named above.

**Placeholders.** None. Every step carries the code or the command it needs.

**Type consistency.** `ConvergeOutcome` is defined in Task 1 Step 3 with fields `PathsGiven` and `PathsConverged`, and Task 2 Step 4 reads `outcome.PathsConverged`. `registerConvergeJob` is defined in Task 2 Step 3 and called in Step 4 with the same signature.

**One risk the structure removes.** A converge could run under one context while `CancelJob` cancels another, which would leave cancel silently ineffective. `registerConvergeJob` returns the context it derived from the same cancel function it stored, so the caller cannot pick a different one. Task 2 Step 7 still names the symptom, because a future edit could reintroduce the split.

**One thing an implementer must check rather than assume.** Task 3 Step 3 depends on the offline harness running its file watcher. If `FileWatcherEnabled` is false there, no filesystem event fires and the test fails for a reason unrelated to the change. The step says what to do in that case.
