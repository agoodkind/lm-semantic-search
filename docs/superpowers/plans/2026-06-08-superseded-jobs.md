# Superseded Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the misleading `waiting (retryable)` job-list bucket with a `superseded` concept: a failed job is superseded when a later terminal job exists for the same codebase, derived at read time from the immutable job ledger.

**Architecture:** Superseded is resolved in the status SOT (`internal/status`) and at the daemon boundary (`internal/daemon/status_present.go`), never in the render layer. The boundary builds a per-codebase successor chain over terminal jobs and feeds each job's successor id into `status.ResolveJob`. Render and the proto format the resolved `JobSurface`. The `render_guard_test.go` invariant stays intact.

**Tech Stack:** Go, `internal/status` SOT, `internal/daemon` gRPC boundary, protobuf via `buf generate`.

**Spec:** `docs/superpowers/specs/2026-06-08-superseded-jobs-design.md`

---

### Task 1: Per-codebase successor chain helper

**Files:**
- Modify: `internal/daemon/status_present.go`
- Modify: `internal/daemon/manager.go` (add one method, near the other job query methods such as `ListJobs`)
- Test: `internal/daemon/status_present_test.go`

- [ ] **Step 1: Write the failing test**

Add to `internal/daemon/status_present_test.go`:

```go
func TestBuildJobSuccessorsLinksEachTerminalJobToTheNext(t *testing.T) {
	t.Parallel()
	t0 := renderTestTime
	mk := func(id, cb string, state model.JobState, completedOffset time.Duration) model.Job {
		completed := t0.Add(completedOffset)
		return model.Job{ID: id, CodebaseID: cb, State: state, StartedAt: t0, CompletedAt: &completed}
	}
	running := model.Job{ID: "live", CodebaseID: "A", State: model.JobStateRunning, StartedAt: t0}
	jobs := []model.Job{
		mk("a1", "A", model.JobStateFailed, 1*time.Minute),
		mk("a2", "A", model.JobStateFailed, 2*time.Minute),
		mk("a3", "A", model.JobStateCompleted, 3*time.Minute),
		running, // active jobs are not part of the terminal chain
		mk("b1", "B", model.JobStateFailed, 1*time.Minute),
	}
	got := buildJobSuccessors(jobs)
	if got["a1"] != "a2" {
		t.Fatalf("a1 successor = %q, want a2", got["a1"])
	}
	if got["a2"] != "a3" {
		t.Fatalf("a2 successor = %q, want a3", got["a2"])
	}
	if _, ok := got["a3"]; ok {
		t.Fatalf("a3 is the latest terminal job and must have no successor, got %q", got["a3"])
	}
	if _, ok := got["live"]; ok {
		t.Fatalf("an active job must not appear in the successor chain")
	}
	if _, ok := got["b1"]; ok {
		t.Fatalf("b1 is the only job for its codebase and must have no successor")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/daemon/ -run TestBuildJobSuccessorsLinksEachTerminalJobToTheNext -v`
Expected: FAIL with `undefined: buildJobSuccessors`.

- [ ] **Step 3: Add the helper**

In `internal/daemon/status_present.go`, add `"sort"` and `"time"` to the import block (it already imports `strings`, `model`, `status`; `time` was added for `codebaseFailureView`). Append:

```go
// isTerminalJobState reports whether a job state is terminal (no further work).
// The successor chain links only terminal jobs, since an active job is not yet a
// recorded outcome in the ledger.
func isTerminalJobState(state model.JobState) bool {
	switch state {
	case model.JobStateCompleted, model.JobStateFailed, model.JobStateCancelled:
		return true
	default:
		return false
	}
}

// jobOrderTime returns the time a terminal job is ordered by: its completion
// time when set, else its start time.
func jobOrderTime(job model.Job) time.Time {
	if job.CompletedAt != nil {
		return *job.CompletedAt
	}
	return job.StartedAt
}

// buildJobSuccessors returns, for each terminal job id, the id of the immediate
// next terminal job for the same codebase, or no entry when it is the latest.
// Active jobs are excluded, so a failure whose only later attempt is still
// running has no successor until that attempt terminates. The chain is the basis
// for the superseded relationship: a failed job with a successor was overtaken.
func buildJobSuccessors(jobs []model.Job) map[string]string {
	byCodebase := make(map[string][]model.Job)
	for _, job := range jobs {
		if !isTerminalJobState(job.State) {
			continue
		}
		byCodebase[job.CodebaseID] = append(byCodebase[job.CodebaseID], job)
	}
	successors := make(map[string]string)
	for _, codebaseJobs := range byCodebase {
		sort.Slice(codebaseJobs, func(first int, second int) bool {
			timeFirst := jobOrderTime(codebaseJobs[first])
			timeSecond := jobOrderTime(codebaseJobs[second])
			if !timeFirst.Equal(timeSecond) {
				return timeFirst.Before(timeSecond)
			}
			return codebaseJobs[first].ID < codebaseJobs[second].ID
		})
		for index := 0; index+1 < len(codebaseJobs); index++ {
			successors[codebaseJobs[index].ID] = codebaseJobs[index+1].ID
		}
	}
	return successors
}
```

- [ ] **Step 4: Add the single-job lookup on the manager**

In `internal/daemon/manager.go`, directly after the `ListJobs` method, add:

```go
// JobSuccessorID returns the id of the immediate next terminal job for job's
// codebase, or empty when job is the latest terminal job. The single-job views
// use it since they do not hold the full job set the list view does.
func (manager *Manager) JobSuccessorID(job model.Job) string {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	codebaseJobs := make([]model.Job, 0)
	for _, candidate := range manager.jobs {
		if candidate.CodebaseID == job.CodebaseID {
			codebaseJobs = append(codebaseJobs, candidate)
		}
	}
	return buildJobSuccessors(codebaseJobs)[job.ID]
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `go test ./internal/daemon/ -run TestBuildJobSuccessorsLinksEachTerminalJobToTheNext -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/daemon/status_present.go internal/daemon/manager.go internal/daemon/status_present_test.go
git commit -m "Add per-codebase terminal job successor chain"
```

---

### Task 2: Resolve superseded in the SOT and thread it through every surface

This task changes the `JobSurface` shape and the `resolveJobSurface` signature, so the status package and all daemon callers must change together to keep the build green. Work top-down, compile at the end.

**Files:**
- Modify: `internal/status/status.go`
- Modify: `internal/status/status_test.go`
- Modify: `internal/daemon/status_present.go`
- Modify: `internal/daemon/render.go`
- Modify: `internal/daemon/grpc_server.go`
- Modify: `internal/daemon/render_test.go`
- Modify: `internal/daemon/banner_test.go`
- Modify: `internal/daemon/grpc_job_tokens_test.go`

- [ ] **Step 1: Update the status SOT model and resolver**

In `internal/status/status.go`:

1a. Add `"strings"` to the import block (currently only `internal/model`).

1b. Replace the `JobRetryableCountLabel` const with:

```go
// JobSupersededCountLabel is the summary-tally word for a failed job overtaken by
// a later terminal job, read from the one vocabulary instead of a renderer
// hard-coding the phrase.
const JobSupersededCountLabel = "superseded"
```

1c. Add `SupersededByJobID` to `JobInputs`:

```go
	// Dependency is the daemon's shared-dependency health mode.
	Dependency DependencyMode
	// SupersededByJobID is the id of the immediate next terminal job for this
	// job's codebase, or empty when this job is the latest. A failed job with a
	// successor is superseded.
	SupersededByJobID string
}
```

1d. Replace the `JobSurface` struct's `RetryableFailure` field with the superseded fields:

```go
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
```

1e. Replace the body of `ResolveJob` with:

```go
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
```

- [ ] **Step 2: Update the status tests**

In `internal/status/status_test.go`, replace `TestResolveJob` with:

```go
func TestResolveJob(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name           string
		in             JobInputs
		wantLabel      string
		wantError      string
		wantSuperseded bool
	}{
		{
			"running healthy",
			JobInputs{State: model.JobStateRunning},
			"running", "", false,
		},
		{
			"retryable failure, healthy, shows error",
			JobInputs{State: model.JobStateFailed, Retryable: true, ErrorMessage: "embedding endpoint is unreachable"},
			"failed, retryable", "embedding endpoint is unreachable", false,
		},
		{
			"retryable failure, degraded, suppresses echo",
			JobInputs{State: model.JobStateFailed, Retryable: true, ErrorMessage: "embedding endpoint is unreachable", Dependency: EmbedderUnreachable},
			"failed, retryable", "", false,
		},
		{
			"superseded retryable failure",
			JobInputs{State: model.JobStateFailed, Retryable: true, ErrorMessage: "boom", SupersededByJobID: "job_B"},
			"failed, retryable, superseded by job_B", "boom", true,
		},
		{
			"superseded hard failure",
			JobInputs{State: model.JobStateFailed, ErrorMessage: "internal error", SupersededByJobID: "job_B"},
			"failed, superseded by job_B", "internal error", true,
		},
		{
			"completed job with a successor is not superseded",
			JobInputs{State: model.JobStateCompleted, SupersededByJobID: "job_B"},
			"completed", "", false,
		},
	}
	for _, testCase := range cases {
		got := ResolveJob(testCase.in)
		if got.StateLabel != testCase.wantLabel {
			t.Errorf("%s: StateLabel = %q, want %q", testCase.name, got.StateLabel, testCase.wantLabel)
		}
		if got.ErrorLine != testCase.wantError {
			t.Errorf("%s: ErrorLine = %q, want %q", testCase.name, got.ErrorLine, testCase.wantError)
		}
		if got.Superseded != testCase.wantSuperseded {
			t.Errorf("%s: Superseded = %v, want %v", testCase.name, got.Superseded, testCase.wantSuperseded)
		}
	}
}
```

- [ ] **Step 3: Run the status tests**

Run: `go test ./internal/status/`
Expected: PASS.

- [ ] **Step 4: Thread the successor id through `resolveJobSurface`**

In `internal/daemon/status_present.go`, change `resolveJobSurface` to accept the successor id and pass it into `JobInputs`:

```go
func resolveJobSurface(job model.Job, pipelineDegraded bool, supersededByJobID string) status.JobSurface {
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
	return status.ResolveJob(status.JobInputs{
		State:             job.State,
		Retryable:         retryable,
		ErrorMessage:      errorMessage,
		Dependency:        dependency,
		SupersededByJobID: supersededByJobID,
	})
}
```

- [ ] **Step 5: Update the render layer**

In `internal/daemon/render.go`:

5a. Change `renderGetJob` to add the successor parameter and pass it down:

```go
func renderGetJob(job *model.Job, pipelineDegraded bool, supersededByJobID string) string {
	if job == nil {
		return "Job not found."
	}
	surface := resolveJobSurface(*job, pipelineDegraded, supersededByJobID)
	lines := []string{
		"🧾 Job " + job.ID,
		"📁 Codebase: " + job.CanonicalPath,
		"⚙️ Operation: " + job.Operation,
		"🚦 State: " + surface.StateLabel,
		"🔧 Phase: " + displayJobPhase(job.Progress.Phase),
		"📊 Progress: " + jobProgressDisplay(*job),
	}
	lines = append(lines, renderJobTimingLines(*job)...)
	if magnitude := renderReconcileMagnitude(job.Progress); magnitude != "" {
		lines = append(lines, magnitude)
	}
	if surface.ErrorLine != "" {
		lines = append(lines, "🧯 Error: "+surface.ErrorLine)
	}
	return strings.Join(lines, "\n")
}
```

5b. Change `renderListJobs` to build the successor map, count superseded, and use the superseded label. Replace the counting loop and the `Terminal:` line:

```go
	successors := buildJobSuccessors(jobs)
	stateCounts := map[model.JobState]int{}
	supersededCount := 0
	for _, job := range jobs {
		stateCounts[job.State]++
		if resolveJobSurface(job, pipelineDegraded, successors[job.ID]).Superseded {
			supersededCount++
		}
		switch job.State {
		case model.JobStateQueued, model.JobStateRunning, model.JobStateCancelling:
			activeJobs = append(activeJobs, job)
		case model.JobStateCompleted, model.JobStateFailed, model.JobStateCancelled:
			terminalJobs = append(terminalJobs, job)
		default:
			terminalJobs = append(terminalJobs, job)
		}
	}

	lines := make([]string, 0, 32)
	lines = append(lines, fmt.Sprintf("Tracked jobs: %d total", len(jobs)))
	lines = append(lines, fmt.Sprintf(
		"Active: %d queued, %d running, %d canceling",
		stateCounts[model.JobStateQueued],
		stateCounts[model.JobStateRunning],
		stateCounts[model.JobStateCancelling],
	))
	// A superseded failure was overtaken by a later terminal job, so it is tallied
	// apart from current failures: the headline "failed" count names only the
	// latest unresolved failure per codebase.
	lines = append(lines, fmt.Sprintf(
		"Terminal: %d completed, %d failed, %d %s, %d canceled",
		stateCounts[model.JobStateCompleted],
		stateCounts[model.JobStateFailed]-supersededCount,
		supersededCount,
		status.JobSupersededCountLabel,
		stateCounts[model.JobStateCancelled],
	))
```

Then update the three `renderJobListEntry(job, pipelineDegraded)` calls inside `renderListJobs` to `renderJobListEntry(job, pipelineDegraded, successors[job.ID])`.

5c. Change `renderJobListEntry` to add the successor parameter:

```go
func renderJobListEntry(job model.Job, pipelineDegraded bool, supersededByJobID string) []string {
	surface := resolveJobSurface(job, pipelineDegraded, supersededByJobID)
	lines := []string{
		fmt.Sprintf(
			"- %s [%s · %s] %s %s",
			job.ID,
			surface.StateLabel,
			jobProgressDisplay(job),
			job.Operation,
			job.CanonicalPath,
		),
	}
	lines = append(lines, renderJobTimingLines(job)...)
	if magnitude := renderReconcileMagnitude(job.Progress); magnitude != "" {
		for line := range strings.SplitSeq(magnitude, "\n") {
			lines = append(lines, "  "+line)
		}
	}
	if surface.ErrorLine != "" {
		lines = append(lines, "  Error: "+surface.ErrorLine)
	}
	return lines
}
```

- [ ] **Step 6: Update the gRPC boundary**

In `internal/daemon/grpc_server.go`:

6a. Change `applyJobDisplayTokens` to add the successor parameter and pass it to `resolveJobSurface`. Do not set proto superseded fields yet; those arrive in Task 3:

```go
func applyJobDisplayTokens(pbJob *pb.Job, job model.Job, pipelineDegraded bool, supersededByJobID string) {
	if pbJob == nil {
		return
	}
	surface := resolveJobSurface(job, pipelineDegraded, supersededByJobID)
	pbJob.DisplayState = surface.StateLabel
	pbJob.DisplayError = surface.ErrorLine
}
```

6b. Change `toJobWithTokens` and `toJobPointerWithTokens` to add the successor parameter:

```go
func toJobWithTokens(job model.Job, pipelineDegraded bool, supersededByJobID string) *pb.Job {
	pbJob := pbconv.ToJob(job)
	applyJobDisplayTokens(pbJob, job, pipelineDegraded, supersededByJobID)
	return pbJob
}

func toJobPointerWithTokens(job *model.Job, pipelineDegraded bool, supersededByJobID string) *pb.Job {
	if job == nil {
		return nil
	}
	return toJobWithTokens(*job, pipelineDegraded, supersededByJobID)
}
```

6c. In the `GetJob` handler, resolve the successor and pass it to both the proto and the render:

```go
	health := server.manager.DependencyHealth()
	successorID := server.manager.JobSuccessorID(job)
	return &pb.GetJobResponse{
		Job:              toJobWithTokens(job, health.Degraded(), successorID),
		DependencyHealth: toDependencyHealth(health),
		DisplayText:      server.envelopeText(ctx, health, renderGetJob(&job, health.Degraded(), successorID), "job_id", job.ID, "codebase_id", job.CodebaseID),
	}, nil
```

6d. In the `ListJobs` handler, build the successor map once and pass per job:

```go
	jobs := server.manager.ListJobs(request.GetCodebaseId())
	health := server.manager.DependencyHealth()
	successors := buildJobSuccessors(jobs)
	response := &pb.ListJobsResponse{
		Jobs: make([]*pb.Job, 0, len(jobs)),
	}
	for _, job := range jobs {
		response.Jobs = append(response.Jobs, toJobWithTokens(job, health.Degraded(), successors[job.ID]))
	}
	response.DependencyHealth = toDependencyHealth(health)
	response.DisplayText = server.envelopeText(ctx, health, renderListJobs(jobs, health.Degraded()), "codebase_id", request.GetCodebaseId())
```

6e. In the `WatchJobs` handler, resolve per streamed job:

```go
	degraded := server.manager.DependencyHealth().Degraded()
	for _, jobID := range request.GetJobIds() {
		job, found := server.manager.GetJob(jobID)
		if !found {
			continue
		}
		if sendErr := stream.Send(&pb.WatchJobsResponse{Job: toJobWithTokens(job, degraded, server.manager.JobSuccessorID(job))}); sendErr != nil {
```

6f. For the `GetIndex` active job and the `SearchCode` active job, pass `""`, because an active job is never superseded:

In `GetIndex`: `response.ActiveJob = toJobPointerWithTokens(activeJob, health.Degraded(), "")`
In `SearchCode`: `ActiveJob: toJobPointerWithTokens(outcome.ActiveJob, health.Degraded(), ""),`

- [ ] **Step 7: Update the daemon tests for the new signatures**

7a. In `internal/daemon/banner_test.go`, `TestRenderGetJobNoEchoWhenDegraded`: change `renderGetJob(job, true)` to `renderGetJob(job, true, "")`, change `renderGetJob(job, false)` to `renderGetJob(job, false, "")`, and change the expectation `"State: failed (retryable)"` to `"State: failed, retryable"`.

7b. In `internal/daemon/render_test.go`, add `, ""` as the third argument to every `renderGetJob(...)` call. The compiler will list each site (`TestRenderGetJobShowsMagnitude`, `TestRenderGetJobUsesAmericanCanceledSpelling`, `TestRenderGetJobPreparingNotZeroPercent`, `TestRenderGetJobSyncPreparingWording`, `TestRenderGetJobKeepsRealZeroPercent`, and any others).

7c. In `internal/daemon/render_test.go`, `TestRenderListJobsSummarizesHistory`: change the expected `"Terminal: 1 completed, 0 failed, 0 waiting (retryable), 1 canceled"` to `"Terminal: 1 completed, 0 failed, 0 superseded, 1 canceled"`.

7d. In `internal/daemon/render_test.go`, replace `TestRenderListJobsSeparatesRetryableFailures` with a superseded test (two failed jobs for one codebase: the earlier is superseded by the later):

```go
// An earlier failure for a codebase that has a later terminal job is tallied as
// superseded, not failed, and the entry names the successor.
func TestRenderListJobsSeparatesSupersededFailures(t *testing.T) {
	t.Parallel()
	t0 := renderTestTime
	older := t0.Add(1 * time.Minute)
	newer := t0.Add(2 * time.Minute)
	jobs := []model.Job{
		{
			ID:            "job_old",
			CodebaseID:    "A",
			CanonicalPath: "/repo/a",
			Operation:     "sync",
			State:         model.JobStateFailed,
			StartedAt:     t0,
			CompletedAt:   &older,
			Error:         &model.JobError{Message: "embedding endpoint is unreachable", Retryable: true},
		},
		{
			ID:            "job_new",
			CodebaseID:    "A",
			CanonicalPath: "/repo/a",
			Operation:     "sync",
			State:         model.JobStateFailed,
			StartedAt:     t0,
			CompletedAt:   &newer,
			Error:         &model.JobError{Message: "internal error", Retryable: false},
		},
	}
	out := renderListJobs(jobs, false)
	if want := "Terminal: 0 completed, 1 failed, 1 superseded, 0 canceled"; !strings.Contains(out, want) {
		t.Fatalf("summary did not separate superseded from failed, want %q in:\n%s", want, out)
	}
	if want := "superseded by job_new"; !strings.Contains(out, want) {
		t.Fatalf("superseded entry did not name its successor, want %q in:\n%s", want, out)
	}
}
```

- [ ] **Step 8: Run the daemon and status tests**

Run: `go test ./internal/daemon/ ./internal/status/`
Expected: PASS, including `TestRenderLayerDoesNotLaunderJobStatus`, which stays green because the new logic lives in the SOT and boundary.

- [ ] **Step 9: Compile and commit**

If the compiler flags `toJobWithTokens(job, true)` in `grpc_job_tokens_test.go`, add `, ""` to that call so the tree compiles; Task 3 rewrites that test fully.

```bash
git add internal/status/status.go internal/status/status_test.go internal/daemon/status_present.go internal/daemon/render.go internal/daemon/grpc_server.go internal/daemon/render_test.go internal/daemon/banner_test.go internal/daemon/grpc_job_tokens_test.go
git commit -m "Resolve superseded jobs in the status SOT and route every job surface through it"
```

---

### Task 3: Emit superseded on the protobuf job

**Files:**
- Modify: `proto/lmsemanticsearch/v1/service.proto`
- Regenerate: `gen/go/lmsemanticsearch/v1/service.pb.go` (via `buf generate`)
- Modify: `internal/daemon/grpc_server.go` (`applyJobDisplayTokens`)
- Modify: `internal/daemon/grpc_job_tokens_test.go`

- [ ] **Step 1: Add the proto fields**

In `proto/lmsemanticsearch/v1/service.proto`, inside `message Job`, after the `string display_error = 17;` field, add:

```proto
  // superseded reports a failed job overtaken by a later terminal job for the
  // same codebase. The daemon resolves it at the boundary, so machine consumers
  // read the same fact the human surfaces do. state (field 7) stays raw.
  bool superseded = 18;
  // superseded_by_job_id is the id of the immediate next terminal job for this
  // job's codebase when superseded, else empty.
  string superseded_by_job_id = 19;
```

- [ ] **Step 2: Regenerate**

Run: `buf generate`
Expected: no output. `gen/go/lmsemanticsearch/v1/service.pb.go` now has `GetSuperseded()` and `GetSupersededByJobId()` on `*Job`.

Verify: `go doc ./gen/go/lmsemanticsearch/v1 Job` lists the two new fields.

- [ ] **Step 3: Set the proto fields at the boundary**

In `internal/daemon/grpc_server.go`, extend `applyJobDisplayTokens` to set the two fields:

```go
func applyJobDisplayTokens(pbJob *pb.Job, job model.Job, pipelineDegraded bool, supersededByJobID string) {
	if pbJob == nil {
		return
	}
	surface := resolveJobSurface(job, pipelineDegraded, supersededByJobID)
	pbJob.DisplayState = surface.StateLabel
	pbJob.DisplayError = surface.ErrorLine
	pbJob.Superseded = surface.Superseded
	pbJob.SupersededByJobId = surface.SupersededByJobID
}
```

- [ ] **Step 4: Write the test**

In `internal/daemon/grpc_job_tokens_test.go`, replace `TestToJobWithTokensEmitsResolvedDisplay` with:

```go
func TestToJobWithTokensEmitsResolvedDisplay(t *testing.T) {
	t.Parallel()
	job := model.Job{
		ID:    "job_x",
		State: model.JobStateFailed,
		Error: &model.JobError{Message: "embedding endpoint is unreachable", Retryable: true},
	}

	degraded := toJobWithTokens(job, true, "")
	if got := degraded.GetDisplayState(); got != "failed, retryable" {
		t.Fatalf("degraded display_state = %q, want %q", got, "failed, retryable")
	}
	if got := degraded.GetDisplayError(); got != "" {
		t.Fatalf("degraded display_error = %q, want empty (banner carries the cause)", got)
	}
	if got := degraded.GetState(); got != string(model.JobStateFailed) {
		t.Fatalf("raw state = %q, want %q kept for machine parsing", got, model.JobStateFailed)
	}

	superseded := toJobWithTokens(job, false, "job_next")
	if !superseded.GetSuperseded() {
		t.Fatalf("expected superseded=true when a successor is supplied")
	}
	if got := superseded.GetSupersededByJobId(); got != "job_next" {
		t.Fatalf("superseded_by_job_id = %q, want job_next", got)
	}
	if got := superseded.GetDisplayState(); got != "failed, retryable, superseded by job_next" {
		t.Fatalf("display_state = %q, want the superseded tag list", got)
	}
}
```

- [ ] **Step 5: Run the test**

Run: `go test ./internal/daemon/ -run TestToJobWithTokensEmitsResolvedDisplay -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add proto/lmsemanticsearch/v1/service.proto gen/go/lmsemanticsearch/v1/ internal/daemon/grpc_server.go internal/daemon/grpc_job_tokens_test.go
git commit -m "Emit superseded fields on the protobuf job"
```

---

### Task 4: Full verification

**Files:** none (verification only).

- [ ] **Step 1: Run the whole test suite**

Run: `go test ./...`
Expected: every package `ok`, zero `FAIL`. Confirm `TestRenderLayerDoesNotLaunderJobStatus` passed.

- [ ] **Step 2: Lint**

Run: `make lint`
Expected: `All checks passed.` If `exhaustruct` flags a `JobInputs` or `JobSurface` literal, set the missing field explicitly. If `staticcheck` flags a `!(A && B)`, apply De Morgan as in the existing `ResolveJob` error-line condition.

- [ ] **Step 3: Build**

Run: `make build`
Expected: `All checks passed.` plus `codesign ok`.

- [ ] **Step 4: Confirm the dead label is gone**

Run: `go doc ./internal/status JobRetryableCountLabel`
Expected: not found, because the const was renamed to `JobSupersededCountLabel`. If it still exists, remove the old declaration.

- [ ] **Step 5: Commit any verification fixes**

```bash
git add -A
git commit -m "Finalize superseded jobs: lint and build clean"
```

If steps 1-4 needed no changes, skip this commit.

---

## Self-Review

**Spec coverage:**
- Definition (failed plus a later terminal job means superseded; latest never superseded; successor is the immediate next terminal job; only failed jobs are chained). Task 1 (`buildJobSuccessors`) plus Task 2 Step 1e (`ResolveJob` gates on `State == Failed`).
- Counts (own bucket, reconciles). Task 2 Step 5b.
- Derived, not persisted. Task 1 (read-time map) plus Task 2 (no writes).
- Resolution at boundary, `JobSurface.Superseded`/`SupersededByJobID`. Task 2 Steps 1d, 4.
- Comma-joined tag label. Task 2 Step 1e.
- Presentation (list entry, single view). Task 2 Steps 5a, 5c.
- Proto fields 18/19. Task 3.
- Tests plus guard intact. Tasks 2, 3, 4.
- Replaces `waiting (retryable)`. Task 2 (label rename) plus Task 4 Step 4.

**Placeholder scan:** No TBD or TODO. Every code step shows complete code. The mechanical test-call-site updates (Task 2 Step 7b) name the exact files and the exact edit (add `, ""`).

**Type consistency:** `resolveJobSurface(job, bool, string)`, `renderGetJob(*Job, bool, string)`, `renderJobListEntry(Job, bool, string)`, `toJobWithTokens(Job, bool, string)`, `toJobPointerWithTokens(*Job, bool, string)`, and `applyJobDisplayTokens(*pb.Job, Job, bool, string)` all carry the same trailing `supersededByJobID string`. The Go field `JobSurface.SupersededByJobID` maps to proto `superseded_by_job_id`, which generates `SupersededByJobId` (used in Task 3 Step 3). `JobSupersededCountLabel` is defined in Task 2 Step 1b and used in Task 2 Step 5b.
