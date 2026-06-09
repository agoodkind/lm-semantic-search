# Superseded jobs in the job ledger

## Problem

The job list is a ledger: it records every job ever run, reduced from an
append-only event log to each job's final state. The `Terminal:` summary line is
therefore a lifetime tally, not a snapshot of what is live.

A prior change split a `waiting (retryable)` bucket out of the failed count, to
keep transient infra stops from inflating `failed`. Inspection of live data
showed all 88 jobs in that bucket were two-day-old failures whose codebases had
since been re-indexed successfully: every one was superseded by a later run. The
label `waiting` was therefore false; nothing was waiting. It overlaid live-state
language onto resolved historical events, the same category error as the bug it
replaced.

Current health belongs on the codebase status, which the status source of truth
(SOT) already resolves correctly. The ledger's job is to record what happened.
The honest concept for an old failure that a later attempt overtook is
**superseded**, not waiting.

## Definition

For each codebase, its terminal jobs form a time-ordered chain. Each terminal
job links to the immediate next terminal job for the same codebase.

- A **failed** job is **superseded** when a later terminal job exists for the
  same codebase, regardless of that later job's outcome.
- The latest terminal job for a codebase is never superseded.
- `supersededBy` is the id of the immediate next terminal job. The links form a
  chain a reader can follow forward to the eventual success (or to the current
  unresolved failure).
- Only failed jobs are chained. Completed and canceled jobs are untouched.

A failed job that is the latest terminal job for its codebase is a current,
unresolved failure: it stays in the `failed` count and carries no superseded
marker.

## Counts

The summary keeps an own bucket for superseded, carved out of the failed total:

```
Terminal: 2728 completed, 26 failed, 88 superseded, 56 canceled
```

- `completed` = all completed jobs (lifetime).
- `failed` = failed jobs that are the latest terminal job for their codebase.
- `superseded` = failed jobs that have a later terminal job.
- `canceled` = all canceled jobs (lifetime).

The parts reconcile to the terminal total, because `superseded` is a subset of
all failed jobs: `completed + failed + superseded + canceled = terminal`.

A still-broken codebase with several failures and no later success counts each
earlier failure as superseded (it has a later attempt) and only its latest
failure as `failed`, so `failed` reads one current failure per broken codebase
rather than one per attempt.

## Resolution at the boundary

Superseded is derived, never persisted: the ledger stays immutable and no
historical record is rewritten. The daemon already holds every job and codebase
in memory, so at the read boundary it builds, once per request, the per-codebase
successor relationship over the terminal jobs:

- `successorJobID[jobID]` = the immediate next terminal job for that job's
  codebase, or empty when the job is the latest.

`resolveJobSurface` (the existing job-side mirror of `computeDisplayStatus` in
`internal/daemon/status_present.go`) takes that successor context and sets two
new resolved fields on `status.JobSurface`:

- `Superseded bool` = the job is failed and has a successor.
- `SupersededByJobID string` = the successor id, empty when not superseded.

The single-job view (`GetJob`) resolves the successor with one manager lookup
against `manager.jobs`, since it does not hold the full set.

The computation lives in the SOT and the boundary, never in the render layer, so
the existing `render_guard_test.go` invariant (render formats resolved views and
never re-derives job status from raw records) holds unchanged.

## Status label

`status.ResolveJob` builds `StateLabel` as a comma-joined tag list rather than a
parenthetical, so tags compose:

- base state word (`running`, `completed`, `failed`, `canceled`, ...)
- `retryable`, appended when the error is self-healing
- `superseded by <id>`, appended when the job is superseded

Examples: `running`, `completed`, `failed, retryable`,
`failed, superseded by job_B`, `failed, retryable, superseded by job_B`.

This replaces the prior `failed (retryable)` parenthetical form.

## Presentation

List entry keeps the full state label and the error line, and shows the chain
link inside the one bracket:

```
- job_A [failed, retryable, superseded by job_B · 0.0%] sync /Users/agoodkind/Sites/tack
  Started: 6/7/2026, 2:30 PM PDT
  Completed: 6/7/2026, 2:31 PM PDT
  Error: embedding endpoint is at capacity (rate limited)
```

Single-job view shows the same label on the state line:

```
🚦 State: failed, retryable, superseded by job_B
```

## Proto

`pb.Job` gains two fields so a machine consumer reads the same resolved fact the
human surfaces do, set at the boundary like the existing display tokens:

- `bool superseded = 18;`
- `string superseded_by_job_id = 19;`

Raw `state` (7) and `error` (13) stay for machine parsing.

## Testing

- `internal/status`: `ResolveJob` builds the tag list in order; sets
  `Superseded` and `SupersededByJobID` only when a successor is supplied; a
  failed job with no successor is not superseded.
- `internal/daemon`: the boundary builds the per-codebase successor chain
  correctly (oldest links forward, latest has none); the list summary tallies
  `superseded` apart from `failed` and the parts reconcile; the single-job view
  resolves the successor.
- Update `TestRenderGetJobNoEchoWhenDegraded` to expect `State: failed, retryable`.
- Replace the `waiting (retryable)` summary expectations from the prior change
  with `superseded` expectations.
- Proto round-trip: a superseded job emits `superseded=true` and the successor
  id.

## Out of scope

- Superseding completed or canceled jobs. Only failed jobs are chained.
- Persisting superseded onto the ledger. It is derived at read time.
- Changing the codebase status surfaces. Current health already resolves through
  the codebase SOT and is unaffected.
