# Converge job registration design

Status: proposed
Date: 2026-07-29

Scope: making file-change indexing addressable by the job commands. The three surfaces `lm-semantic-search job list`, `job get`, and `job cancel` reach every other kind of indexing work; this makes them reach this kind too.

## The gap

Saving a file makes the daemon update the search index for that file. That work carries no job record, so the job commands cannot see it, resolve it, or stop it.

Two loops in `internal/daemon/background_sync.go` drive indexing and they diverge on this one point.

The periodic loop calls `manager.SyncIndex`, which creates a job. Those are the `sync` jobs the job list reports, 17,212 of them completed in the current ledger.

The watcher loop calls `manager.ConvergePaths` directly, which creates nothing. The identifier `watch-<codebase-id>-<unix>` that appears in the logs beside this work is built as a `correlation.IdentityAttribute`, a logging label. It is not a job id, which is why `job get` on one returns NotFound and why no record in the job ledger carries one.

## What changes

A converge becomes a job before it does any work, and keeps doing exactly the work it does today.

The job carries `Operation: "converge"`. Its scope is the count of paths in the batch, not a file total for the codebase, because a converge reads only the paths it was handed.

The wire surfaces report a trigger token, and `pbconv.jobTrigger` derives it from the operation rather than from a stored field: any operation other than `index` reads as `changed_files`. A converge therefore reports as changed-files work without any change to that function, which is accurate, since a converge is exactly a response to changed files.

Work stays path-scoped. `SyncIndex` diffs a whole codebase and `ConvergePaths` reads a named path list, so registration is taken from one and the work is left with the other. Routing the watcher through `SyncIndex` would turn one saved file into a full-tree diff.

### Registration is reused, not rebuilt

The job store, the journal, the cancel registry, and the three job commands all exist and already serve two creation paths. This adds a third caller, not a second mechanism.

Cancellation needs no new machinery either. `Manager.CancelJob` stops work by calling a `context.CancelFunc` held in `manager.cancels`, keyed by job id. `ConvergePaths` already takes a context. Registering the converge's cancel function under its job id is what connects the two.

## The journal moves off the manager lock

`appendJobLocked` opens a file, encodes one event, and closes it, all while holding `manager.mu`. Every job state change therefore serializes every other manager operation behind a disk write.

A single-writer goroutine takes the append instead, fed by an ordered channel. One writer preserves event order, and the journal stays the source of truth for job history.

This is included because the change adds a third caller to that path. It improves every job path rather than only converges.

### Cost

Measured under ambient load on a machine running ten worktrees: five paths upserted per ninety seconds, no contention events over the window, and 94 percent of touched chunks served from stored vectors rather than re-embedded.

A converge journals three times, at queued, running, and terminal. Progress events are throttled to one per ten seconds by `jobProgressJournalInterval`, so a long converge does not multiply appends.

## Cancel stops between paths

`ConvergePaths` walks its path list one path at a time. It checks for cancellation before each path, and on cancellation it stops and writes the merkle snapshot covering the paths it did converge. The job ends cancelled.

Paths that did not converge keep their previous index entries, which makes them drift. Drift is the case the periodic sync already exists to repair.

That repair is measurable. The periodic loop passes over each codebase every five minutes. Of 17,212 completed sync jobs in the ledger, 1,721 embedded at least one file, so roughly one pass in ten finds drift and fixes it while the rest confirm the index already matches disk. The worst cost of a cancelled converge is one such interval.

The snapshot stays consistent because a path is written to the snapshot only after it converges, so a path that never ran keeps its previous hash and reads as changed on the next pass.

## What an operator sees

`job list` reports a running converge alongside sync and index jobs, with its path count as scope.

`job get <id>` resolves it and reports its state, its path count, and its progress.

`job cancel <id>` stops it, and the surviving paths converge on the next sync pass at the latest.

`status` continues to report converges as it does now. Its rows gain a job id where they currently report none, which makes the two surfaces agree rather than one showing work the other denies.

## Testing

A test starts a converge, cancels it partway, and asserts the job reaches cancelled while the paths converged before the cancel are present in the snapshot and the rest are absent.

A test asserts a converge appears in `ListJobs` while running and resolves through `GetJob`, which is the pair of claims the ticket's done condition names.

A test asserts the journal append happens off `manager.mu`, by holding the manager lock and requiring a job transition to complete.

A live check writes a file into a watched codebase and asserts the same work appears in both `job list` and `status`, since the defect being closed is precisely those two surfaces disagreeing.

## The job journal grows without bound, and three quarters of it is already dead

The journal at `jobs.jsonl` stands at 223 megabytes holding 169,195 events for 43,877 jobs, written over 72.6 days. That is 605 jobs and about 3 megabytes per day. Nothing deletes from it and nothing compacts it.

Every daemon boot reads all of it. `Manager.load` calls `store.ReadJobEvents`, which scans the file line by line into a map keyed by job id, keeping only the last event for each. Startup cost rises with the file forever.

Registering converges makes that file grow faster. The size of the increase is unmeasured, because a converge is one debounced batch rather than one file save, and nothing counts batches today.

The file averages 3.9 events per job, so 74 percent of it is superseded events that the replay discards. No single field explains the 1,383-byte average event: `progress` is the largest at 31 percent, and it is needed on the surviving record, so trimming fields is not the lever.

Four remedies apply, in this order.

**Compact.** Rewrite the file with one line per job, atomically. This is lossless by construction, because the replay already keeps only the last event per id, so the rebuilt map is identical. It takes the file from 223 megabytes to about 61 and needs no policy decision.

**Bound the boot read.** Read backwards from the end and stop once every non-terminal job is resolved. Terminal jobs matter only to history, so startup stops paying for them. This changes no bytes on disk.

**Rotate or drop old entries.** Move jobs outside a retention window to a rotated file the boot read ignores. This needs one decision: how far back `job list` should reach.

**Replace old jobs with counts.** The terminal tallies already exist in the job list summary, currently 29,099 completed, 61 failed, 14,337 superseded, and 365 canceled. Storing those in place of the job objects is the largest long-term saving and the only option that loses per-job history.

Compaction and the bounded read are pure wins. The other two trade history for size.

This work is tracked as LMS-1, separately from the converge registration, because it applies to every job path and its correctness rests on the replay semantics rather than on anything the converge does. The two changes can land in either order.

## Out of scope

Cancelling the periodic sync or an adoption build, both of which already cancel.
