# Index registration: caller-relative paths and a single discovery walk

## Problem

The index command accepts a folder path and asks the daemon to index that folder.
When the caller passes a relative path such as `.`, the daemon resolves it against
its own working directory instead of the caller's. The daemon runs under launchd
with `/` as its working directory, so `codebase index .` registers the entire disk
as a codebase.

Registration also scans the requested tree for ignore rules before it replies.
On a large tree the command sits silent for the whole scan. Pressing Ctrl-C kills
the client, prints the command usage text, and leaves the daemon job running.
In the incident that motivated this design (daemon trace
`75038c3e73fda05937610ad8863eef8e`), the registration call ran for 18.6 seconds
and left a failed codebase record rooted at `/`.

Each index job also reads the tree more times than it needs to. The rule scan
(`EffectiveIgnorePatterns`, which reads every nested `.gitignore`), the file scan
(`Discover`, which lists every indexable file), the registration scan, and a
fallback scan in the file watcher each walk the same directories.

## Design

### Registration replies immediately

Registration does no tree scanning. It resolves the path, runs guards that each
cost one file stat, saves the codebase record with empty ignore rules, queues the
background job, and replies with the job id. Reply time does not depend on
repository size. The inode stability check, which stats the root twice to detect
filesystems with unstable file ids, stays in registration because it is constant
time.

Watch registration also leaves the reply path. The lifecycle hook that adds a new
codebase to the file watcher runs in a goroutine owned by the daemon's run
context, so even the watch setup call (`notify.Watch`) cannot delay the reply.
Adding a codebase to the watcher is idempotent per codebase id, which makes the
goroutine safe to fire on every registration.

### One walk produces both the file list and the ignore rules

The background job walks the tree once. When the walk enters a directory, it
reads that directory's `.gitignore`, adds the parsed rules to a rule tree, and
applies the rules it has collected so far, so the walk never descends into an
ignored directory such as `node_modules`. The walk returns the surviving file
list together with the finished rule tree. `Discover`, the function that performs
this walk, returns the rule tree to its caller instead of computing rules in a
separate pass. The job saves the rule tree onto the codebase record when the walk
completes.

`EffectiveIgnorePatterns` remains available for callers that need rules without a
file list. For any given tree it must produce the same rule tree as the
single-pass walk. The fixture tests in `discovery_test.go` lock that equivalence.

### No other component walks the tree

The file watcher reuses the rules saved on the codebase record. It never scans
the tree itself. During the seconds between registration and the first save of
the rule tree, the watcher may forward events for files that the rules would
ignore. The converge pass, the periodic cleanup that reconciles watcher events
with the index, already drops those events.

One lazy fallback remains. The status classification path in `manager_status.go`
resolves and caches rules when a status query arrives for a codebase whose record
holds none. That fallback runs outside the registry lock and outside the
registration call, so it cannot delay registration or block other requests.

The rule cache lives in memory only (`ResolvedIgnoreRules` is excluded from the
registry JSON), so after a daemon restart every record holds no rules until the
next capture persists them. The periodic interval sync recaptures each codebase
within minutes of boot, which repopulates the cache without any boot-time scan.

### Relative paths resolve against the caller

Every request that carries a path also carries the caller's working directory, in
the `caller_cwd` field of the `ClientInfo` message. The CLI fills the field from
`os.Getwd()` on every path-taking command: `index`, `sync`, `status`, `search`,
and `clear`. The MCP adapter, the process that serves editor clients, does the
same. `GetIndexRequest` and `SearchCodeRequest` carry a `ClientInfo` field so
that every path-carrying request has one.

The daemon joins a relative request path onto `caller_cwd` before it
canonicalizes the result. Codebase-id arguments, which start with `cb_` and
contain no path separator, skip the join. Paths that contain `://` are rejected
as URIs in `canonicalizePath`.

### Refusals

The daemon rejects a relative path when `caller_cwd` is empty, with an
InvalidArgument error. The daemon's own working directory never matches the
caller's, so resolving against it silently is never correct.

The daemon refuses to register the filesystem root `/`. This guard sits next to
the existing guards in `manager_guards.go`, which reject the daemon's own state
directory and any path that is not a directory.

### The index command returns immediately, and `--wait` attaches with a timeout

`codebase index` and `codebase sync` print the job id as soon as the daemon
accepts the job and then return. This is the default in every output mode, so
scripts and machine consumers always get the return-immediately contract.

A `--wait <duration>` flag attaches to the job after the job id prints and
renders live progress from the `WatchJobs` stream, the daemon's existing
job-event subscription. Progress lines show the phase, the percent complete, and
the file counts the job already reports. Bare `--wait` uses 300 seconds, the same
default as the MCP tool's `wait_timeout_seconds`. There is no indefinite wait.
`--wait` requires human output mode; combining it with `--json` or
`--output single-line` is an error, so machine output never interleaves with
progress rendering.

While attached, the command exits 0 when the job completes and exits non-zero
with the job's error message when the job fails or is cancelled. When the timeout
expires first, the command detaches, prints a line saying the job was sent to the
background along with the job id and the `job get <id>` command to keep checking,
and exits 0, because a timeout says nothing about whether the job will succeed.

### Ctrl-C prints one line

The CLI suppresses the usage text once arguments have parsed, so a failed call or
an interrupt prints a single error line. Ctrl-C while attached under `--wait`
detaches from the stream without cancelling the job and prints the same
sent-to-background line as a timeout: the job keeps running in the daemon, and
`job get <id>` checks on it.

## Testing

Unit tests cover each behavior: a relative path joins against `caller_cwd`; a
relative path with an empty `caller_cwd` is rejected; `cb_*` ids skip the join;
registration of `/` is refused; the single-pass walk returns the same rule tree
as `EffectiveIgnorePatterns` for the fixtures in `discovery_test.go`; the watcher
never calls `EffectiveIgnorePatterns`; the default path returns without attaching
to the job stream; `--wait` detaches with the sent-to-background message when the
timeout expires before the job finishes.

The standard gates run before completion: `go test ./...`, `make lint`, and
`make build`.

A live smoke test confirms the user-facing behavior: `codebase index .` from a
real repository prints a job id immediately and returns; `codebase index . --wait`
renders progress and exits when the job finishes; a short `--wait 1s` on a large
repository prints the sent-to-background line; the command against a stopped
daemon prints a clear error; Ctrl-C while attached prints one line and the job
finishes in the daemon.

## Out of scope

- Any change to collection naming, chunk ids, or the TypeScript-compatibility
  surface.
