# Index registration: caller-relative paths and a single discovery walk

## Problem

`lm-semantic-search codebase index .` resolves `.` inside the daemon process, whose
working directory is `/` under launchd. The daemon registers the filesystem root as a
codebase, scans the whole disk for `.gitignore` rules before replying, and the command
prints nothing until the scan ends. Ctrl-C kills the client, prints a usage dump, and
leaves the daemon job running. Reference incident: trace
`75038c3e73fda05937610ad8863eef8e`, where `StartIndex` ran 18.6 seconds and left a
failed codebase record rooted at `/`.

Independent of the bad path, one index job reads the tree multiple times: once to
collect ignore rules (`EffectiveIgnorePatterns`), once to list files (`Discover`),
once more at registration (`resolveIgnoreRulesOrLog`), and a fourth time in the
watcher when the registration result is empty.

## Design

### Registration replies immediately

`StartIndex` performs no tree walks. It canonicalizes the path, runs the existing
single-file stat guards plus the new refusals below, saves the codebase record with
empty ignore rules, queues the job, and replies with the job id. The reply time is
independent of repository size. The two-stat inode stability check stays inline
because it is O(1).

### One walk produces files and rules

The background job walks the tree exactly once. On entering each directory the walk
reads that directory's `.gitignore`, appends the parsed rules to the rule tree, and
applies the accumulated rules immediately, so the walk never descends into ignored
directories. The walk returns both the surviving file list and the finished rule
tree. `Discover` exposes the rule tree to its caller instead of recomputing it, and
the job persists the rule tree onto the codebase record when discovery completes.

`EffectiveIgnorePatterns` remains for callers that need rules without a file list
(the status classification path). Its output for a given tree must equal the rule
tree the single-pass walk produces; the existing `discovery_test.go` cases lock this.

### No other component walks

`Watcher.AddCodebase` uses the rules stored on the codebase record and never calls
`EffectiveIgnorePatterns`. Until the first job persists rules, the watcher may
enqueue events for ignored files; the converge pass already filters them, and the
window closes when discovery finishes.

The one remaining lazy resolver is the status classification path
(`manager_status.go`), which resolves and caches rules when a status query arrives
for a codebase whose record holds none. It runs outside the registry lock and
outside the registration RPC, so it cannot reintroduce the hang.

### Relative paths resolve against the caller

`ClientInfo` carries `caller_cwd`, the absolute working directory of the calling
process. The CLI fills it from `os.Getwd()` on every path-taking command (`index`,
`sync`, `status`, `search`, `clear`); the MCP adapter does the same. `GetIndexRequest`
and `SearchCodeRequest` gain a `ClientInfo` field so every path-carrying request has
one.

The daemon joins a relative request path onto `caller_cwd` before canonicalization.
Codebase-id arguments (`cb_*` with no path separator) bypass joining. The URI
rejection in `canonicalizePath` (paths containing `://`) stays.

### Refusals

- A relative path with an empty `caller_cwd` is rejected with InvalidArgument: the
  daemon's own working directory is never the caller's, so silent resolution against
  it is never correct.
- Registering the filesystem root `/` is rejected, alongside the existing state-root
  and non-directory guards in `manager_guards.go`.

### Ctrl-C prints one line

The CLI sets `SilenceUsage` once arguments have parsed, so a runtime RPC failure or
Ctrl-C prints a single error line instead of the usage block. The interrupt message
states that an already-accepted job continues in the daemon and names
`codebase list` as the way to check.

## Testing

- Unit: relative path joins against `caller_cwd`; relative path with empty
  `caller_cwd` is rejected; `cb_*` ids bypass joining; `/` registration is refused;
  the single-pass walk returns the same rule tree as `EffectiveIgnorePatterns` for
  the fixtures in `discovery_test.go`; the watcher does not call
  `EffectiveIgnorePatterns`.
- Gates: `go test ./...`, `make lint`, `make build`.
- Live smoke: `codebase index .` from a real repository prints a job id immediately;
  `codebase index .` against a stopped daemon and from a directory with no
  `caller_cwd` analog (empty env) produces clear errors; Ctrl-C during a slow call
  prints one line.

## Out of scope

- Progress streaming or watch-by-default for `codebase index`; the fire-and-forget
  contract stays.
- Async watcher registration (`notify.Watch` in a goroutine); FSEvents registration
  is cheap.
- Any change to collection naming, chunk ids, or the TS-compatibility surface.
