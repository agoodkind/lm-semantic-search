# Worktree-aware indexing design

Status: approved for planning
Date: 2026-06-05

## Problem

A codebase's identity today is its resolved real path (`canonicalizePath` ->
`filepath.EvalSymlinks` in `internal/daemon/manager_paths.go`), and the Milvus
collection name is `hybrid_code_chunks_<md5(abs path)[:8]>`
(`internal/semantic/service.go:CollectionName`). A git worktree is a separate
working directory at a different real path, so the daemon treats it as a brand
new, unrelated codebase and indexes it from scratch.

There is no git or worktree awareness anywhere in the daemon. Two concrete
failures follow:

1. An agent working inside a worktree either searches nothing (worktree never
   indexed) or, when the worktree is nested under an already-indexed repo root
   (the Claude Code `EnterWorktree` convention places worktrees at
   `.claude/worktrees/<name>`), is served by the covering parent index via
   `findCodebasesByCoverage` (`manager_status.go:28`) and gets the parent
   branch's content, not its own branch.
2. The parent repo's discovery walk has no boundary at a nested worktree root,
   so worktree files get swept into the parent collection, duplicating content
   and mixing branches. `.claude/` is not in the discovery defaults
   (`internal/discovery/discovery.go`), and in a worktree `.git` is a *file*,
   not a directory, so the `.git/` directory ignore pattern does not match it.

## Goal

An agent searches a worktree and receives results for that worktree's branch.
Each worktree has its own codebase, collection, registry record, and merkle
snapshot. For each changed file, the build looks up chunk vectors by content
hash in the target collection, then in one recorded source collection for
chunks without a match. The build embeds chunks without a compatible match.

## Locked decisions

- **Index model**: each worktree is its own codebase with its own Milvus
  collection (`md5(worktree real path)`), its own registry record, and its own
  merkle snapshot. Per-branch correctness and the TS shared-collection
  invariant both hold (a non-symlinked worktree path hashes identically in TS
  `path.resolve` and Go).
- The build looks up chunk vectors for each changed file in the target
  collection, then in one recorded source collection for chunks without a
  match. Reuse requires equal content and a compatible embedding model and
  vector dimension. The build embeds unmatched chunks. The build reads no
  other sibling collection, preloads no collection, and does not read the
  content vector catalog for a codebase in a Git repository.
- A status or search read discovers an unregistered Git worktree when a sibling
  worktree is indexed or runs its first build. `discoverWorktree` registers the
  worktree in the discovered state and records a selected source.
  `scheduleDeferredBuild` starts the build after a short delay. The read itself
  starts no embedding work. Without an indexed sibling or a sibling running its
  first build, the read registers no codebase. An explicit index request builds
  that worktree with a full embed.
- **Locations**: both nested worktrees (under the repo root) and external or
  sibling worktrees (`git worktree add ../feature`, centralized worktree dirs)
  are supported. Nested requires a discovery boundary and a coverage override.
- `.git` is the only source for worktree topology. The registry stores index
  metadata, including `WorktreeSourcePath`, `WorktreeSourceID`, and the optional
  `WorktreeStartingCommit`. `WorktreeSourceID` records the codebase identifier
  registered at the source path and is empty when none exists. The daemon does
  not derive a starting commit from HEAD. The recorded source is a reuse choice
  and does not identify the checkout that created the worktree.

## Topology from `.git`

All worktree topology is derived live at resolve time from `.git`:

- Is this path a worktree: a linked worktree root has a `.git` *file* whose
  content is `gitdir: <common>/.git/worktrees/<name>`; the main worktree has a
  `.git` *directory*.
- Repo group id: the resolved `commondir` referenced from
  `<common>/.git/worktrees/<name>/commondir`.
- Sibling worktree roots: enumerate `<common>/.git/worktrees/*/gitdir`; each
  points back at a worktree's `.git` pointer, so the worktree root is its
  dirname.
- Current branch: the `HEAD` file in that worktree's gitdir.

Reading `.git` live is strictly more correct than a sidecar cache, which would
go stale the moment someone runs `git worktree add/remove/move` while the daemon
is up.

### Reliability notes

- The forward direction (the path being resolved -> its own `.git` -> common
  dir) starts from the path we hold and is always reliable, so "am I a worktree
  and what is my group" never depends on a possibly-stale reverse list.
- Enumeration from the common directory to sibling worktree roots can list a
  stale entry when a worktree moves on disk without `git worktree move`; the
  worktree's `gitdir` pointer remains stale until `git worktree repair`.
  `gitworktree.SiblingWorktreeRoots` omits roots with nonexistent directories.
  Source selection at discovery uses a sibling root only when a registered code
  index at that root has the same embedding provider, model, and dimension.
  A stale entry supplies no reuse source and does not cause resolution to fail.
- Detection parses files directly rather than shelling out to `git`, so it has
  no dependency on git being on PATH and adds no subprocess per resolve.
- A bare repo, a corrupt `.git` pointer, or a non-git directory all fail soft to
  "no worktree group" and behave exactly as today.

## Three mechanisms

### 1. Worktree detection (`internal/gitworktree`, new package)

A small, isolated, unit-tested package that, given an absolute path, returns:

- whether the path is inside a git worktree, and that worktree's root;
- the repo group id (resolved common dir);
- the checked-out branch or detached commit;
- the set of sibling worktree roots in the same group;
- a boundary predicate `IsNestedWorktreeRoot(dir, commonDir)` that reports
  whether `dir` is a worktree pointer into the given repo's common dir.

It parses the `.git` pointer file and the `worktrees/<name>/{commondir,gitdir,
HEAD}` files. It does not depend on the daemon and does not call git.

### 2. Build with reuse from one recorded source

Discovery selects one source with `selectDiscoveredWorktreeSourceLocked`.
Candidates are registered code indexes at other worktree roots in the same
repository with the same embedding provider, model, and dimension.
The daemon samples at most 200 relative paths evenly from the sorted union of
candidate snapshot paths. It hashes the corresponding files in the new worktree
with `merkle.ContentHash`, which uses the same SHA-256 hash as the snapshots,
and counts exact matches for each candidate. Candidates with more matches rank
first. Ties favor the same HEAD commit, then the main worktree, then the latest
completed run, newest first. A checkout HEAD does not identify the contents of
its saved index. Without usable snapshots, all match counts are zero. Without an
indexed sibling, the daemon selects a sibling running its first build.
Remaining ties favor the main worktree, then path order. The daemon skips
candidates that would create a source cycle and records no source without an
eligible candidate.

Selection reads the registry, Git pointer files, saved merkle snapshot files
of candidates through `loadLiveCheckpoint`, and sampled files in the new
worktree without querying the vector store. The daemon skips file hashing
when fewer than two candidates own a live collection.
`gitworktree.HeadCommit` reads HEAD, loose refs, and `packed-refs` from disk.

An index request can declare a source with `source_path` and `starting_commit`.
The command `lm-semantic-search codebase index` accepts `--source-path` and
`--starting-commit`. The MCP `index_codebase` tool accepts `sourcePath` and
`startingCommit`. A later request without source fields preserves the stored
source. A new valid source replaces the stored source and clears the stored
starting commit unless the request supplies a commit.

The daemon selects the recorded source once per job. A usable source must have
a registered codebase, a live collection, and the same embedding provider,
model, and dimension as the target.

A deferred build after discovery, an automatic sync, or a resumed build waits
while the recorded source runs its first build. The waiting build starts after
the source build completes, fails, or is canceled. An operator's index request
does not wait; an unrelated sibling build does not delay the target build.

1. The build processes the chunks of each changed file.
2. The build looks up vectors by content hash in the target collection.
3. The build looks up unmatched chunks in the recorded source collection when
   the source is usable.
4. The build embeds chunks without a compatible match.

The build reads no other sibling collection and preloads no collection. An
unavailable source limits lookup to the target collection. Status reports
`worktree_reuse_unavailable_reason` as `no_source_recorded`,
`source_not_registered`, `source_not_indexed`, or
`source_embedding_configuration_differs`. A source collection read error other
than cancellation stops source reads without failing the build. The build
continues with the target collection and embeds unmatched chunks.

### 3. Worktree-bounded resolution and discovery

- **Resolution** (`GetIndex` in `manager_status.go`, plus a new
  `manager_worktree.go`): a worktree root is a hard codebase boundary. After
  canonicalizing, if the requested path lives inside a git worktree whose root
  differs from the covering codebase that `findCodebasesByCoverage` would
  return, resolve to (or auto-create) the worktree's own codebase rather than
  serving the covering parent. Once the worktree codebase exists, longest-prefix
  ordering already prefers it; the boundary check is what handles the very first
  resolve before the worktree codebase exists. The check runs in both resolve
  branches: the covered branch (a nested worktree under an indexed parent) and
  the no-coverage branch (an external worktree whose path no registry codebase
  covers), so an external worktree of an indexed repo auto-creates instead of
  returning out-of-scope.
- `discoverWorktree` registers an unregistered worktree in the discovered state
  and records a selected source when a sibling is indexed or runs its first
  build. `scheduleDeferredBuild` starts the build after a short delay. A first
  build waits for the recorded source's first build to complete, fail, or be
  canceled. The status or search read itself starts no embedding work.
- **Discovery boundary** (`discovery.go`, `walkFiles` and `walkGitignore`): stop
  descending into a subdirectory that is a worktree pointer into *this same*
  repo's common dir. The rule is precise to same-repo worktree pointers, so
  submodules and unrelated nested repos keep today's behavior. This also keeps
  the `.git` pointer file of a nested worktree out of the parent index.

## Status and observability

`get_indexing_status` shows `git worktree of <main path>`, the recorded
`WorktreeSourcePath`, `WorktreeSourceID`, optional `WorktreeStartingCommit`, and
`worktree_reuse_unavailable_reason`. Job progress reports chunks reused and
chunks embedded.

## Use-case battery

Location / nesting:

1. An external sibling worktree (`git worktree add ../feature`) has its own
   codebase and collection after discovery or an explicit index request.
   Discovery records one selected source.
2. Nested worktree under repo root (`.claude/worktrees/foo`): discovery boundary
   keeps it out of the parent index; resolution prefers the worktree over the
   covering parent.
3. Centralized worktree dir (`~/worktrees/<repo>/<branch>`): same as external
   sibling.
4. Worktree nested under a different already-indexed codebase that is not its
   repo: the boundary keys on the worktree pointer, so it still resolves to
   itself.
5. Symlinked worktree root: `canonicalizePath` resolves first; detection runs on
   the real path; that collection is not TS-shared (documented symlink caveat).

Reuse source availability:

6. On first use with an indexed main worktree, discovery selects one eligible
   source. An indexed sibling with more sampled content matches takes priority
   over the indexed main worktree.
7. When the main worktree is not indexed, discovery can select another indexed
   sibling or a sibling running its first build. Without either candidate, a
   read registers no codebase. An explicit index request performs a full embed.
   The daemon does not index the main worktree to create a source.
8. With several indexed siblings, discovery selects one source by sampled
   content match count first and applies the tie order above. The build reads
   only the target collection and the recorded source collection.
9. A sibling with a different embedding provider, model, or dimension is
   ineligible for discovery selection. If the recorded source's configuration
   differs at job time, status reports `source_embedding_configuration_differs`
   and the build reads only the target collection.
10. A second worktree on the same branch reuses matching chunk vectors when the
    first worktree is its recorded, usable source.

Branch / content state:

11. When a recorded, usable source exists, a clean worktree on the same branch
    as the main worktree reuses matching chunk vectors from the target or
    recorded source collection. The build embeds chunks without a match.
12. When a recorded, usable source exists, a worktree that diverges by N files
    looks up vectors for chunks in those changed files. The build embeds only
    chunks without a compatible match.
13. When a recorded, usable source exists, uncommitted or untracked changes
    reuse matching chunk vectors from the target or recorded source collection.
    The build embeds chunks without a compatible match.
14. A branch checkout inside an indexed worktree changes files on disk. The
    merkle diff selects the changed files for vector lookup in the target
    collection and, when usable, the recorded source collection.
15. Detached HEAD worktree: treated as a group member; branch label is the
    commit sha.

Lifecycle:

16. `git worktree remove`: the directory vanishes; existing missing-path/repair
    logic handles it; no special GC is added (matches the no-orphan-GC rule).
17. Worktree removed then recreated at the same path: same real path -> same
    collection, re-adopted and reused.
18. After `git worktree move`, the new path is a new codebase. Discovery selects
    one source for the new path by the same source selection rules.
19. If the removed main codebase is the worktree's recorded source, later
    worktree builds read only the target collection. Status reports
    `source_not_registered`.

Resolution / search:

20. Search at the worktree root: served by the worktree collection.
21. Search at a nested subdir of the worktree: worktree collection scoped via
    `subtreePrefix`.
22. Status query inside a worktree: surfaces the worktree relationship line plus
    reuse stats.
23. Concurrent first-use from two agents in the same new worktree: dedupes on
    `dedupAgainstActiveJob`; one build.
24. `index_codebase(force=true)` rebuilds the worktree without reading the
    recorded source collection.

Migration / correctness:

25. Parent repo already indexed with worktree files swept in before this change:
    after the boundary lands, the next parent sync sees those files as removed
    and prunes them from the parent collection.
26. `.git` pointer file in a worktree (matched by neither the `.git/` directory
    pattern today): excluded by the boundary rule, so it is never indexed.
27. Non-git directory: detection is a no-op; behaves exactly as today.
28. Bare repo or corrupt `.git` pointer: detection fails soft to "no worktree
    group"; normal indexing.

Invariants held:

29. Each worktree collection stays TS-drop-in compatible (a non-symlinked path
    hashes identically in TS and Go).
30. No collection is ever dropped or renamed; worktree collections are adopted,
    never garbage collected.

## Components and files touched

- `internal/gitworktree/` (new): pointer parsing, repo group, sibling
  enumeration, boundary predicate; unit-tested in isolation.
- `internal/discovery/discovery.go`: nested same-repo-worktree boundary in the
  walk.
- `internal/daemon/manager_status.go` (`GetIndex`) and a new
  `internal/daemon/manager_worktree.go`: worktree-bounded resolution and
  auto-on-use creation, modeled on `adoptUnregisteredCodebase`.
- The build selects the recorded source once per job and looks up chunk vectors
  by content hash in the target collection before querying the recorded source
  collection for chunks without a match.
- `internal/daemon/render.go`: the worktree status line.
- Tests alongside each package, plus a portability test confirming worktree
  collection names match the shared invariant.

## Out of scope

- Auto-indexing the main worktree to manufacture a reuse source.
- Orphan-collection garbage collection for removed worktrees.
- Submodule-aware indexing changes (behavior preserved, not extended).
- The daemon uses no sidecar or parallel bookkeeping file for worktree topology.
  The registry stores the recorded reuse source as index metadata.

## Verification

- `go test ./...`
- `make lint` (golangci-lint, gofumpt, gocyclo, deadcode, staticcheck-extra)
- `make build` (vet, govulncheck)
- Live smoke against the local Milvus and `lmd-serve` on port 5400 for the
  worktree build, reuse counts, and per-branch search correctness.
