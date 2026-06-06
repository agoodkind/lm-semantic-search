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

Let an agent search the worktree it is in and get correct per-branch results,
while making the worktree's index cheap to build. Concretely: a worktree
becomes its own first-class codebase whose collection reflects its branch, but
the build reuses the sibling repo's already-embedded vectors so only the branch
diff plus uncommitted changes hit the embedder. This is the "linked clone with
a diff" model.

## Locked decisions

- **Index model**: each worktree is its own codebase with its own Milvus
  collection (`md5(worktree real path)`), its own registry record, and its own
  merkle snapshot. Per-branch correctness and the TS shared-collection
  invariant both hold (a non-symlinked worktree path hashes identically in TS
  `path.resolve` and Go).
- **Build**: linked-clone via content reuse. Preload `LoadReuseVectors`
  (`internal/semantic/reuse.go`) from already-indexed sibling worktree
  collections in the same repo group, filtered to the same embedding model and
  dimension. Reuse is keyed on content SHA-256 and embeddings are a pure
  function of content with no path salt (`reuse.go:20-28`), so unchanged files
  reuse vectors and only changed content is embedded. No new copy logic is
  required.
- **Trigger**: auto on first use. When `GetIndex` resolves a path that is a git
  worktree of an already-indexed repo and has no codebase of its own yet, the
  daemon auto-creates the worktree codebase and starts a reuse-seeded build in
  the background, mirroring `adoptUnregisteredCodebase`
  (`internal/daemon/manager_adopt.go`).
- **Locations**: both nested worktrees (under the repo root) and external or
  sibling worktrees (`git worktree add ../feature`, centralized worktree dirs)
  are supported. Nested requires a discovery boundary and a coverage override.
- **Source of truth**: `.git` is the sole SOT for worktree topology. No sidecar
  file is introduced. The daemon reads the same on-disk files git reads. The
  existing registry (`registry.json`) and per-codebase merkle remain the SOT for
  index metadata (collection name, embedding model, snapshot); these are
  orthogonal to topology and already exist.

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
- The reverse enumeration (common dir -> sibling roots) can list a stale entry
  if a worktree was moved on disk without `git worktree move` (its `gitdir`
  pointer is stale until `git worktree repair`). We tolerate this by checking
  that each sibling root still exists and has a registry entry before using it
  as a reuse source, so a stale entry contributes nothing rather than breaking
  resolution.
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

### 2. Linked-clone build via content reuse

When building a worktree's collection (the Replace/index pass):

1. Detect the repo group and sibling worktree roots.
2. Intersect siblings with registry records at those paths whose effective
   embedding model and dimension match the worktree's effective config. Prefer
   the main worktree, then any other indexed sibling.
3. `LoadReuseVectors(ctx, [sibling collections])` to build the content-hash ->
   vector map.
4. Run the normal discovery + chunk pass over the worktree, passing the reuse
   map through the existing `reuse` parameter of
   `Reindex`/`StageReindex`/`Replace`. Unchanged files reuse vectors; only the
   branch diff and uncommitted edits are embedded.

When no eligible reuse source exists (main not indexed, or all siblings on a
different embedding model), the build falls back to a full embed. It stays
correct, it is just not cheap. We do not auto-index the main worktree to create
a reuse source (out of scope, YAGNI).

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
- **Auto-create**: a new `adoptWorktreeCodebase` path modeled on
  `adoptUnregisteredCodebase` mints a stable id, persists a registry record,
  selects reuse sources, enqueues one reuse-seeded build, starts the watcher,
  and returns. It dedupes against any in-flight job via the existing
  `dedupAgainstActiveJob` contract.
- **Discovery boundary** (`discovery.go`, `walkFiles` and `walkGitignore`): stop
  descending into a subdirectory that is a worktree pointer into *this same*
  repo's common dir. The rule is precise to same-repo worktree pointers, so
  submodules and unrelated nested repos keep today's behavior. This also keeps
  the `.git` pointer file of a nested worktree out of the parent index.

## Status and observability

`get_indexing_status` gains a `git worktree of <main path> (branch <x>)` line
when the queried path is a worktree, mirroring the existing
`symlink resolved to: <real path>` line in `internal/daemon/render.go`. The
reuse-seeded build logs how many vectors were reused versus embedded, mirroring
`semantic.reuse_vectors_loaded`.

## Use-case battery

Location / nesting:

1. External sibling worktree (`git worktree add ../feature`): detected, own
   collection, reuse from main; no nesting work triggered.
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

6. Main indexed, worktree first use: reuse from main; only diff embeds.
7. Main not indexed, worktree first use: no reuse source; full build (correct,
   not cheap); main is not auto-indexed.
8. Several siblings indexed: reuse merges across all same-model sibling
   collections; main preferred for logging.
9. Sibling indexed with a different embedding model or dimension: excluded from
   reuse so vectors stay valid; it contributes nothing.
10. Two worktrees on the same branch: the second reuses the first almost
    entirely.

Branch / content state:

11. Worktree branch equals main branch, clean tree: ~100% reuse, near-zero
    embed.
12. Worktree diverged by N files: N files embed, the rest reuse.
13. Worktree with uncommitted or untracked edits: dirty files embed, committed
    files reuse.
14. Branch checkout inside an already-indexed worktree: files change on disk,
    merkle diff drives incremental re-embed, reuse from siblings still applies
    to the changed set.
15. Detached HEAD worktree: treated as a group member; branch label is the
    commit sha.

Lifecycle:

16. `git worktree remove`: the directory vanishes; existing missing-path/repair
    logic handles it; no special GC is added (matches the no-orphan-GC rule).
17. Worktree removed then recreated at the same path: same real path -> same
    collection, re-adopted and reused.
18. Worktree moved with `git worktree move`: new real path -> new collection;
    reuse from siblings makes the rebuild cheap.
19. Main repo deleted while a worktree lives on: the worktree is self-contained
    as its own codebase; the reuse source simply disappears.

Resolution / search:

20. Search at the worktree root: served by the worktree collection.
21. Search at a nested subdir of the worktree: worktree collection scoped via
    `subtreePrefix`.
22. Status query inside a worktree: surfaces the worktree relationship line plus
    reuse stats.
23. Concurrent first-use from two agents in the same new worktree: dedupes on
    `dedupAgainstActiveJob`; one build.
24. `index_codebase(force=true)` on a worktree: full rebuild, still
    reuse-seeded.

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
- `internal/daemon/manager_adopt.go` and the build path: reuse-source selection
  from siblings, wired into the index pass via the existing `reuse` parameter.
- `internal/daemon/render.go`: the worktree status line.
- Tests alongside each package, plus a portability test confirming worktree
  collection names match the shared invariant.

## Out of scope

- Auto-indexing the main worktree to manufacture a reuse source.
- Orphan-collection garbage collection for removed worktrees.
- Submodule-aware indexing changes (behavior preserved, not extended).
- Any sidecar or parallel bookkeeping file for worktree topology.

## Verification

- `go test ./...`
- `make lint` (golangci-lint, gofumpt, gocyclo, deadcode, staticcheck-extra)
- `make build` (vet, govulncheck)
- Live smoke against the local Milvus and `lmd-serve` on port 5400 for the
  worktree build, reuse counts, and per-branch search correctness.
