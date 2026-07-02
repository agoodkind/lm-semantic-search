# Task 5 Report: One reuse rule

## Implementation

Implemented `deltaState.itemReuseEnabled` and `Manager.resolveItemReusePolicy` in `internal/daemon/reuse_policy.go`.

The policy returns `semanticReady` for live delta writes. For staging bootstrap writes, it enables per-item live reuse only when `probeCollectionEvidence` reports `collectionPresencePresent` and the job is not forced.

`runDeltaSync` now sets `itemReuseEnabled` once for the run and loads build-wide reuse for added files without checking `codebase.Kind`.

`runBootstrap` now sets `itemReuseEnabled` once for the run, keeps the existing unconditional `resolveReuseSeed` call, and still writes to staging while item reuse reads from the source's live collection name.

`itemReuse` now checks `!state.itemReuseEnabled` instead of `state.staging`, so staging builds can reuse live vectors after the single run-level eligibility probe.

The fake semantic test double now records StageReindex reuse the same way Reindex does and supports `stageReindexWithReuse`, which makes bootstrap reuse assertions possible.

## Tests Added Or Changed

Added `TestRunBootstrapReusesLiveCollectionVectors`.

Added `TestRunBootstrapMissingLiveCollectionEmbedsEverything`, including the first conversation ingest bootstrap variant.

Added `TestRunBootstrapForcedSkipsLiveItemReuse`.

Added `TestResolveItemReusePolicy`.

Inverted the existing delta reuse subtest to `document-kind added delta seeds siblings like code`.

Adjusted two existing test fixtures without changing their expectations:

- `TestCodeItemReuseLoadsExactPath` now sets `itemReuseEnabled: true` because it calls `itemReuse` directly.
- `TestForceReindexUnchangedRepoBootstrapsWhenCollectionDisappears` now accounts for the new initial-bootstrap policy probe before the force job probe.

## Decisions

I followed the direct user instruction to use plain `git commit` and no `gt`, even though the brief still mentions `gt create`.

I kept `resolveReuseSeed` unchanged. The bootstrap path still calls it unconditionally.

I kept the item reuse load-failure fallback unchanged.

I treated the forced-job escape hatch as applying to staging bootstrap reuse, matching the requested `resolveItemReusePolicy` contract where `staging=false` returns `semanticReady`.

## Commands

```sh
sed -n '1,260p' .superpowers/sdd/task-5-brief.md
```

Output:

```text
### Task 5 (PR 5): One reuse rule

**Files:**
- Modify: `internal/daemon/manager_delta.go` (deltaState, runDeltaSync :288-313, runBootstrap :472-489, itemReuse :791-798)
- Modify: fakes in `internal/daemon/converge_concurrency_test.go` (StageReindex :254 currently discards the reuse param; record it, add `stageReindexWithReuse` hook)
- Test: `internal/daemon/manager_bootstrap_test.go`, `internal/daemon/manager_delta_reuse_test.go`

**Interfaces:**
- Produces: `deltaState.itemReuseEnabled bool` and `func (manager *Manager) resolveItemReusePolicy(ctx, job model.Job, staging bool, semanticReady bool) bool`. staging=false returns semanticReady. staging=true requires evidence Present (via probeCollectionEvidence) and !job.Forced.

- [ ] **Step 1: Failing tests**:
  - `TestRunBootstrapReusesLiveCollectionVectors`: live collection Present with vectors for the content; bootstrap embeds 0, reuses > 0, reuse reads target the LIVE collection name while writes go to staging; promote-before-merkle order asserted with the existing fake hooks.
  - `TestRunBootstrapMissingLiveCollectionEmbedsEverything` (the trap pin): InspectCollection {Exists:false} produces zero per-item reuse probes and every file embedded into staging, then promoted. Add a conversation variant: a genuinely first ingest bootstrap embeds all documents.
  - `TestRunBootstrapForcedSkipsLiveItemReuse`: job.Forced true, live Present, no per-item loads, everything embeds.
  - Invert the subtest "non-code added delta does not seed siblings" (manager_delta_reuse_test.go:166) to "document-kind added delta seeds siblings like code".
- [ ] **Step 2: Confirm failures.**
- [ ] **Step 3: Implement**: delete the Kind check at :308 (keep `len(plan.diff.Added) > 0 && state.semantic`); replace the staging early-out at :796-798 with `if !state.itemReuseEnabled { return state.reuse, 0 }`; set itemReuseEnabled from resolveItemReusePolicy in both runDeltaSync and runBootstrap; keep resolveReuseSeed in runBootstrap unchanged (descendant and sibling seeding; it is a natural no-op for chat:/// paths). The per-item load-failure fallback (:815-818) stays untouched so `TestConversationIngestReuseLoadFailureFallsBackToFullEmbed` (:1019) stays green.
- [ ] **Step 4: Gates green.** **Step 5: Commit** `Apply one reuse rule across staging and document builds`. `gt create`.
```

```sh
go test ./internal/daemon -run 'TestRunBootstrapReusesLiveCollectionVectors|TestRunBootstrapMissingLiveCollectionEmbedsEverything|TestRunBootstrapForcedSkipsLiveItemReuse|TestRunDeltaSyncSeedsSiblingReuseOnlyForAddedFiles'
```

Output, expected red before implementation:

```text
--- FAIL: TestRunBootstrapReusesLiveCollectionVectors (0.00s)
    manager_bootstrap_test.go:158: ChunksEmbedded = 2, want 0 when live vectors cover every bootstrap chunk
--- FAIL: TestRunDeltaSyncSeedsSiblingReuseOnlyForAddedFiles (0.01s)
    --- FAIL: TestRunDeltaSyncSeedsSiblingReuseOnlyForAddedFiles/document-kind_added_delta_seeds_siblings_like_code (0.00s)
        manager_delta_reuse_test.go:197: sibling collection cc_repo was not loaded; calls = []
FAIL
FAIL	goodkind.io/lm-semantic-search/internal/daemon	0.736s
FAIL
```

```sh
go test ./internal/daemon -run 'TestRunBootstrapReusesLiveCollectionVectors|TestRunBootstrapMissingLiveCollectionEmbedsEverything|TestRunBootstrapForcedSkipsLiveItemReuse|TestRunDeltaSyncSeedsSiblingReuseOnlyForAddedFiles|TestResolveItemReusePolicy'
```

Output:

```text
ok  	goodkind.io/lm-semantic-search/internal/daemon	0.739s
```

```sh
make build
```

Output:

```text
gksyntax-grammars: Swift parser already generated
🔎 logs=.make/logs trace_id=930ed26c33beab423a77ef9aed2d46bb span_id=4c15d96ed76960b8
go-mk build-check

  vet                ok
  lint-golangci      ok
  lint-format        ok
  lint-gocyclo       ok
  lint-deadcode      ok
  staticcheck-extra  ok
  govulncheck        ok
  cgo-stub           ok
  platform-stub      ok

  All checks passed.
dist/lm-semantic-search-daemon: replacing existing signature
dist/lm-semantic-search: replacing existing signature
dist/lm-semantic-search-mcp: replacing existing signature
codesign   ok
```

```sh
go test ./...
```

Output:

```text
ok  	goodkind.io/lm-semantic-search/client	(cached)
ok  	goodkind.io/lm-semantic-search/cmd/lm-semantic-search	(cached)
?   	goodkind.io/lm-semantic-search/cmd/lm-semantic-search-daemon	[no test files]
?   	goodkind.io/lm-semantic-search/cmd/lm-semantic-search-mcp	[no test files]
?   	goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1	[no test files]
ok  	goodkind.io/lm-semantic-search/internal/adapterr	(cached)
ok  	goodkind.io/lm-semantic-search/internal/archguard	0.397s
?   	goodkind.io/lm-semantic-search/internal/clock	[no test files]
ok  	goodkind.io/lm-semantic-search/internal/config	(cached)
ok  	goodkind.io/lm-semantic-search/internal/daemon	10.645s
ok  	goodkind.io/lm-semantic-search/internal/debugserver	(cached)
ok  	goodkind.io/lm-semantic-search/internal/discovery	(cached)
ok  	goodkind.io/lm-semantic-search/internal/embedding	(cached)
ok  	goodkind.io/lm-semantic-search/internal/gitworktree	(cached)
?   	goodkind.io/lm-semantic-search/internal/grpcutil	[no test files]
ok  	goodkind.io/lm-semantic-search/internal/indexability	0.849s
ok  	goodkind.io/lm-semantic-search/internal/indexer	(cached)
ok  	goodkind.io/lm-semantic-search/internal/mcpserver	(cached)
ok  	goodkind.io/lm-semantic-search/internal/merkle	(cached)
ok  	goodkind.io/lm-semantic-search/internal/metrics	(cached)
ok  	goodkind.io/lm-semantic-search/internal/migrate	(cached)
?   	goodkind.io/lm-semantic-search/internal/model	[no test files]
ok  	goodkind.io/lm-semantic-search/internal/pbconv	(cached)
ok  	goodkind.io/lm-semantic-search/internal/render	(cached)
ok  	goodkind.io/lm-semantic-search/internal/response	(cached)
ok  	goodkind.io/lm-semantic-search/internal/semantic	(cached)
?   	goodkind.io/lm-semantic-search/internal/spans	[no test files]
ok  	goodkind.io/lm-semantic-search/internal/status	(cached)
?   	goodkind.io/lm-semantic-search/internal/store	[no test files]
ok  	goodkind.io/lm-semantic-search/internal/tshash	(cached)
ok  	goodkind.io/lm-semantic-search/internal/view	(cached)
```

```sh
make lint
```

Output:

```text
gksyntax-grammars: Swift parser already generated
🔎 logs=.make/logs trace_id=8b8fb53c5497f761c537b48a7fad6b7a span_id=5ace9d67f051d9d3
lint

  lint-golangci      ok
  lint-format        ok
  lint-gocyclo       ok
  lint-deadcode      ok
  staticcheck-extra  ok

  All checks passed.
```

```sh
perl -ne 'print "$ARGV:$.:$_" if /[\x{2013}\x{2014}]/' internal/daemon/converge_concurrency_test.go internal/daemon/manager_bootstrap_test.go internal/daemon/manager_delta.go internal/daemon/manager_delta_reuse_test.go internal/daemon/manager_progress_test.go internal/daemon/manager_repair_test.go internal/daemon/reuse_policy.go internal/daemon/reuse_policy_test.go
```

Output:

```text
```

```sh
git diff --check
```

Output:

```text
```

## Self-review

The one reuse rule is decided once per run through `state.itemReuseEnabled`.

The bootstrap policy probes the live counterpart once before the per-item loop, so a missing live collection produces zero per-file reuse loader calls.

The bootstrap per-item reuse path reads the live collection named by `itemSource.reuseSource`; StageReindex still writes to staging.

Document-kind added deltas now get build-wide sibling reuse by deleting only the `CodebaseKindCode` gate.

No existing test expectation was changed except the specified inverted subtest. The other existing-test edits are fixture updates required by the new policy bit and the extra run-level probe.

The per-item load-failure fallback remains unchanged.

## Concerns

No known code concerns.

A read-only reviewer subagent was spawned after gates passed, but it did not finish before the closeout window and was closed. The self-review and required local gates completed.

## Fix round 1

`TestForceReindexUnchangedRepoBootstrapsWhenCollectionDisappears` now uses an explicit answer queue for the fake collection presence probes.

The first answer serves the initial `StartIndex` presence probe. It returns missing so the first request bootstraps a new index.

The second answer serves the bootstrap reuse-eligibility probe from `resolveItemReusePolicy`. The one-reuse-rule change added this run-level probe because a staging bootstrap only enables per-item reuse when the live counterpart collection still exists.

The third answer serves the forced `StartIndex` presence probe. It returns present so the force request chooses `streaming_reindex`, which preserves the test path that later discovers collection loss during the run.

The fourth answer serves the empty-diff probe from `planSyncDiff`. It returns missing so the streaming reindex falls back to bootstrap and embeds files after the collection disappears.

The fixture now checks that every queued answer is consumed and that no extra probe appears. It also checks the stack for each queued answer, so a same-count production probe shift must be re-derived deliberately instead of silently consuming the next boolean.
