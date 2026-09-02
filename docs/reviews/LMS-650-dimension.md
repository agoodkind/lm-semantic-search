# LMS-650 reuse dimension adversarial review

The dimension-zero fix-forward is MERGE-READY. It resolves the source vector
dimension before transport packing, and the generation guard closes the stale
cache race found during review.

## Review record

| Field | Value |
| --- | --- |
| Date | 2026-09-02 06:44:42 PDT |
| PR 262 cleanup date | 2026-09-02 07:02:04 PDT |
| Branch | `codex/lms-650-resolve-reuse-dimension` |
| Live base | `ac2b5814fc92fe66885aedc8d92697f8879effbd` |
| Initial fix | `c8f978b640005c2d9eaee8b9366aa7e432ba8708` |
| Race fix | `b474aacf94ea536bb912c29ecce67e1e553df98f` |
| PR 262 cleanup | `3c619e3d8eb71162760da457bf6dec2e0c2a7670` |
| Class | Source schema validation, transport budgeting, and lifecycle cache correctness |
| Reviewer tier | Strongest-model whole-branch adversarial review |
| Verdict | MERGE-READY |
| Catches | B/SF/N: 1/0/0, addressed |
| Escapes | None |

## Deployed failure

Job `job_1788351066_2a19dc320787` failed on
`codex-rs/models-manager/models.json` with:

```text
reuse vector dimension must be positive: 0
```

The merged base passed configured dimension zero directly into response
packing before the selected Get. The live source collection schema carries a
positive 4,096-float dense vector dimension.

## Fix contract

For a nonempty selected candidate set, the fixed path describes that source
collection, requires a dense `FloatVector` field with a positive dimension,
checks any positive configured dimension for equality, checks conversion to the
local integer width, then uses the resolved dimension for both 64 MiB packing
and selected-response validation.

Empty candidate sets return before schema or vector reads. The cache is scoped
by collection name, stores only successful resolutions, and clears with the
existing collection lifecycle caches. Catalog ordering, scalar discovery,
exact legacy content queries, nullable identity, cold fallback behavior, one
selected Get, and all no-write and no-delete invariants remain unchanged.

## Original catch history

1. **[BLOCKER, ADDRESSED] An in-flight schema read could restore stale cache state after invalidation.**

   At `c8f978b640005c2d9eaee8b9366aa7e432ba8708`, a blocked
   `DescribeCollection` captured dimension 1,536. Cache invalidation then
   represented a same-name replacement with dimension 4,096. When the blocked
   call resumed, its unconditional Store repopulated 1,536. The next lookup
   returned stale 1,536 without describing the replacement.

   The deterministic review attack failed with:

   ```text
   dimension after invalidation = 1536, want 4096
   ```

   Commit `b474aacf94ea536bb912c29ecce67e1e553df98f` adds a
   per-name generation guarded by the same mutex as cache invalidation. A cache
   miss captures the generation. Invalidation increments the generation and
   deletes the value atomically. The later Store succeeds only if the captured
   generation still matches.

   The in-flight caller may finish with the dimension it already described.
   It cannot repopulate stale cache state. The next lookup describes the
   replacement and returns 4,096. The blocker is addressed.

## Adversarial attacks

| Attack | Result |
| --- | --- |
| Base red | The public zero-dimension regression failed at `ac2b581...` with `reuse vector dimension must be positive: 0`. |
| Positive source schema | The fixed public path returned the exact selected 4,096-float vector after scalar discovery and one selected Get. |
| Stale cache | Red at `c8f978b...`; green at `b474aac...`. The replacement lookup described and returned 4,096. |
| Concurrent readers | Eight blocked readers plus five invalidations passed under the race detector. Stale stores were rejected. |
| Different collections | Review-only 1,536 and 4,096 sources resolved and cached independently by collection name. |
| Successful-only cache | A zero schema dimension failed, was not cached, and a corrected schema was described again and accepted. |
| Cache error | An invalid cached type or nonpositive value returns an authoritative logged error. It cannot fall through to Describe or Get. |
| Describe failure | `Unavailable` propagated before selected Get, embedding, or mutation. |
| Invalid schemas | Missing schema, missing dense field, wrong type, absent dimension, malformed dimension, zero, and negative dimensions failed authoritatively. |
| Configured mismatch | A positive configured dimension that differed from the schema failed before Get. |
| Integer conversion | Negative configured values failed without caching. A maximum `int64` schema dimension failed packing overflow before Get. The round-trip conversion rejects values outside local `int`. |
| Empty candidates | No source Describe or selected Get occurred. |
| Missing selected IDs | Empty and partial selected responses failed before reuse was copied, with zero side effects. |
| Transport limit | The 65,536-byte fake transport test retained scalar discovery and one selected vector Get. |
| Hidden mutations | Public fakes recorded zero Embed, EmbedBatch, Insert, Delete, Upsert, CreateCollection, DropCollection, AddCollectionField, and CreateIndex calls. |
| Nil schema member | This direct internal shape panics, but Milvus client v2.6.5 converts each protobuf field, including nil entries, into a nonnil `entity.Field` before returning `Collection.Schema`; the shape cannot cross the typed client boundary. |

## Reproduced checks

| Check | Result |
| --- | --- |
| Initial functional diff | `git diff --check` exited 0. Three files changed. |
| Functional signature | `git verify-commit c8f978b640...` exited 0. The raw object contains `gpgsig` and the Codex trailer. |
| Race-fix diff | `git diff --check c8f978b640...b474aac...` exited 0. |
| Race-fix signature | `git verify-commit b474aacf...` exited 0. The raw object contains `gpgsig` and the Codex trailer. |
| Fresh final binary | Built after epoch `1788356450`; mtime was `1788356469`; build exited 0. |
| Fresh focused suite | The binary listed and passed 11 top-level tests and 20 subtests, including the zero-dimension path, schema failures, both generation tests, malformed selected responses, and fake transport. |
| Race detector | Four cache and invalidation tests passed under `-race` in 2.146 seconds. |
| Full semantic package | `go test -count=1 ./internal/semantic` exited 0 in 170.966 seconds. |
| Complete check gate | `GO_MK_SKIP_FETCH=1 make check` exited 0. All five checks passed. |
| Repository tests | `GO_MK_SKIP_FETCH=1 make test` exited 0 for every package. The reviewer run used Go cache; the producer's uncached semantic package passed in 173.103 seconds. |
| Fake transport | The standalone transport test exited 0 in 0.752 seconds. |
| Live preservation | All four isolated live tests exited 0 in 56.637 seconds. The duplicate smoke retained 16,387 source rows, one catalog row, and vector SHA-256 `1361a0b47601e409bb624759fbb6c1f32a338468e645a062f9f0fa6edcc74c6c`. |
| Merge reality | Fetched `origin/main` was `ac2b5814fc92fe66885aedc8d92697f8879effbd`. `git merge-tree --write-tree origin/main HEAD` exited 0 with tree `c86a3a45d268452f51b2ec1158c7d2c99d647c3c`. |

Exact scoped commands:

```sh
git fetch --prune origin
git diff --check ac2b5814fc92fe66885aedc8d92697f8879effbd..c8f978b640005c2d9eaee8b9366aa7e432ba8708
git diff --check c8f978b640005c2d9eaee8b9366aa7e432ba8708..b474aacf94ea536bb912c29ecce67e1e553df98f
git verify-commit c8f978b640005c2d9eaee8b9366aa7e432ba8708
git verify-commit b474aacf94ea536bb912c29ecce67e1e553df98f

env PKG_CONFIG_PATH=/Users/agoodkind/.worktrees/-Users-agoodkind-Sites-lm-semantic-search/lms-650-resolve-reuse-dimension/.make/cgo/darwin-arm64/lib/pkgconfig \
    CGO_LDFLAGS_ALLOW=-Wl,-rpath,@loader_path \
    go test -race ./internal/semantic \
    -run '^TestReuseSourceDimension(InvalidationRejectsInFlightCacheStore|RepeatedInvalidationRejectsConcurrentStores|CacheInvalidatesWithCollectionCaches|CacheStoresOnlySuccessfulResults)$' \
    -count=1 -v

/tmp/lms650-dimension-final.test -test.v -test.count=1 \
    -test.run '^(TestLoadReuseVectorsForContentsUsesSourceSchemaDimension|TestReuseSourceDimensionErrorsAreAuthoritative|TestReuseSourceDimensionCacheInvalidatesWithCollectionCaches|TestReuseSourceDimensionInvalidationRejectsInFlightCacheStore|TestReuseSourceDimensionRepeatedInvalidationRejectsConcurrentStores|TestReuseSourceDimensionCacheStoresOnlySuccessfulResults|TestEmptyReuseCandidatesSkipSourceSchema|TestLoadReuseVectorsForContentsBoundsCandidateTransportResponse|TestReuseSelectedRowValidation|TestReuseReadErrorHasNoSideEffects|TestReuseMissingSelectedIDHasNoSideEffects)$'

env PKG_CONFIG_PATH=/Users/agoodkind/.worktrees/-Users-agoodkind-Sites-lm-semantic-search/lms-650-resolve-reuse-dimension/.make/cgo/darwin-arm64/lib/pkgconfig \
    CGO_LDFLAGS_ALLOW=-Wl,-rpath,@loader_path \
    go test -count=1 ./internal/semantic

env GO_MK_SKIP_FETCH=1 make check
env GO_MK_SKIP_FETCH=1 make test

env PKG_CONFIG_PATH=/Users/agoodkind/.worktrees/-Users-agoodkind-Sites-lm-semantic-search/lms-650-resolve-reuse-dimension/.make/cgo/darwin-arm64/lib/pkgconfig \
    CGO_LDFLAGS_ALLOW=-Wl,-rpath,@loader_path \
    go test -tags=live ./test/live \
    -run 'Test(DuplicateLegacyCorpusReuseImmutabilitySmoke|UntaggedReuseAcrossCorpusPreservesSourceRow|ReuseCatalogStoresEachKnownEmbeddingModel|CompleteCatalogHitSkipsCollectionFallback)' \
    -count=1

git merge-tree --write-tree origin/main HEAD
```

## Current findings

No open blocker, should-fix, or nit remains. The stale-cache blocker is retained
above as addressed catch history.

## PR 262 test-cleanup re-review

The range
`00dc3cc193723dc3e6ca67d954c9eeeb1d279d35..3c619e3d8eb71162760da457bf6dec2e0c2a7670`
adds one test-only line. The timeout path now closes `releaseDescribe` before
`t.Fatal`, so a fake gRPC handler cannot remain blocked while cleanup waits for
the server to stop. The normal test path and all production code are unchanged.

| Review comment | Disposition | Evidence |
| --- | --- | --- |
| Release the blocked Describe on timeout | Valid and fixed | The added close occurs only on timeout and precedes `t.Fatal`; the normal path still closes the channel once after invalidation. |
| Guard nil `peerInfo.String()` | Invalid | Pinned gRPC v1.83.2 defines nil-safe `(*peer.Peer).String` and returns `Peer<nil>`. |

`git diff --check` and `git verify-commit 3c619e3...` passed. The raw commit
contains `gpgsig` and the Codex trailer. The five focused schema and cache tests,
including eight-reader repeated invalidation, passed under `-race` in 24.088
seconds. The one-line cleanup changes no assertion, cache ordering, schema read,
Milvus call, or production behavior.

No open finding remains from PR 262.

## Deliberately skipped

This review did not deploy the fix or restart the installed daemon. The
installed Codex index completion is post-merge acceptance. No merge verdict
depends on unverified deployment behavior.

## Evidence tiers

### Verified

- The deployed failure text and failing job identity.
- Functional red and green behavior at the public reuse boundary.
- Every schema, configuration, cache, invalidation, concurrency, transport,
  missing-response, and mutation attack listed above.
- Fresh focused, race, full semantic, Make, fake transport, live preservation,
  signature, and merge-tree results.

### Inferred

- A successfully cached dimension avoids another Describe until lifecycle
  invalidation. Direct call counts verify this process-local behavior.

### Assumed

- None supports the merge verdict.

## Ledger row

`2026-09-02 | codex/lms-650-resolve-reuse-dimension | source dimension and cache generation | strongest-model adversarial | MERGE-READY | B/SF/N 1/0/0 addressed | escapes 0 | stale cache race caught and fixed by b474aac`

`2026-09-02 | codex/lms-650-resolve-reuse-dimension | PR 262 test cleanup | strongest-model adversarial | MERGE-READY | B/SF/N 0/0/0 | escapes 0 | timeout cleanup fixed, nil-peer claim rejected`

MERGE-READY
