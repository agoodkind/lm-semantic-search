# Return complete filtered search pages

## Goal

Implement complete dense and hybrid search through the [public library contract](../specs/2026-09-27-shared-search-library-design.md). L3 integrates L1 storage and L2 lexical ranking. The [coordination plan](https://github.com/agoodkind/clyde/blob/docs/embedded-conversation-search/docs/superpowers/plans/2026-09-27-shared-search-coordination.md) controls cross repository acceptance.

## Current behavior

[Collection search](../../../internal/semantic/collection_search.go) requests a fixed number of backend candidates. A filtered page can end before enough eligible rows are ranked. The new library query must enumerate every eligible vector identity and maintain occurrence multiplicity.

## Constraints

- A query reads committed occurrences and effective metadata from one SQLite snapshot. Copy the selected IDs, rank configuration, requested term statistics, metadata, and blob references into a temporary query database, then release the read transaction.
- All selected vector identities remain pinned and immutable until cursor expiry. A missing or changed selected vector is an error.
- Dense uses exact COSINE scores over the selected IDs. Hybrid adds occurrence-weighted BM25 and RRF. `Config.SearchMode` selects the mode.
- Resource exhaustion, deadline expiry, backend cardinality mismatch, and expired cursors return typed errors without a successful partial page.

## Tasks

### 1. Compile typed predicates against one snapshot

Files:

- Create `library/filter_validate.go`, `library/filter_sql.go`, and `library/search_snapshot.go`.
- Create `test/live/library_search_live_test.go`.

Steps:

1. Validate declared scalar type, `All`, `Any`, `Not`, `Equal`, `In`, `Range`, `Prefix`, `IsNull`, and `IsPresent` nodes. Treat `Prefix` as literal text and bind SQL values instead of interpolating them.
2. Select committed occurrences using current effective scalar projections. Distinguish absent from null. Copy selected occurrence IDs, vector IDs, sort keys, group values, effective scalars, source blob references, corpus generation, analyzer/model identity, and requested term statistics under one SQLite read transaction.
3. Release the read transaction before backend scoring. Pin vector and blob references through cursor expiry. Persist a request hash and snapshot generation; reject changed request fields or expired cursors.
4. Test nested predicates, escaped literal prefixes, null versus absent values, metadata reprojection between pages, and owner replacement between pages. Later changes cannot alter a persisted page sequence.

### 2. Score all eligible vector IDs

Files:

- Create `library/search_dense.go`, `library/search_hybrid.go`, and `library/search_merge.go`.

Steps:

1. Divide distinct selected vector IDs into bounded blocks. Ask `VectorStore.ScoreExact` for every ID in each block. The Milvus adapter uses exact FLAT COSINE. Verify one score for every requested ID and reject duplicates or missing IDs.
2. Spill vector scores to the query database, expand scores to every selected occurrence, and assign stable modality ranks by raw score descending and occurrence ID ascending.
3. Add L2 BM25 scores only in hybrid mode. A sparse absence contributes zero and has no sparse rank. Fuse ranks with configured positive RRF `k`. Sort final scores descending, `SortKey` ascending, then occurrence ID ascending.
4. Apply a positive `MinScore`, then one group quota over the entire ordered result, then page ordinals. Persist ordered occurrence IDs and immutable blob references rather than copying each excerpt per query. Return exact `HasMore`, `NextCursor`, source text, and effective scalars.

### 3. Prove completeness and measure cost

Files:

- Extend `test/live/library_search_live_test.go` through the library public boundary and isolated real dependencies.

Steps:

1. Build more than 20,000 distinct eligible vector identities plus duplicate occurrences, multiple groups, and filtered matches below the prior fixed depth. Compare every page with an independent exhaustive per-occurrence oracle.
2. Test page sizes 1, 10, and 100, cursor replay, score ties, group saturation, writes between pages, backend missing IDs, timeouts, disk budget, and expiry. Assert no empty page before a later eligible hit and no success page after a failure.
3. Record p50, p95, and worst-case latency, peak memory, query disk bytes, vector count, and post-compaction storage on an isolated production-shaped corpus. Compare representative queries to a matched healthy recovered baseline. Require zero missing hits and no timeout on the agreed workload before cutover. Route a material latency regression through the coordination plan.

## Verification

- Run `make library-live-l3`. Its strict runner must report executed passing tests with no skips.
- Run `make test && make check` and `make offline-live`.
- Compare full ordered occurrence IDs, scores, `HasMore`, and final cursor exhaustion with the exhaustive oracle. A successful short or prematurely empty page fails acceptance.
