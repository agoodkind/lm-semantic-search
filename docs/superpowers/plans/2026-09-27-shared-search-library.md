# Shared search library implementation plan

## Goal

Clyde and codebase search import `goodkind.io/lm-semantic-search/library`. The library stores one live canonical vector per exact embedding identity, stores source occurrences separately, and returns complete filtered pages. Each client owns source policy, extraction, and Milvus connection configuration. The [design](../specs/2026-09-27-shared-search-library-design.md) defines storage, search, and recovery.

## Current behavior

[Vector reuse](../../../internal/semantic/reuse_catalog.go) stores reusable vectors, while [staging](../../../internal/semantic/staging.go) writes another vector on every source row. [Collection search](../../../internal/semantic/collection_search.go) filters those rows and ranks a fixed candidate depth. The [daemon registry](../../../internal/store/store.go) persists registry and jobs in JSON and JSONL.

## Constraints

- The caller owns the Milvus client. `Close` releases library resources only. No new service or RPC is required.
- New code and conversation writes use the shared layout. Existing collections remain until a deliberate migration or retirement. The new layout does not promise TypeScript schema compatibility.
- Codebase search retains replacement and deletion. Clyde retains append-only historical occurrences.
- Search returns complete pages or a typed error. It never returns partial success or uses a fixed global candidate cutoff.
- Public tests use SQLite and isolated real Milvus. The existing offline profile uses an exact embedded vector adapter and must pass its existing acceptance suite before codebase cutover.

## Pull request lanes

| Lane | Deliverable | Dependency | Owned files |
| --- | --- | --- | --- |
| L0 | Public contracts, generic splitter, embedding adapter, and importable build | None | `library/types.go`, `library/config.go`, `library/errors.go`, `library/open.go`, `library/prepare.go`, `library/embedding/adapter.go`, `library/import_test.go` |
| L1 | Transactional occurrence and vector writes | L0 | `library/catalog_*.go`, `library/vector_*.go`, `library/ingest_*.go`, `library/recovery_*.go` |
| L2 | Occurrence-weighted lexical ranking | L0; parallel with L1 | `library/lexical_*.go`, `library/lexical_schema.go` |
| L3 | Complete filtered search | L1 and L2 | `library/filter_*.go`, `library/search_*.go`, `library/snapshot_*.go` |
| L4 | Codebase adoption | L1 and L3 | [manager_search.go](../../../internal/daemon/manager_search.go), [manager_delta.go](../../../internal/daemon/manager_delta.go), [staging.go](../../../internal/semantic/staging.go), code adapter |
| L5 | Conversation-specific LMS removal | Clyde cutover and L4 verified | [service.proto](../../../proto/lmsemanticsearch/v1/service.proto), generated bindings, conversation-specific daemon and semantic modules |

Each lane uses a separate pull request. L1 and L2 agree on SQL schema extensions before editing. L3 integrates both. L4 and L5 follow in sequence. L5 rebases after L4 and removes only code with no remaining callers.

L0 publishes contracts and an importable build, not a running search implementation. Clyde can compile its adapter after L0. Ingestion starts after L1; search starts after L3. L2 owns `library/lexical_schema.go`. L1 registers that migration during integration. The two workers edit separate schema files.

## L0: Publish the facade

Files:

- Create: `library/types.go`, `library/config.go`, `library/errors.go`, `library/open.go`, `library/prepare.go`, `library/embedding/adapter.go`, `library/import_test.go`
- Modify: [embedding.go](../../../internal/embedding/embedding.go) and provider files to expose the existing production adapter through the new importable package without a second implementation
- Modify: [go.mod](../../../go.mod) only for a required direct dependency

Behavior:

- Export the exact `StoreDescriptor`, `Config`, `Embedder`, `VectorStore`, `NamespaceSpec`, `ScalarValue`, `Filter`, `OccurrenceID`, `Occurrence`, `Batch`, `GenerationKey`, `GenerationSeal`, `SearchRequest`, `SearchHit`, `SearchPage`, `ApplyReceipt`, and `OwnerState` contracts in the design. `SearchPage` returns exact `HasMore` and `NextCursor`. Export the design's typed errors.
- `Open` validates canonical catalog and lock paths, backend pool identity, model revision, dimensions, and normalization. `Close` leaves the caller's vector backend usable. `PrepareText` splits model-sized text into stable prepared parts; clients own source selection, not generic splitting.

Steps:

1. Declare consumer-facing interfaces and types, implement validation and generic `PrepareText`, and export the production embedding adapter. Do not create no-op successful implementations of methods assigned to later lanes.
2. Build an external module that imports the pinned LMS module without a local `replace`. Pin a reproducible `gksyntax` submodule workspace for Clyde because the published module ZIP omits its C sources. Do not claim `GOWORK=off` works until the dependency becomes self-contained.
3. Compile on supported macOS and Linux toolchains with CGO enabled.

Verification:

- Run: `make test && make check`.
- Run: the external import fixture in a clean directory with the pinned submodule workspace.
- Expect: the public package imports and compiles; `Close` does not close Milvus.

## L1: Publish vectors and occurrences

Files:

- Create: `library/catalog_schema.go`, `library/catalog.go`, `library/catalog_lock.go`, `library/vector_store.go`, `library/vector_outbox.go`, `library/ingest.go`, `library/recovery.go`, `library/milvus/store.go`, `library/embedded/store.go`, `test/live/library_write_live_test.go`
- Modify: `library/open.go`

Behavior:

- Create the SQLite WAL schema from the design. Store source text separately from exact embedding input identity. Compare full identity bytes before vector reuse.
- Use a kernel cross-process writer lock at the canonical lock path. Persist complete vector payload in a durable outbox before Milvus RPC. Use deterministic upsert and strong-read verification of primary key, identity, and checksum. Publish occurrence generation only after verification. Recover ambiguous timeouts and interrupted writes by replay. Do not automatically sweep orphan vectors.
- `AppendOnly` namespaces reject `Batch.Mode=Replace` and `Delete`. Append permits identical replay but rejects changed history. `Stage` writes bounded batches; `CommitGeneration` verifies row count and sorted manifest hash from `GenerationSeal`, then publishes all staged rows atomically. Replace swaps one complete owner generation. Persist old batch receipts for idempotent replay after later generations. `ReprojectScalars` appends a metadata event and atomically updates only declared mutable scalars with zero vector writes. The Milvus adapter binds the catalog UUID in the actual vector collection's initial schema description and rejects a different UUID. The embedded adapter enforces the same descriptor and exact scoring contract.

Steps:

1. Implement schema, descriptor checks, WAL transactions, and process locking.
2. Implement vector identity, outbox, bounded embedding batches, deterministic upsert, and restart replay.
3. Implement owner generation, staged publication, metadata projection, durable receipts, and explicit deletion.
4. Test `Open`, `Stage`, `CommitGeneration`, `Apply`, `ReprojectScalars`, `Delete`, and `GetOwnerState` with a temporary SQLite catalog and isolated real Milvus. Restart a helper process at each durability boundary. Include truncated staged streams, a shrinking code file, and a two-process first-open catalog-binding race.

Verification:

- Run: `go test -tags live -run '^TestLibraryWrite' -count=1 ./test/live/`.
- Run: `make test && make check`.
- Expect: one canonical live vector per identity, no occurrence references an unverified vector, and cross-process replay preserves append and replace rules. Measure compacted backend storage separately.

## L2: Rank lexical matches per occurrence

Files:

- Create: `library/lexical_analyzer.go`, `library/lexical_index.go`, `library/lexical_rank.go`, `test/live/library_lexical_live_test.go`
- Create: `library/lexical_schema.go` with lexical migration definitions; L1 registers the migration at integration after both lanes finish

Behavior:

- Index `SearchText` terms separately from model embedding input. Store terms in indexed postings and record occurrence multiplicity. Compute namespace corpus size, term counts, document lengths, and document frequencies before request filters. Version the local analyzer and BM25 parameters. Preserve repeated query terms and verify arithmetic against Milvus. Use disk-backed accumulation beyond memory bounds.

Steps:

1. Agree on SQL schema fields with L1 before either lane edits the schema.
2. Implement postings and occurrence-weighted scoring.
3. Compare public results with an independent exhaustive per-occurrence BM25 oracle on duplicate-heavy data.

Verification:

- Run: `go test -tags live -run '^TestLibraryLexical' -count=1 ./test/live/`.
- Expect: duplicate rows affect corpus statistics and ranks exactly as separate documents.

## L3: Return complete filtered pages

Files:

- Create: `library/filter_validate.go`, `library/filter_sql.go`, `library/search_dense.go`, `library/search_hybrid.go`, `library/search_merge.go`, `library/search_snapshot.go`, `test/live/library_search_live_test.go`

Behavior:

- Validate the typed filter, literal `Prefix`, and group column against namespace declarations. SQLite evaluates committed occurrences with current effective metadata, copies eligible occurrences, namespace corpus statistics, and selected vector IDs into a temporary query database under one committed snapshot, then releases the read transaction.
- Partition vector IDs into bounded backend requests. The Milvus adapter uses exact `FLAT` scoring; the embedded adapter uses exact selected-vector scoring. Verify the exact requested ID set for every block. Merge scores on disk and expand to eligible occurrences. Compute the lexical leg from the same active corpus snapshot. Fuse ranks with versioned RRF and order ties by `SortKey`, then occurrence ID. Apply a positive score floor, one global group cap, then paging.
- Persist the ordered result with immutable source blob IDs and the effective scalar projection. Pin those blobs through snapshot expiry. The opaque cursor identifies snapshot, request hash, corpus generation, rank configuration, and next ordinal. `HasMore` is exact. A changed request or expired snapshot returns a typed error. A later owner replacement does not alter the pages. Deadline or resource failure returns no success page.

Steps:

1. Implement typed validation and SQL predicates with distinct null and absent behavior.
2. Implement exhaustive partition scoring and disk-backed merge.
3. Integrate L2 scoring and persist ordered snapshots.
4. Test 20,000 or more eligible IDs, duplicate content, group saturation, nulls, writes between pages, and deadline failures against an independent exhaustive oracle.

Verification:

- Run: `go test -tags live -run '^TestLibrarySearch' -count=1 ./test/live/`.
- Run: `make test && make check`.
- Expect: requested pages fill when enough rows qualify; every eligible result appears in global order; failures produce no partial success.
- Measure p50, p95, worst-case latency, peak memory, snapshot disk bytes, vector count, and post-compaction storage on an isolated production-shaped corpus. Compare representative queries to a matched healthy recovered baseline. Require zero missing hits and no timeout on the agreed workload. A latency regression needs a user-approved threshold before cutover.

## L4: Adopt the library for codebase search

Files:

- Modify: [manager_search.go](../../../internal/daemon/manager_search.go), [manager_delta.go](../../../internal/daemon/manager_delta.go), [staging.go](../../../internal/semantic/staging.go), [service.go](../../../internal/semantic/service.go)
- Create: `internal/daemon/library_code_adapter.go`, `test/live/library_codebase_live_test.go`

Behavior:

- Preserve file extraction and chunk identities. Use library `PrepareText` after source extraction. Supply exact embedding input, `SourceText`, `SearchText`, path and language scalars, and owner generation. Stage every file part and seal its complete manifest before Replace publication. Use `Delete` only for codebase retention. Keep old collections until deliberate migration. Switch reads after parity, completeness, and performance gates. A failed new build leaves old search available. Preserve the existing offline profile through the embedded adapter.

Steps:

1. Add the code adapter and validate both paths in an isolated testbed.
2. Build new occurrences from code sources, verify owner and chunk counts, then switch reads.
3. Stop duplicate vector writes on the new path. Do not clear old collections.

Verification:

- Run: `go test -tags live -run '^TestLibraryCodebase' -count=1 ./test/live/`.
- Run: `make test && make check`.
- Expect: changed and deleted files follow codebase retention, interrupted builds do not expose partial owner generations, and pages are complete.

## L5: Remove conversation-specific LMS code

Files:

- Modify: [service.proto](../../../proto/lmsemanticsearch/v1/service.proto), [service.pb.go](../../../gen/go/lmsemanticsearch/v1/service.pb.go), [service_grpc.pb.go](../../../gen/go/lmsemanticsearch/v1/service_grpc.pb.go), [grpc_server.go](../../../internal/daemon/grpc_server.go), [conversation ingest](../../conversationingest/overview.md)
- Delete candidates after call-site proof: [manager_conversations.go](../../../internal/daemon/manager_conversations.go), [manager_conversation_text.go](../../../internal/daemon/manager_conversation_text.go), [manager_conversation_storable.go](../../../internal/daemon/manager_conversation_storable.go), [manager_conversation_tools.go](../../../internal/daemon/manager_conversation_tools.go), [grpc_server_conversation_backfill.go](../../../internal/daemon/grpc_server_conversation_backfill.go), [grpc_server_conversation_stream.go](../../../internal/daemon/grpc_server_conversation_stream.go), [conversation_search_filter.go](../../../internal/daemon/conversation_search_filter.go), [conversation_batch.go](../../../internal/semantic/conversation_batch.go), [conversation_filter_expr.go](../../../internal/semantic/conversation_filter_expr.go), [conversation_backfill.go](../../../internal/semantic/conversation_backfill.go), [conversation_state.go](../../../internal/semantic/conversation_state.go), and [conversation_columns.go](../../../internal/semantic/conversation_columns.go)
- Review and remove generic wrappers that only serve those conversation modules, including their tests and conversation-specific configuration and documentation
- Modify: tests for the retired protocol

Behavior:

- Remove seven conversation RPCs and their LMS implementation only after Clyde's direct library path passes testbed end-to-end validation. Preserve generic library mechanics and existing collections. Remove tests for deleted behavior and retain real public-boundary shared-library tests.

Steps:

1. Inventory all protocol and source callers and confirm Clyde uses the library.
2. Remove conversation-specific code and regenerate bindings with `make proto`.
3. Verify Clyde append ingestion, search, restart recovery, complete pages, and codebase search on testbed against pinned revisions.

Verification:

- Run: `make proto && make test && make check`.
- Run: the Clyde and LMS testbed acceptance suite.
- Expect: generated service omits conversation RPCs; both clients pass public behavior gates.
