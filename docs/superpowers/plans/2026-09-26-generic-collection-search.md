# Generic collection search implementation plan

## Goal

Complete LMS-18. A client searches a registered document collection with typed filters, group caps, and a score floor, then reads one item's indexed fingerprint. Existing conversation RPCs use the generic search implementation and return the same results.

## Current behavior

`proto/lmsemanticsearch/v1/service.proto` declares `SearchConversations` and `SearchWithinConversation`. `pbConversationSearchFilter` in `internal/daemon/grpc_server.go` decodes the wire filter; `internal/daemon/conversation_search_filter.go` converts the manager filter for semantic storage. `internal/semantic/conversation_filter_expr.go` builds Milvus expressions and pages ranked results for smaller ID sets. `internal/semantic/conversation_search.go` embeds queries and batches searches when a filter contains more than 256 conversation IDs. `internal/daemon/manager_conversations.go` reads the indexed fingerprint from the Merkle checkpoint. Older rows can have null workspace and archived scalars.

Both paths return different results for different page sizes. `fillCappedConversationSearchWith` pages the hybrid search by offset, but each hybrid leg retrieves only `max(limit, 10)` rows. A page after the first reranks only those rows and comes back short, which ends the fill early. A live harness query for 10 hits over 30 matching rows with a per-conversation cap of 2 returns 2 hits. The batched path sorts hits from separately fused searches by score, and RRF scores from separate searches are not comparable.

## Constraints

- LMS-15 registration is required. LMS-16 and LMS-17 can proceed independently.
- Clyde constructs conversation filters before it calls a retrieval provider. LMS compiles a typed filter tree into its Milvus expression. The generic request never accepts a raw Milvus expression.
- Validate every filter and group column against the saved declaration. Reject unknown columns, incorrect value types, excessive tree depth, and oversized membership sets before query execution. Size the membership limit for Clyde's full allowed conversation set.
- Preserve query embedding, collection leases, score-floor behavior, and `loadRules` on hits.
- Make search deterministic. Each query computes one ranking at a fixed fusion depth that does not depend on the limit or group cap. Equal scores order by `relativePath`, then primary key. One walk over that ranking applies the score floor, the group cap, and the limit. A smaller limit returns a prefix of a larger limit's results. No search merges separately fused results.
- Keep the old search RPCs until CLYDE-643 passes.
- Apply the daemon's existing maintenance refusal to generic search before collection load or query execution. Preserve the old search handlers' maintenance error when they delegate.

## Pull request boundary

Implement the Task 2 single-ranking change for the existing conversation search RPCs as its own pull request first, with its regression and stability tests. Implement the rest of LMS-18 Tasks 1 through 3 in one pull request on top of it. The new RPC, expression compiler, public tests, live parity battery, generated proto code, and documentation must pass together before merge. LMS-18 depends on LMS-15; it does not require LMS-16 or LMS-17. Integrate against the latest `service.proto` before merging. Implement Task 4 as the separate protocol retirement pull request after both Clyde cutovers.

## Tasks

### 1. Add generic search and item-state RPCs

Files:

- Modify: `proto/lmsemanticsearch/v1/service.proto`
- Modify: `internal/daemon/grpc_server.go`
- Modify: `internal/daemon/conversation_search_filter.go`
- Modify: `internal/daemon/manager_conversations.go`
- Create: `internal/daemon/grpc_server_collection_search.go`
- Create: `internal/daemon/collection_search_filter.go`
- Create: `internal/daemon/grpc_server_collection_search_test.go`

Behavior:

- `SearchCollection` accepts collection ID, query, limit, `min_score`, optional `group_by`, optional `per_group_limit`, and an AND/OR/NOT filter tree. Leaves support equality, set membership, inclusive lower and exclusive upper numeric or time bounds, null, and presence. The old adapter preserves `from_unix >=`, `until_unix <`, `message_index_from >=`, `message_index_until <`, lowercase role matching, exact provider and workspace matching, and archived-null exclusion for either boolean value.
- Each hit returns logical `row_key`, content, score, and the declared scalar values. Preserve null versus absent values for old rows.
- `GetCollectionItemState` accepts collection ID and item ID and returns its Merkle checkpoint fingerprint. An unknown item returns an empty fingerprint.
- Old conversation handlers convert their filters into the same tree. `per_conversation_limit` becomes `group_by=conversationId` with `per_group_limit`. Preserve their default limit of 10. `SearchConversations` returns an empty result without registration when the collection is absent; `SearchWithinConversation` registers first, scopes to one ID, and reads its checkpoint through the item-state method.

Steps:

1. Add typed proto requests, filter nodes, scalar hit values, and responses. Run `make proto`.
2. Add a generic manager search method. Validate the tree against the saved schema, then call the existing semantic search path. Reuse the existing checkpoint reader for item state.
3. Adapt both old handlers without changing their wire contracts, default limits, or empty-filter meaning.
4. Add public gRPC tests with the real temporary store and embedder. Cover each predicate, nested boolean expressions, invalid columns and types, nulls, group caps, score floor, returned `loadRules`, and missing item state.

Verification:

- Run: `make proto && go test ./internal/daemon`
- Expect: invalid filters fail before expression execution; both old search RPCs return the same results.

### 2. Generalize expression compilation and rank once per query

Files:

- Modify: `internal/semantic/conversation_filter_expr.go`
- Modify: `internal/semantic/conversation_search.go`
- Modify: `internal/semantic/service.go`
- Modify: `internal/semantic/result_sets.go`
- Modify: `internal/localvec/conversation.go`
- Modify: `internal/localvec/search.go`
- Create: `internal/semantic/collection_filter_expr.go`
- Create: `internal/semantic/collection_search.go`
- Create: `internal/semantic/collection_search_test.go`

Behavior:

- Compile only validated declared columns into Milvus syntax. Escape literal values with the existing rules. Apply the same typed predicates in the local vector store. Send a large membership set in one search through Milvus expression template parameters. Old and generic RPCs never batch a membership set into separate searches.
- Replace offset paging with one ranking per query. Embed the query once. Run both hybrid legs at one fixed depth, starting at the 16,384-row Milvus ceiling. Request only the primary key, `relativePath`, the group column, and the score for the ranking. Sort, then walk the ranking once for the score floor and the group cap. Read group identity from its scalar column.
- Query content and output scalars by primary key for the selected rows only, and return them in ranking order.
- Request every declared native scalar from Milvus and decode it through a typed hit representation that distinguishes absent, null, and concrete values. Return stored `relativePath` as `row_key`. Keep old JSON-metadata decoding for legacy identity and old response fields.

Steps:

1. Extract the generic expression builder from `buildExpr` and keep `buildExpr` as a conversation adapter.
2. Build the single ranking and walk for the existing conversation search first. Remove offset paging and the batched merge. Confirm that Milvus 2.6.18 accepts a membership set of 20,000 IDs through template parameters. Measure ranking latency at the fixed depth on the live harness.
3. Parameterize the walk by group column for the generic RPC.
4. Implement matching filter, ordering, and grouping behavior in `internal/localvec`.
5. Test through the public gRPC boundary with a corpus where the rows ranked first contain too many hits from one group. Test that repeated queries return the same order and that a smaller limit returns a prefix of a larger one. Test one search over a large membership set.

Verification:

- Run: `go test ./internal/semantic ./internal/localvec ./internal/daemon`
- Expect: a capped search fills its limit whenever enough rows qualify within the ranking depth. Old and generic RPCs return equal shared fields, order, and scores. Direct store reads confirm row keys and native scalar values because the old response does not expose them.

### 3. Prove live parity and release the search surface

Files:

- Modify: `docs/conversationingest/overview.md`
- Create: `test/live/generic_collection_search_live_test.go`

Behavior:

- A read-only live battery calls both RPCs on the same isolated harness collection. Queries cover provider, role, time, message index, parent, workspace, archived, large conversation-ID sets, group caps, score floor, and within-conversation fingerprints. Repeat each query and compare limits of 5 and 10 to confirm the same order and a shared prefix. Compare shared wire fields, ordered scores, and fingerprints; inspect stored rows separately for row-key and native-scalar parity. Do not print transcript content.
- Clyde continues calling the old RPCs during this LMS release.

Steps:

1. Add the battery to `test/live/harness.go`. Report only identifiers and scalar metadata needed to diagnose mismatches.
2. Document the typed request and scalar echo in the existing ingest documentation. Deploy LMS with Clyde unchanged and run the parity battery before CLYDE-643.

Verification:

- Run: `go test -tags live -run '^TestGenericCollectionSearchParity$' -count=1 ./test/live/`
- Expect: exact shared hit fields, order, scores, and fingerprint parity against the conversation RPCs; direct store inspection confirms row keys and native scalars.
- Run: `make test && make check`
- Expect: repository tests and lint pass, and deployed Clyde search still succeeds through the old RPCs.

### 4. Retire the conversation protocol after both Clyde cutovers

Files:

- Create: `docs/superpowers/plans/2026-09-26-generic-collection-retirement.md`

Behavior:

- Execute the separate [protocol retirement plan](2026-09-26-generic-collection-retirement.md) after CLYDE-629 and CLYDE-643 pass their deployed gates. That plan removes all seven conversation-specific RPCs, their handlers and obsolete conversion paths, and Clyde's old wire helpers. It preserves the existing collection schema, stored rows, and checkpoint data.

Steps:

1. Confirm both Clyde cutovers and the supported installed-client version before removing the old protocol.
2. Execute the protocol retirement plan as a separate change.

Verification:

- Run: `make proto && make test && make check`
- Expect: the generic RPCs remain available and the retired conversation RPCs no longer appear in the generated service.
