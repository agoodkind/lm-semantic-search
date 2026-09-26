# Generic collection search implementation plan

## Goal

Complete LMS-18. A client searches a registered document collection with typed filters, group caps, and a score floor, then reads one item's indexed fingerprint. Existing conversation RPCs use the generic search implementation and return the same results.

## Current behavior

`proto/lmsemanticsearch/v1/service.proto` declares `SearchConversations` and `SearchWithinConversation`. `pbConversationSearchFilter` in `internal/daemon/grpc_server.go` decodes the wire filter; `internal/daemon/conversation_search_filter.go` converts the manager filter for semantic storage. `internal/semantic/conversation_filter_expr.go` builds Milvus expressions and pages ranked results for smaller ID sets. `internal/semantic/conversation_search.go` embeds queries and batches searches when a filter contains more than 256 conversation IDs. `internal/daemon/manager_conversations.go` reads the indexed fingerprint from the Merkle checkpoint. Older rows can have null workspace and archived scalars.

## Constraints

- LMS-15 registration is required. LMS-16 and LMS-17 can proceed independently.
- Clyde constructs conversation filters before it calls a retrieval provider. LMS compiles a typed filter tree into its Milvus expression. The generic request never accepts a raw Milvus expression.
- Validate every filter and group column against the saved declaration. Reject unknown columns, incorrect value types, excessive tree depth, and oversized membership sets before query execution.
- Preserve query embedding, collection leases, ID batching, deterministic ordering, score-floor behavior, and `loadRules` on hits. The old RPC uses paged cap fill for smaller ID sets and batched limited searches followed by cap reduction for more than 256 IDs; preserve both paths until the old RPC retires.
- Keep the old search RPCs until CLYDE-643 passes.
- Apply the daemon's existing maintenance refusal to generic search before collection load or query execution. Preserve the old search handlers' maintenance error when they delegate.

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

### 2. Generalize expression compilation and ranked cap fill

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

- Compile only validated declared columns into Milvus syntax. Escape literal values with the existing rules. Apply the same typed predicates in the local vector store. For more than 256 item IDs, both old and generic RPCs use the existing per-batch limited search, ordered merge, and cap reduction during this cutover. Test that path separately from paged cap fill.
- Generalize `fillCappedConversationSearchWith` for the generic paged path: embed the query once, request ranked pages, apply the group cap and score floor at the existing stage, preserve tie order, and stop at the existing 16384-row window. Read group identity from its scalar column.
- Request every declared native scalar from Milvus and decode it through a typed hit representation that distinguishes absent, null, and concrete values. Return stored `relativePath` as `row_key`. Keep old JSON-metadata decoding for legacy identity and old response fields.

Steps:

1. Extract the generic expression builder from `buildExpr` and keep `buildExpr` as a conversation adapter.
2. Parameterize the cap-fill reducer by group column. Keep the existing pagination and score comparisons for the paged path. Retain the old batched path for more than 256 conversation IDs and test its ordered merge separately.
3. Implement matching filter and grouping behavior in `internal/localvec`.
4. Test through the public gRPC boundary with a corpus where the first ranked page contains too many hits from one group and a later page must fill the result limit.

Verification:

- Run: `go test ./internal/semantic ./internal/localvec ./internal/daemon`
- Expect: old and generic RPCs return equal shared fields, order, and scores. Direct store reads confirm row keys and native scalar values because the old response does not expose them.

### 3. Prove live parity and release the search surface

Files:

- Modify: `docs/conversationingest/overview.md`
- Create: `test/live/generic_collection_search_live_test.go`

Behavior:

- A read-only live battery calls both RPCs on the same isolated harness collection. Queries cover provider, role, time, message index, parent, workspace, archived, large conversation-ID sets, group caps, score floor, and within-conversation fingerprints. Compare shared wire fields, ordered scores, and fingerprints; inspect stored rows separately for row-key and native-scalar parity. Do not print transcript content.
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
