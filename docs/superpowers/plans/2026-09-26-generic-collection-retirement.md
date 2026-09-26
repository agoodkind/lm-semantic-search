# Conversation LMS protocol retirement implementation plan

## Goal

Remove the conversation-specific LMS protocol and obsolete engine conversion code after Clyde uses the generic collection RPCs for ingestion and search. Existing Milvus collections, stored vectors, scalar columns, and Merkle checkpoints remain readable without a rebuild.

## Current behavior

`proto/lmsemanticsearch/v1/service.proto` declares seven conversation-specific RPCs: `RegisterConversationCollection`, `SyncConversationManifest`, `UpsertConversationDocumentsStream`, `BackfillConversationScalars`, `DeleteConversation`, `SearchConversations`, and `SearchWithinConversation`. `internal/daemon/grpc_server.go`, `grpc_server_conversation_stream.go`, and `grpc_server_conversation_backfill.go` implement them. `internal/daemon/manager_conversations.go` and `item_source.go` include conversation-specific adapters and conversion. `internal/semantic` and `internal/localvec` include conversation-specific search, backfill, and old-row compatibility code. Clyde's current `internal/conversation/semsearch/client.go` calls all seven old RPCs.

## Constraints

- Start only after deployed CLYDE-629 and CLYDE-643 pass their parity gates. Confirm the oldest supported installed Clyde client uses the generic RPCs. A client still calling an old RPC blocks this removal.
- Remove protocol and conversion code; do not delete a Milvus collection, local vector file, registry record, stored row, vector, scalar column, or checkpoint. Keep the physical `conv/`, `convtool/`, and `convthink/` keys already stored.
- Preserve the ability to search old rows with null `conversationId`, `workspaceRoot`, or `archived` fields. Preserve the accepted conversation schema on registered codebase records. Keep only the compatibility reader or schema migration code required by those records and rows.
- Regenerate protobuf code with `make proto`. Do not edit generated Go files manually or reuse retired RPC method identifiers for different behavior.

## Tasks

### 1. Remove Clyde's remaining old wire calls

Files:

- Modify: `internal/conversation/semsearch/client.go` in Clyde
- Modify: `internal/conversation/semsearch/client_test.go` in Clyde
- Modify: `internal/daemon/generic_lms_search_live_test.go` in Clyde, if the temporary parity test remains
- Modify: `go.mod` and `go.sum` in Clyde if the final LMS version changes

Behavior:

- Clyde's production code calls only `RegisterCollection`, `SyncCollectionManifest`, `UpsertCollectionItemsStream`, `BackfillCollectionScalars`, `DeleteCollectionItem`, `SearchCollection`, and `GetCollectionItemState`.
- Remove old request encoders such as `conversationDocuments`, `conversationFingerprints`, `conversationSearchHits`, and `SearchFilter.wire()` after the generic equivalents replace them. Remove the temporary old/new parity call from shipped tests. Keep Clyde's conversation-level methods and domain types when daemon callers still use them.

Steps:

1. Inspect all old-RPC references in Clyde after the two cutovers. Remove each old wire call and conversion helper with no remaining generic use while Clyde still pins the generic-capable LMS release.
2. Delete tests that only assert retired wire framing. Keep or rewrite tests that assert public ingestion and search results through the generic client.
3. Build Clyde with `GOWORK=off` against the current generic-capable LMS release. After task 2 creates the LMS removal commit, pin that commit and repeat this build before deployment.

Verification:

- Run in Clyde: `GOWORK=off go test ./internal/conversation/semsearch ./internal/daemon ./internal/cli/daemon`
- Expect: Clyde's public ingestion and search tests pass. Repeat the command after pinning the LMS removal commit; that build proves Clyde has no old RPC dependency.

### 2. Remove seven old RPC declarations and handlers

Files:

- Modify: `proto/lmsemanticsearch/v1/service.proto`
- Modify: `internal/daemon/grpc_server.go`
- Delete or reduce: `internal/daemon/grpc_server_conversation_stream.go`
- Delete or reduce: `internal/daemon/grpc_server_conversation_backfill.go`
- Modify: `internal/daemon/grpc_server_conversation_stream_test.go`
- Modify: `gen/go/lmsemanticsearch/v1/service.pb.go` through `make proto`
- Modify: `gen/go/lmsemanticsearch/v1/service_grpc.pb.go` through `make proto`

Behavior:

- Remove all seven old RPC methods, old request and response messages, old stream frame messages, and their handler adapters. Keep only proto messages used by the generic service or other live APIs. Delete tests for removed RPC behavior; keep the same assertions at the generic public boundary.

Steps:

1. Remove the seven method declarations and their now-unused messages from the proto. Do not reuse field numbers or names in retained messages. Record that removing RPC methods is intentionally wire-breaking under the repository's `buf.yaml` `FILE` breaking policy; `make proto` regenerates code but does not prove backward compatibility.
2. Run `make proto`. Remove old handlers and stream validators. Leave generic handlers and their input validation intact.
3. Update the public gRPC tests to call generic RPCs for registration, manifest, upsert, backfill, delete, search, and item state.

Verification:

- Run in LMS: `make proto && go test ./internal/daemon`
- Expect: the generated service exposes the seven generic operations and none of the retired conversation operations.

### 3. Remove conversation-only manager and store conversions

Files:

- Modify: `internal/model/types.go`
- Modify: `internal/daemon/manager_conversations.go`
- Modify: `internal/daemon/item_source.go`
- Delete or reduce: `internal/daemon/manager_conversation_text.go`
- Delete or reduce: `internal/daemon/manager_conversation_tools.go`
- Delete or reduce: `internal/daemon/manager_conversation_storable.go`
- Modify: `internal/semantic/collection.go`
- Delete or reduce: `internal/semantic/conversation_filter_expr.go`
- Delete or reduce: `internal/semantic/conversation_search.go`
- Delete or reduce: `internal/semantic/conversation_backfill.go`
- Delete or reduce: `internal/semantic/conversations.go`
- Modify: `internal/localvec/conversation.go`
- Modify: conversation-only tests beside those files
- Create: `test/live/generic_collection_legacy_rows_live_test.go`

Behavior:

- Remove `ConversationDocument` and `ConversationToolCall` only when no generic type or adapter uses them. Remove `conversationItemSource`, `fingerprintConversationDocuments`, conversation row generation, conversation filter compilation, and conversation-only backfill and delete adapters after their generic equivalents pass the same behavior tests.
- Keep generic delta indexing, chunk splitting, content-hash reuse, job coalescing, scalar backfill, group-cap search, and item deletion. Preserve the existing scalar-column migration and asynchronous backfill. A native `conversationId` filter does not match null values before backfill; test legacy rows before and after migration. Validate each existing Milvus schema before adopting a declaration for a document codebase without one; validate only saved declarations in the offline profile.

Steps:

1. Validate every legacy document record against its live collection schema and persist the accepted declaration. Abort removal if a record cannot be validated; keep that record and its collection unchanged.
2. Inspect each listed symbol after the RPC removal. Move any generic behavior still required into the generic implementation before deleting its conversation wrapper. Do not retain a second implementation of the same ingest or search operation.
3. Remove the obsolete model types, manager adapters, row conversion, and store wrappers. Delete tests for removed behavior and keep equivalent generic public-boundary coverage.
4. Add an isolated live-harness test that reads rows with null conversation scalars before and after the supported backfill. The harness uses a per-test Milvus database and cannot inspect the production collection. Separately, use an operator read-only store audit to record production row and checkpoint counts before and after deployment. Run production search only after confirming its schema is already prepared; `PrepareCollection` can add columns and start backfill.

Verification:

- Run in LMS: `go test ./internal/daemon ./internal/semantic ./internal/localvec && go test -tags live -run '^TestGenericCollectionReadsLegacyRows$' -count=1 ./test/live/`
- Expect: generic ingestion, maintenance, search, and isolated legacy-row reads pass. The separate operator audit confirms unchanged production row and checkpoint counts.

### 4. Publish the protocol removal

Files:

- Modify: `docs/conversationingest/overview.md`
- Modify: `README.md` only if it still documents an old RPC

Behavior:

- Documentation lists the generic collection RPCs as the supported contract and removes instructions for the old conversation RPCs. Clyde's deployed ingest, search, scalar backfill, and context-window read continue to succeed.

Steps:

1. Remove retired RPC instructions from the existing documentation and inspect all remaining references for a real compatibility need.
2. Run the full LMS gates. Pin the removal commit in Clyde and run Clyde's gates before deploying LMS. Deploy the LMS removal release, then repeat Clyde's live ingest, search, and `--around` checks against the existing collection. Compare the separate pre-deployment and post-deployment store observations and run an unchanged manifest sync before claiming no re-offer or re-embedding.

Verification:

- Run in LMS: `make test && make check`
- Run in Clyde: `GOWORK=off make test && GOWORK=off make check`
- Expect: both repositories pass. Separate deployed checks prove the generic client serves the existing collection and an unchanged manifest begins no re-offer or re-embedding.
