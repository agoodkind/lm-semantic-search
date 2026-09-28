# Remove the conversation service path from LMS

## Goal

Remove conversation-specific LMS protocol and implementation after Clyde imports the shared library directly and joint codebase validation passes. The [coordination plan](https://github.com/agoodkind/clyde/blob/docs/embedded-conversation-search/docs/superpowers/plans/2026-09-27-shared-search-coordination.md) controls the cross repository cutover. The [library design](../specs/2026-09-27-shared-search-library-design.md) defines the active data contract.

## Current behavior

[Service protocol](../../../proto/lmsemanticsearch/v1/service.proto) declares conversation RPCs. [gRPC registration](../../../internal/daemon/grpc_server.go), [conversation manager](../../../internal/daemon/manager_conversations.go), and [semantic conversation modules](../../../internal/semantic/conversation_batch.go) implement the old LMS path. Existing collections and checkpoints still exist.

## Constraints

- Begin removal after Clyde C4 direct library ingestion and search, L4 codebase adoption, and joint testbed acceptance pass. Confirm Clyde's pinned LMS revision builds after the removal change.
- Remove conversation-only protocol, implementation, configuration, tests, and active documentation. Keep generic codebase behavior and the shared library.
- Preserve historical Milvus collections, local vector files, registry records, checkpoints, and source data.
- Regenerate protobuf bindings with `make proto`; never hand-edit generated files.

## Tasks

### 1. Remove seven conversation RPCs and handlers

Files:

- Modify [service.proto](../../../proto/lmsemanticsearch/v1/service.proto) and [grpc_server.go](../../../internal/daemon/grpc_server.go).
- Regenerate [service.pb.go](../../../gen/go/lmsemanticsearch/v1/service.pb.go) and [service_grpc.pb.go](../../../gen/go/lmsemanticsearch/v1/service_grpc.pb.go).
- Delete or reduce [grpc_server_conversation_backfill.go](../../../internal/daemon/grpc_server_conversation_backfill.go) and [grpc_server_conversation_stream.go](../../../internal/daemon/grpc_server_conversation_stream.go).

Steps:

1. Search the pinned Clyde build and LMS callers for `RegisterConversationCollection`, `SyncConversationManifest`, `UpsertConversationDocumentsStream`, `BackfillConversationScalars`, `DeleteConversation`, `SearchConversations`, and `SearchWithinConversation`. Confirm that Clyde uses direct library methods for each supported operation.
2. Remove these RPC declarations, request and response messages used only by them, gRPC registrations, handlers, and retired validation helpers. Keep code used by remaining codebase RPCs.
3. Run `make proto`. Confirm that the regenerated service exposes no retired conversation method and still compiles every remaining service method.

### 2. Remove conversation-only storage and configuration

Files:

- Delete or reduce [manager_conversations.go](../../../internal/daemon/manager_conversations.go), [manager_conversation_text.go](../../../internal/daemon/manager_conversation_text.go), [manager_conversation_storable.go](../../../internal/daemon/manager_conversation_storable.go), [manager_conversation_tools.go](../../../internal/daemon/manager_conversation_tools.go), and [conversation_search_filter.go](../../../internal/daemon/conversation_search_filter.go).
- Delete or reduce [conversation_batch.go](../../../internal/semantic/conversation_batch.go), [conversation_filter_expr.go](../../../internal/semantic/conversation_filter_expr.go), [conversation_backfill.go](../../../internal/semantic/conversation_backfill.go), [conversation_state.go](../../../internal/semantic/conversation_state.go), and [conversation_columns.go](../../../internal/semantic/conversation_columns.go).
- Inspect [item_source.go](../../../internal/daemon/item_source.go), [collection.go](../../../internal/semantic/collection.go), [conversation.go](../../../internal/localvec/conversation.go), and [types.go](../../../internal/model/types.go) for remaining conversation-only wrappers. Remove them after proving no codebase caller depends on them.
- Modify [conversation ingest documentation](../../conversationingest/overview.md) and remove `MaxConversationsPerIngest`, `defaultMaxConversationsPerIngest`, and `CLAUDE_CONTEXT_MAX_CONVERSATIONS_PER_INGEST` from [config.go](../../../internal/config/config.go) after the last conversation RPC caller is removed.

Steps:

1. Enumerate references to every conversation symbol and configuration key. Remove code without a remaining codebase or library caller. Delete tests that assert retired wire framing or deleted conversion behavior.
2. Preserve codebase search and offline storage operations. Rewrite a public test only when it protects behavior that remains supported.
3. Replace active conversation service documentation with the direct library boundary. Keep historical migration records intact.
4. Inspect old collection and checkpoint identifiers through read-only testbed queries before and after removal. Do not execute collection drop or automatic cleanup.

### 3. Verify the remaining public surface

Files:

- Create `test/live/library_retirement_live_test.go` for the remaining LMS public boundary.

Steps:

1. Run codebase ingest and search against isolated real Milvus and the production embedding adapter. Assert complete pages and replacement behavior after the conversation server code is absent.
2. Build Clyde against the removal revision and run its direct library ingestion and search acceptance in the same testbed. Record source counts, complete page results, and restart recovery.
3. Verify that historical collections and checkpoints retain their recorded identifiers and counts. Treat a mismatch as a stop condition.

## Verification

- Run `make proto`, `make library-live-l5`, `make test`, and `make check` in LMS. The strict lane target rejects skips and zero matching tests.
- Run the coordinator's Clyde build and testbed acceptance against the pinned LMS revision. The LMS build alone cannot establish cross repository acceptance.
