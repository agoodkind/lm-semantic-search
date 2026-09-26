# Generic collection ingestion implementation plan

## Goal

Complete LMS-15, LMS-16, and LMS-17. A client registers a document collection, syncs item fingerprints, streams client-keyed rows, backfills scalars, and deletes one item. Existing conversation RPCs use the generic implementation and preserve stored rows and checkpoints.

## Current behavior

`proto/lmsemanticsearch/v1/service.proto` declares conversation-specific registration, manifest, upsert, backfill, and delete RPCs. `internal/daemon/manager_conversations.go` registers document codebases, diffs Merkle checkpoints, caps needed IDs with `capNeededConversations`, and queues jobs. `internal/daemon/item_source.go` invokes the existing delta indexer. `internal/semantic/collection.go` creates conversation scalars. `internal/semantic/conversation_backfill.go` rewrites scalars while preserving vectors. Registration currently returns an existing record without validating the live Milvus schema.

## Constraints

- Reuse the existing delta indexer, embedder, 60000-byte UTF-8 splitter, vector reuse, job coalescing, and checkpoint store. Extend their contracts instead of implementing a parallel ingestion pipeline.
- Map the client `row_key` to the existing `relativePath` column. Preserve deterministic part suffixes and primary-key derivation for split rows.
- Include `item_id_column` in registration. The conversation adapter sets it to `conversationId`, so generic item operations can select existing rows without a migration. The engine populates that scalar from `item_id` and rejects conflicting row values.
- Preserve the exact `conversationScalarFields()` schema, including Milvus string lengths. Preserve the existing `fingerprintConversationDocuments` bytes when the old RPC derives an omitted manifest.
- Default reconcile mode to retain. Require an explicit full manifest for authoritative mode. Keep old RPCs available through both Clyde cutovers.
- Apply the daemon's existing maintenance refusal to generic backfill and delete before loading a collection or queuing a job. Preserve the old handlers' maintenance error when they delegate.

## Pull request boundaries

Implement LMS-15, LMS-16, and LMS-17 as separate pull requests in that order. Each pull request includes its proto source, regenerated Go code, implementation, and public-boundary tests. Keep the ingest parity test and its documentation with LMS-16. Keep backfill and delete parity coverage with LMS-17. Each merged unit must compile and serve its complete RPCs while the old conversation RPCs remain available. Run the corresponding deployed parity gate before Clyde uses that unit.

## Tasks

### 1. Register declared schemas (LMS-15)

Files:

- Modify: `proto/lmsemanticsearch/v1/service.proto`
- Modify: `internal/model/types.go`
- Modify: `internal/daemon/manager_conversations.go`
- Modify: `internal/daemon/grpc_server.go`
- Modify: `internal/semantic/collection.go`
- Modify: `internal/semantic/conversation_columns.go`
- Modify: `internal/adapterr/respond.go`
- Create: `internal/daemon/manager_collections.go`
- Create: `internal/daemon/grpc_server_collections.go`
- Create: `internal/daemon/grpc_server_collections_test.go`

Behavior:

- `RegisterCollection` accepts collection ID, `item_id_column`, and scalar declarations with a column identifier, type, nullability, and string length when applicable. Support string, bool, and int64. Reject duplicate, reserved, unsupported, or missing item-ID columns.
- Persist the accepted declaration on `model.Codebase`. Validate repeat registration against the saved declaration in both storage profiles. Describe and compare the Milvus schema when its physical collection exists; the offline local-vector profile has no Milvus collection to inspect. A mismatch returns a gRPC `ErrorInfo` reason `collection_schema_mismatch` and `column` metadata. Never rebuild an existing collection to satisfy a conflicting declaration.
- `RegisterConversationCollection` passes the declaration produced by `conversationScalarFields()` and `conversationId` to the generic registration path. For an old document record without a saved declaration, validate an existing Milvus collection before adoption. In the offline profile, compare the legacy declaration with the saved record and local rows without requiring Milvus. Registration leaves the Merkle checkpoint untouched.

Steps:

1. Add the proto contract and typed codebase declaration. Keep old registry JSON readable. Run `make proto` to regenerate `gen/go/lmsemanticsearch/v1/`.
2. Parameterize collection creation and schema validation by the declaration. Run the existing conversation scalar migration before validating an existing Milvus collection. Validate saved declarations without Milvus in the offline profile.
3. Implement `Manager.RegisterCollection`, including persistence rollback. Make old manager and gRPC registration methods delegate to it. Attach the stable reason and conflicting column through `adapterr.RespondGRPC`; `adapterr.Respond` alone does not attach `ErrorInfo`.
4. Add a public gRPC test that registers, restarts, re-registers, and checks the saved declaration and checkpoint in both storage profiles. Test a conflicting scalar through gRPC status details and an existing conversation collection with the old RPC.

Verification:

- Run: `make proto && go test ./internal/daemon ./internal/semantic ./internal/adapterr`
- Expect: repeated registration preserves codebase ID and checkpoint; a schema conflict identifies its column without string matching.

### 2. Sync manifests and ingest generic rows (LMS-16)

Files:

- Modify: `proto/lmsemanticsearch/v1/service.proto`
- Modify: `internal/model/types.go`
- Modify: `internal/daemon/item_source.go`
- Modify: `internal/daemon/manager_conversations.go`
- Modify: `internal/daemon/grpc_server_conversation_stream.go`
- Modify: `internal/semantic/collection.go`
- Modify: `internal/semantic/conversation_batch.go`
- Modify: `internal/semantic/insert_batch.go`
- Modify: `internal/semantic/removal.go`
- Modify: `internal/localvec/conversation.go`
- Modify: `internal/localvec/format.go`
- Create: `internal/daemon/collection_item_source.go`
- Create: `internal/daemon/grpc_server_collection_stream.go`
- Create: `internal/daemon/grpc_server_collection_stream_test.go`

Behavior:

- `SyncCollectionManifest` accepts item ID and fingerprint pairs. It uses the existing checkpoint, per-ingest cap, and rotation cursor. Its needed-ID ordering matches `capNeededConversations`.
- `UpsertCollectionItemsStream` accepts a header, bounded row frames, and manifest frames. A row supplies `row_key`, `item_id`, text, and typed declared scalar values. Validate the stream order, byte cap, declared columns, and value types. Preserve retain, authoritative, backfill_delivered, and force_reexamine semantics.
- The generic item source invokes the existing delta indexer. Select and replace rows by the declared item-ID scalar. Preserve text splitting, vector reuse, coalescing, and retain-on-absence. Select legacy conversation rows with null `conversationId` through the existing relative-path fallback during compatibility.
- The old conversation stream derives text, tool, and thinking rows with the existing conversion functions, then calls the generic manager path. Its missing-manifest fallback still uses `fingerprintConversationDocuments`.

Steps:

1. Add typed scalar values, item fingerprints, row frames, header flags, and manifest frames to the proto. Regenerate with `make proto`.
2. Generalize the manifest diff and item-source callbacks in `manager_conversations.go` and `item_source.go`. Resolve scalar declarations from the registered codebase. Select generic replacement and vector reuse by declared item ID; retain `conv/`, `convtool/`, and `convthink/` path-prefix handling only for legacy conversation rows without `conversationId`.
3. Carry typed declared scalar values through stored chunks, splitting, Milvus column insertion, local-vector serialization, and readback. Update both stores' item removal and vector-reuse selection. Keep existing physical row keys, vectors, checkpoint writes, and legacy row serialization.
4. Adapt `SyncConversationManifest` and `UpsertConversationDocumentsStream` to the generic manager calls. Preserve their wire contracts.
5. Add a public gRPC test with a temporary local-vector store. Use the live harness for a real temporary Milvus database and deterministic fake embedding server. Submit the same realistic transcript through both RPCs into isolated collections. Compare row keys, text, scalars, vectors, checkpoint fingerprints, and the next needed set. Cover retain, authoritative, backfill, force, long text, invalid scalars, and stream limits.

Verification:

- Run: `make proto && go test ./internal/daemon ./internal/semantic ./internal/localvec`
- Expect: old and generic requests store equal rows and checkpoints; an unchanged second manifest requires no items.

### 3. Backfill scalars and delete items (LMS-17)

Files:

- Modify: `proto/lmsemanticsearch/v1/service.proto`
- Modify: `internal/daemon/manager_conversations.go`
- Modify: `internal/daemon/grpc_server_conversation_backfill.go`
- Modify: `internal/semantic/conversation_backfill.go`
- Modify: `internal/semantic/conversations.go`
- Modify: `internal/localvec/conversation.go`
- Create: `internal/daemon/grpc_server_collection_maintenance.go`
- Create: `internal/daemon/grpc_server_collection_maintenance_test.go`

Behavior:

- `BackfillCollectionScalars` streams item IDs with declared scalar updates. Dry run reports changed and orphan counts without writes. Execution changes only null or empty targets and preserves vectors, text, row keys, and checkpoints.
- `DeleteCollectionItem` queues a job that removes one item's rows by the declared item-ID scalar. Explicit deletion leaves its Merkle checkpoint unchanged, matching `runConversationDelete`; a later manifest sync converges the checkpoint. Use the legacy relative-path fallback only for old conversation rows without `conversationId`.
- `BackfillConversationScalars` and `DeleteConversation` delegate to these generic operations. Their responses and counts remain equal.

Steps:

1. Add maintenance RPCs and regenerate with `make proto`.
2. Parameterize the existing vector-preserving backfill and delete by the declared item-ID column. Retain null-or-empty selection and changed/orphan accounting.
3. Adapt the old handlers. Add public gRPC tests comparing dry-run counts on real stored rows, byte-for-byte vectors after execution, and unaffected rows after single-item deletion.

Verification:

- Run: `make proto && go test ./internal/daemon ./internal/semantic ./internal/localvec`
- Expect: dry run writes nothing; old and generic counts agree; delete affects only one item.

### 4. Release with Clyde unchanged

Files:

- Modify: `docs/conversationingest/overview.md`
- Create: `test/live/generic_collection_ingest_live_test.go`

Behavior:

- The live test uses separate temporary collections and a bounded real transcript sample. It compares both RPC paths without printing transcript content or modifying provider artifacts.
- A deployed LMS release serves both RPC surfaces. Clyde still calls the old surface, and its unchanged manifest produces no re-offer.

Steps:

1. Document the generic RPC contract and conversation adapters in the existing ingest documentation.
2. Add the parity test using `test/live/harness.go`. Deploy LMS, then inspect the Clyde feeder log for an unchanged normal pass before CLYDE-629.

Verification:

- Run: `go test -tags live -run '^TestGenericCollectionIngestParity$' -count=1 ./test/live/`
- Expect: equal stored rows and checkpoint fingerprints in isolated collections.
- Run: `make test && make check`
- Expect: repository tests and lint pass. The deployed feeder reports zero needed items for an unchanged manifest.
