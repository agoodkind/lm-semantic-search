# Conversation ingest

The daemon indexes clyde's conversation corpus as a document-kind codebase with canonical path `chat:///<collection id>` and one Milvus collection per corpus. This page describes the ingest pipeline as it works today and links each behavior to the code and test that own it.

## Manifest contract

clyde sends the full manifest of per-conversation fingerprints every pass, the engine replies with the conversation ids it needs, and clyde streams documents only for those ids (`proto/lmsemanticsearch/v1/service.proto`, `SyncConversationManifest` and `UpsertConversationDocumentsStream`).

The needed reply is capped. `capNeededConversations` (`internal/daemon/manager_conversations.go`) names at most `Config.MaxConversationsPerIngest` conversations per reply (default 100, zero disables). Modified conversations always fit before added ones, and the added window rotates behind a per-codebase cursor so no id starves. `TestSyncConversationManifestCapsNeededReply` and `TestSyncConversationManifestRotatesModifiedOverflow` (`internal/daemon/manager_conversations_test.go`) pin the cap and the rotation, and the cap bounds every ingest job regardless of backlog size.

## Message-level delta

A delivered conversation touches only its changed messages. `conversationItemSource.indexOne` (`internal/daemon/item_source.go`) loads the conversation's stored rows from the live collection through `LoadConversationMessageState` (`internal/semantic/conversation_state.go`) and diffs the delivered documents against them with `diffConversationMessages` (`internal/daemon/manager_conversations.go`), whose doc comment states the comparison rule. Unchanged messages contribute nothing, new messages insert, edited messages replace exactly their own rows, and stale stored messages are deleted. Removal targets a message by its exact path plus its slash-suffixed prefix so sibling indices can never match (`TestConversationIndexOneSiblingIndexSafety`).

The delta rides the generic per-item override fields on `indexer.OneFileResult` (`internal/indexer/indexer.go`), honored by `handleChangedFile` (`internal/daemon/manager_delta.go`). A fully unchanged delivery advances the checkpoint with zero Milvus writes. `TestConversationIngestWritesOnlyMessageDeltas` walks the whole lifecycle (cold, unchanged, appended, edited, stale). When the row load fails, `indexOne` falls back to whole-conversation reindexing with per-item prefix reuse (`TestConversationIndexOneStateLoadFailureFallsBackToFullReindex`).

The merkle checkpoint stays at conversation granularity: one entry per conversation id holding the manifest fingerprint, which keeps the checkpoint aligned one-to-one with clyde's wire manifest.

## Seed and reuse invariants

`resolveSeed` (`internal/daemon/seed.go`) is the only seed decision, and its doc comment states the invariant that a seed may only cause skips for rows provably present in the collection the build writes to. `resolveItemReusePolicy` (`internal/daemon/reuse_policy.go`) is the only reuse decision: a build whose live collection exists reuses its vectors regardless of staging target or codebase kind, and only a forced job opts out. `TestRunBootstrapReusesLiveCollectionVectors` and `TestRunBootstrapMissingLiveCollectionEmbedsEverything` (`internal/daemon/manager_bootstrap_test.go`) pin both sides, so a bootstrap over an already-populated corpus reuses instead of re-embedding it.

## Collection evidence and bootstrap reasons

Routing decisions judge the stored collection name, never a re-derived one. `probeCollectionEvidence` (`internal/daemon/collection_evidence.go`) returns presence plus row-count evidence, and `decideEmptyDiffMode` (`internal/daemon/collection_policy.go`) sends an empty-diff job to bootstrap only on definitive evidence. `TestConversationEmptyDiffStoredNamePresentCompletesNoop` (`internal/daemon/manager_conversations_test.go`) pins the incident regression where a populated collection was treated as missing.

Every job that routes to bootstrap records why. `routeToBootstrap` (`internal/daemon/manager_jobs_state.go`) owns the reason vocabulary and stamps `Progress.BootstrapReason`, which persists in the job ledger, so an unexpected full rebuild is diagnosable from the job record alone.
