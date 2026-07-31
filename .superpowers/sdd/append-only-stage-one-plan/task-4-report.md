# Task 4 report

## Result

LMS-5 separates content identity from embedding model identity.

New Milvus rows store indexed `contentHash` and nullable `embeddingModel` fields. The content hash is the normalized exact content only. Provider, endpoint, dimension, and inactive offline model settings do not affect it.

Reuse checks the current-dimension state-root catalog, then every registered current or legacy collection. Untagged rows remain valid. Only two known unequal model names block reuse. Lookup never updates an existing row.

Each dimension-scoped catalog uses a separate model-qualified `catalogKey` primary key. This permits one indexed content hash to retain vectors for multiple known models at that dimension.

## Verified evidence

- Unit red: changing embedding configuration changed the old storage key. The content-only key test failed, then passed after the change.
- Unit red: the insert round trip lacked `contentHash` and `embeddingModel`. It failed, then passed after schema and insert changes.
- Live red: restricting lookup to the target collection produced reused/embedded `0/1`, not `1/0`.
- Live red: removing the model from `catalogKey` retained only one of two known models.
- Live red: forced conversation reconciliation called whole-store reuse. The new unit test failed before the no-reuse-scope gate.
- Live green: `TestUntaggedReuseAcrossCorpusPreservesSourceRow` reused an untagged vector across collections with zero embedding.
- Live green: real semantic search returned the untagged source row before reuse.
- Live green: the source row kept the same ID, content, absent identity, and vector checksum.
- Live green: the new occurrence stored its own ID, content hash, current model, and the reused vector.
- Live green: repeated delivery reported zero modified files, embedded files, reused chunks, and embedded chunks. Its row snapshot stayed identical.
- Live green: an absent stored model was accepted under a different current model. A known unequal stored model was rejected.
- Live green: `TestReuseCatalogStoresEachKnownEmbeddingModel` stored and selected two model rows for one content hash. A third known model received no reuse.
- Live green: the duplicate control reused one vector while the unique control embedded once.
- Live green: explicit force embedded selected rows and reused zero after the no-reuse-scope gate.
- Live snapshot: legacy ID `legacy-247c2730bfc4fc9f9686afba069ba5af` and new ID `chunk_f48ade17ff2e9a6c` both had vector SHA-256 `ae5140b4f95bf59256e1f82bc650f9391c3b06f5b7adf63bd92497a9c7c5bfc2`.
- Performance: lookup p95 was 21.764 ms. Configured embedding p95 was 535.455 ms. Lookup was 4.06 percent of embedding.
- `go clean -testcache && make test`: passed.
- `make check`: passed all five gates.
- `make live`: passed the isolated real-Milvus suite in 92.537 seconds.
- Production daemon and production collections were untouched. Background feeders stayed disabled in the live harness.

## Review round 1

- Real Milvus red: a state-root catalog seeded at dimension 1536 rejected the next 4096 insert with `vector dim 4096 not match collection definition, which has dim of 1536`. The target staging collection did not exist.
- Unit red: dimensions 3 and 1536 resolved to the same catalog name.
- Fix: catalog collection names now include the configured embedding dimension. Indexed `contentHash`, nullable `embeddingModel`, and model-qualified `catalogKey` remain unchanged.
- Real Milvus green: `TestDimensionChangeIndexesFreshVector` indexed identical old content plus one new control at dimension 4096. Progress reported embedded/reused `2/0`, and the stored shared-content vector had dimension 4096.
- Fresh `make test`: passed.
- `make check`: passed all five gates.
- Full `make live`: passed in 100.160 seconds.
- Enabled p95 gate: lookup p95 was 10.198 ms. Configured embedding p95 was 624.213 ms. Lookup was 1.63 percent of embedding.
- Production remained untouched. The live harness kept background feeders disabled.

## Inferred

- Later reuse normally hits the current-dimension state-root catalog or an indexed content hash. Untagged legacy fallback remains an exact-content query.
- SHA-256 collision resistance makes the catalog hash an effective exact-content locator. Legacy queries still compare exact stored text.

## Assumed

- The state-root registry contains every collection that belongs to that store. The lookup also includes each registered legacy collection name.

## Deviations

- The implementation also changed insert request size accounting because the two new scalar values cross the same transport boundary.
- The implementation added a daemon no-reuse-scope gate because full live verification exposed catalog reuse during explicit force.
- No production deployment or feeder run occurred.

## Self-attack

1. Whole-store lookup could silently remain collection-local. A targeted regression produced a live `0/1` failure, and the restored registry enumeration produced `1/0`.
2. One catalog primary key could strand later models. A targeted content-only primary key retained one model, while the model-qualified key retained and selected both.
3. Reuse could mutate or stamp the source row. The live test compares ID, content, relative path, nullable identity markers, and vector checksum before and after cross-collection reuse.

## Verification limits

- The performance sample used 20 candidates on the configured embedder. It is not a capacity benchmark.
