# Publish canonical vectors and source occurrences

## Goal

Implement the [library storage contract](../specs/2026-09-27-shared-search-library-design.md) with durable publication, independent source occurrences, and exact backend verification. L0 public contracts and strict targets must exist first.

## Current behavior

Vector reuse identifies reusable content. Staging writes a vector for each source row. The current live harness skips unavailable Milvus and uses synthetic embeddings; its helpers cannot establish this lane's acceptance.

## Constraints

- L1 owns the common catalog schema, writer lock, vector outbox, ingestion, recovery, backend adapters, and strict live harness. L2 owns `library/lexical_schema.go`; L1 registers that migration only after L2 integration.
- One canonical SQLite WAL catalog and lock path bind a vector pool. The actual Milvus vector collection records the catalog UUID in its initial schema description. A different catalog cannot write to that pool.
- Persist exact vector bytes before the first backend RPC. Verify identity and checksum with a strong read before occurrence publication. Never claim exactly once physical backend bytes.
- Preserve old code collections. Do not run automatic vector deletion during routine Replace, Delete, or append retention.

## Tasks

### 1. Open the catalog and bind the backend

Files:

- Create `library/catalog_schema.go`, `library/catalog.go`, `library/catalog_lock.go`, and `library/vector_store.go`.
- Create `library/open.go` with `Open`, `Close`, and the concrete storage methods from the design.
- Create `library/milvus/store.go` and `library/embedded/store.go`.

Steps:

1. Create the design's catalog UUID, namespaces, typed scalar declarations, occurrences, source blobs, owner generations, staged rows, vector identities, outbox, receipts, projection events, and snapshot reference tables. Use indexed typed scalar tables for equality, range, and literal prefix predicates.
2. Open SQLite in WAL mode at the canonical path. Acquire a kernel cross-process writer lock at the descriptor lock path for all publication and recovery. Reject descriptor or schema version mismatches before writing.
3. Implement `VectorStore.PoolIdentity`, `BindCatalog`, `PutCanonical`, `VerifyStrong`, and `ScoreExact`. The Milvus adapter creates the actual vector collection with UUID and descriptor in the initial schema description, then describes and checks the winner after success, already-exists, or ambiguous timeout. The embedded adapter atomically binds its pool identity and scores every selected vector exactly.
4. Test two processes opening one pool with different catalog UUIDs. Assert that exactly one descriptor wins and the losing process writes no vector or occurrence.

### 2. Publish complete owner generations

Files:

- Create `library/vector_outbox.go`, `library/ingest.go`, and `library/recovery.go`.
- Create `test/live/library_write_live_test.go` and `test/live/library_harness.go`.

Steps:

1. Compute vector identity from exact transformed document input, resolved model and revision, dimensions, normalization, and role prefix. Compare full identity bytes before reuse. Validate finite vectors with nonzero norm.
2. Persist embedding bytes and checksum in the outbox before `PutCanonical`. Retry only those exact bytes after ambiguous RPC failure. Verify the backend's primary key, identity digest, and checksum with a strong read.
3. Implement `Stage` with bounded physical batches. `CommitGeneration` validates `GenerationSeal` row count and sorted manifest digest before one SQLite transaction publishes the complete owner. Queries read the prior owner generation until publication succeeds. `AbortGeneration` releases uncommitted stage rows.
4. Implement `Apply` append and replace policy checks, durable batch token and hash receipts, `Delete` for ReplaceAllowed namespaces, `GetOwnerState`, and `ReprojectScalars` for declared mutable columns only. Reprojection writes an event and effective metadata, never vectors or immutable source text.
5. Replay outbox and publication state after process interruption at every durable boundary. Return an original committed receipt for identical replay even after later owner generations. Reject an unknown stale generation and conflicting token reuse.
6. Use isolated real Milvus plus a real production embedding adapter in the new harness. Fail unavailable dependencies. Test duplicate content across namespaces, changed source metadata with reused vector, shrinking file generation, incomplete seal, append policy violation, ambiguous upsert, process death, and receipt replay.

## Verification

- Run `make library-live-l1`. The strict L0 gate must report selected tests that ran and passed without skips.
- Run `make test && make check` and `make offline-live`.
- Query backend vectors and catalog occurrences independently. Assert one live canonical vector for each identity, no published occurrence referencing an unverified vector, and intact previous generations after failures.
- Measure post-compaction backend storage separately from logical vector count. Record any physical duplication rather than interpreting an upsert acknowledgement as a storage guarantee.
