# Adopt shared storage for codebase search

## Goal

Use the [shared library](../specs/2026-09-27-shared-search-library-design.md) for new codebase ingestion and search while retaining codebase replacement, deletion, and the offline profile. L1 and L3 must pass their library gates. The [coordination plan](https://github.com/agoodkind/clyde/blob/docs/embedded-conversation-search/docs/superpowers/plans/2026-09-27-shared-search-coordination.md) controls release timing and joint Clyde validation.

## Current behavior

Delta indexing processes changed files. Semantic staging writes source rows. Codebase search calls the existing semantic service. The offline acceptance suite runs an embedded vector store and real ONNX model.

## Constraints

- Preserve source extraction and language selection outside the library. Use `PrepareText` for generic model limits after extracting source text.
- Stage every chunk of one file owner and commit only after its complete row count and manifest digest validate. A truncated stream cannot remove previously indexed chunks.
- New code writes and reads use the generic layout after acceptance. Do not clear, rewrite, or silently retire existing collections. The old local offline profile must work through an exact embedded vector adapter.
- Update only durable instructions that conflict with the approved new layout. Historical collections retain their schema and data.

## Tasks

### 1. Map code rows to occurrences

Files:

- Create `internal/daemon/library_code_adapter.go` and `test/live/library_codebase_live_test.go`.
- Modify [manager_delta.go](../../../internal/daemon/manager_delta.go), [staging.go](../../../internal/semantic/staging.go), and [service.go](../../../internal/semantic/service.go).

Steps:

1. Map each file owner to a ReplaceAllowed namespace. Preserve extracted source, code path, language, and current chunk identity as occurrence fields and typed scalars. Supply separate `SourceText`, lexical `SearchText`, and exact transformed `EmbeddingInput`.
2. Use `PrepareText` on extracted input. Stage bounded batches for one owner, compute the complete sorted row manifest, and call `CommitGeneration` once. Call `Delete` for an explicit codebase retention deletion.
3. Recover a process interruption before and after backend vector publication. Verify the old owner generation remains searchable until the complete new generation commits.
4. Test one file that grows, one that shrinks from three parts to two, one deletion, and repeated identical content in separate files. Assert codebase owner counts, source excerpts, vector reuse, and unchanged files.

### 2. Switch code search and retain offline acceptance

Files:

- Modify [manager_search.go](../../../internal/daemon/manager_search.go), [semantic_backend.go](../../../internal/daemon/semantic_backend.go), [config.go](../../../internal/config/config.go), and [profile.go](../../../internal/config/profile.go) for backend selection and dense or hybrid mode mapping.
- Extend `test/live/library_codebase_live_test.go` and create `test/offlinelive/library_codebase_offline_live_test.go`.

Steps:

1. Map current dense and hybrid modes to `Config.SearchMode`. Map filters, score floor, group cap, and result fields to `SearchRequest`. Return the library's complete pages without applying an additional fixed candidate cutoff.
2. Open the embedded exact vector adapter for the offline profile. Exercise the same public codebase search behavior with the real ONNX model.
3. Build a new catalog in an isolated testbed. Compare source owner counts, ordered hits, and complete pages with a matched healthy recovered codebase baseline before switching reads. Preserve the prior catalog for recovery.
4. Run Clyde's C4 integration and L4 validation together before L5 removes the conversation service path. The coordinator records the joint gate.

### 3. Update layout instructions at cutover

Files:

- Modify [AGENTS.md](../../../AGENTS.md) and [README.md](../../../README.md). [CLAUDE.md](../../../CLAUDE.md) references AGENTS.md and needs no separate edit.

Steps:

1. Update the TypeScript compatibility, switch-back, and incremental sync statements in AGENTS.md to distinguish historical collections from new shared pools.
2. Update README.md codebase index layout instructions for the new pool and intact historical collections.
3. Read both edited documents in full and verify each path and active behavior against the implementation before commit.

## Verification

- Run `make library-live-l4` and `make library-live-l4-offline`. The strict runner rejects skips and zero matched tests.
- Run `make offline-live`, `make test`, and `make check`.
- Run the coordinated Clyde integration gate before L5. Verify changed files, deleted files, offline search, complete pages, and intact old collections on an isolated testbed.
