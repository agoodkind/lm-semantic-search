# Publish shared search contracts and strict acceptance commands

## Goal

Export the [library contract](../specs/2026-09-27-shared-search-library-design.md) for both applications. This lane establishes compilable public types, generic input preparation, production embedding adapters, and strict acceptance commands. The [coordination plan](https://github.com/agoodkind/clyde/blob/docs/embedded-conversation-search/docs/superpowers/plans/2026-09-27-shared-search-coordination.md) controls integration.

## Current behavior

The production embedding provider and oversize splitter are internal. `GO_MK_PREREQS` prepares generated grammars, the pinned gksyntax workspace, and native CGO dependencies. The current `live` target permits tests to skip when Milvus is unavailable. Its harness uses a synthetic embedding server.

## Constraints

- L0 owns [Makefile](../../../Makefile), the acceptance runner, and public contract files. Later lanes add separate tests and adapters without editing the common target definitions concurrently.
- Public types can compile before storage and query implementations exist. Do not publish successful no-op methods.
- The caller owns its Milvus client. The library owns its SQLite connection, writer lock, and query snapshots.
- A published LMS module must import in Clyde with a pinned gksyntax submodule workspace and CGO. A local `replace` cannot stand in for that import check.
- Live acceptance uses a real production embedding adapter and isolated real Milvus. Missing dependencies, skipped tests, and zero matching tests fail.

## Tasks

### 1. Define the public package

Files:

- Create `library/types.go`, `library/config.go`, and `library/errors.go`. L1 creates `library/open.go` when `Open` can construct a working store.
- Create `library/config_test.go` for public validation errors.

Steps:

1. Declare `StoreDescriptor`, `Config`, `Embedder`, `VectorStore`, `NamespaceSpec`, `ScalarColumn`, `ScalarValue`, `Filter`, `OccurrenceID`, `Occurrence`, `Batch`, `GenerationKey`, `GenerationSeal`, `SearchRequest`, `SearchHit`, `SearchPage`, `ApplyReceipt`, and `OwnerState` exactly as the design specifies.
2. Define public data types and `Embedder` and `VectorStore` interfaces. Keep the `Library` methods in the design until L1 creates the concrete implementation and L3 adds `Search`. Do not add fake methods or a successful placeholder `Open`.
3. Validate descriptor path identity, backend identity, immutable model revision, dimensions, normalization, namespace policy, typed columns, and search configuration. Return the design's typed errors.
4. Test invalid descriptors, unknown scalar columns, and valid `BM25B=0` through public validation. L1 verifies caller backend ownership with a real adapter.

Verification:

- Run `make test && make check` after the package compiles. Both commands must pass.

### 2. Export generic preparation and production embedding

Files:

- Create `library/prepare.go`, `library/embedding/adapter.go`, `library/embedding/onnx/onnx.go`, `library/internal/embedadapter/adapter.go`, and `library/prepare_test.go`.
- Modify [embedding.go](../../../internal/embedding/embedding.go) to expose the existing OpenAI-compatible provider without a duplicate implementation. Move `onnx.go`, `onnx_cache.go`, `onnx_tokenizer.go`, and the native bridge into package `internal/embedding/onnx`, and select the configured provider in package `internal/embedding/providers`. Package `internal/embedding` then contains no cgo code.

Steps:

1. Implement `PrepareText` with stable part ordinals and exact transformed document embedding input. With a `Tokenizer`, limit each transformed input to `MaxTokens` counted tokens. Without one, limit each transformed input, document prefix included, to `int(MaxTokens * 0.9)` bytes, one byte per token; NV-EmbedCode on lmd-serve measured at most bytes plus 2 tokens in the 2026-09-29 02:48:15 to 02:51:24 UTC probe (21 requests). Keep source selection and scalar policy with each caller.
2. Export `embedding.OpenAIConfig{BaseURL, APIKey, Model, Dimension, RequestTimeout, MaxAttempts, BackoffBase}` and `embedding.NewOpenAI(context.Context, OpenAIConfig) (library.Embedder, error)` without importing `internal/config` in the public signature. Reuse the production provider and its four-attempt, 200-millisecond retry defaults. `go list -deps ./library/embedding` lists no cgo package. Export `onnx.Config{ModelName string, ModelCacheRoot string}`, `onnx.New(context.Context, Config) (library.Embedder, error)`, and `onnx.NewTokenizer` from `library/embedding/onnx` using the current preset resolver and cached model assets. Adapt `BatchResult{Vectors,Skipped}` to the public `Embedder` contract. Reject skipped or missing input vectors with typed paused, busy, rejected, or cancellation errors; never publish an incomplete batch.
3. Return prepared parts without writing occurrences. `Stage` accepts those parts and never splits again.
4. Test input just below, at, and above the model limit with the real tokenizer. Assert complete text coverage, stable part IDs, no truncation, and separate document and query transforms. Exercise a real provider rejection and assert that no prepared part is silently omitted.

Verification:

- Run `make test && make check`. The preparation test must call the exported function.

### 3. Add strict lane acceptance targets

Files:

- Modify [Makefile](../../../Makefile).
- Create `cmd/library-live-gate/main.go` and `cmd/library-live-gate/main_test.go`.

Steps:

1. Add `library-live-prereqs` with the same `GO_MK_PREREQS` as `live`. Add `library-live-l1` through `library-live-l5` as explicit targets. Each test target depends on `library-live-prereqs` and invokes the gate with a lane-specific test prefix. Define L1 through L4 under the `live` tag and L5 under the `live` tag after protocol removal.
2. Make the gate execute `go test -json -count=1 -tags live -run '^TestLibraryWrite' ./test/live/` for L1. Use `^TestLibraryLexical`, `^TestLibrarySearch`, `^TestLibraryCodebase`, and `^TestLibraryRetirement` for L2 through L5. Decode test events and fail when the command fails, zero tests run, any selected test skips, or no selected test passes. Preserve subprocess stderr and exit status. Do not interpret a package-only `pass` event as a selected test.
3. Define `library-live-l4-offline` through the same gate with `-tags offlinelive -run '^TestLibraryCodebaseOffline' ./test/offlinelive/`. The L4 gate must run both targets. Existing `make offline-live` remains an additional regression check.
4. Test the gate with a real temporary Go test package containing pass, fail, skip, and no-match cases. The gate test must assert the process exit status and useful diagnostics for each case.

Verification:

- Run `make test && make check`. Do not invoke a lane target until its lane tests exist. `make library-live-l1` through `make library-live-l5` are acceptance commands only after the corresponding implementation and tests are present.

### 4. Prove an external module import

Files:

- Modify [go.mod](../../../go.mod) only if an actual direct dependency changes.
- Create `test/import/go.mod`, `test/import/library_test.go`, and `test/import/README.md` as a separate import fixture.

Steps:

1. Build LMS with the repository's generated gksyntax workspace and CGO enabled.
2. In `test/import/go.mod`, declare a separate module. In `test/import/library_test.go`, import `goodkind.io/lm-semantic-search/library`, call `PrepareText` with a real tokenizer, and assert its returned part spans and transformed inputs. Add no local LMS `replace`.
3. Resolve the published commit with `GOWORK=off go list -m -json goodkind.io/lm-semantic-search@<published-commit>` and save its canonical `Version`. Verify `Origin.Hash` against the full commit when the response supplies it. In the fixture directory, run `GOWORK=off go get goodkind.io/lm-semantic-search@<published-commit>`. Initialize the matching pinned gksyntax checkout and generate its grammar. Run `go work init <absolute fixture directory> <absolute pinned gksyntax checkout>`. Never include an LMS checkout in this workspace. Run `go list -m -json goodkind.io/lm-semantic-search` and require the saved `Version` and an absent `Replace`. Record exact revision IDs and CGO environment values observed on macOS and Linux.

Verification:

- Run `make test && make check` in LMS. Run `go test ./...` inside the external import fixture workspace after verifying the published LMS version. A module ZIP missing grammar C sources cannot be reported as a passing standalone `GOWORK=off` import.
