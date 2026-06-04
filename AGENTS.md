# AGENTS

Durable instructions for coding agents working in this repository.

Keep this file short, current, and focused on rules that should affect day-to-day code changes. Move long runbooks, dated audits, generated examples, and machine-specific workflows into `docs/` if they ever appear.

## Project purpose

`claude-context-go` is a ground-up Go rewrite of `zilliztech/claude-context`. The repo owns three binaries: the long-lived `claude-contextd` daemon, the `claude-context` operator CLI, and the `claude-context-mcp` stdio adapter. VS Code and Chrome extension clients are intentionally out of scope here. The Go port is independent of and not affiliated with Zilliz. The daemon's service identity is `io.goodkind.claude-contextd` (a launchd agent on macOS, a `claude-contextd.service` systemd user unit on Linux); the only compatibility requirement with the upstream TS tool is the shared Milvus data store (collection names, schema, and chunk ids), not the service identity.

## Transport contract

The daemon transport is gRPC only, with protobuf definitions managed by `buf`. The repo does not define or accept a JSON-RPC control plane.

Sources of truth:

- `proto/claudecontext/v1/service.proto`
- `buf.yaml`
- `buf.gen.yaml`

## TS upstream drop-in compatibility

The Milvus collection is the portable index, shared byte-for-byte with the upstream TS adapter. Both tools name it `hybrid_code_chunks_<md5(path)[:8]>`, use the same schema, and compute the same deterministic chunk id `chunk_<sha256(path:start:end:content)[:16]>`. Each tool keeps its own private bookkeeping outside Milvus: TS at `~/.context/mcp-codebase-snapshot.json` and `~/.context/merkle/<md5(path)>.json`; the Go daemon at `XDG_STATE_HOME/lm-semantic-search/registry.json`, fallback `~/.local/state/lm-semantic-search/registry.json`, and the matching `merkle/<codebase-id>.json` files under that same state root. The `~/.context` directory stays read-only for daemon compatibility inputs such as `.env`, `.sync-trigger`, and TS snapshot or merkle adoption.

The contract is two-way and reduces to one rule the daemon fully controls: never silently drop or rename a shared collection, and never write TS's bookkeeping files (the daemon only reads the TS merkle, to seed adoption).

Adoption (TS to Go): when `GetIndex` resolves a path whose collection exists but which has no registry entry, `adoptUnregisteredCodebase` (`internal/daemon/manager_adopt.go`) persists a first-class registry record with a stable id, seeds the Go merkle from the TS merkle via `internal/migrate/snapshot.go` (`LoadTSMerkle`) so the first sync re-embeds only changed files, starts the watcher, and enqueues one refresh sync. The collection is never touched.

Switch-back (Go to TS): a Go-written or Go-modified collection stays in the shared format and is never dropped or renamed, so TS resolves it by md5 collection name, self-recovers its own snapshot, reuses the collection non-destructively, and re-embeds only changed files. Verified against the TS source at `~/Sites/claude-context`.

Rules:

- Code that constructs collection names, the schema, or chunk ids must preserve the shared invariant; the tests in `internal/semantic/portability_test.go` lock it.
- The only collection deletion is explicit `clear_index`. There is no orphan-collection garbage collection; a collection without a registry entry is adopted, never dropped.

## Path identity

A codebase's identity is the real directory its path resolves to. `canonicalizePath` (`internal/daemon/manager_paths.go`) makes the requested path absolute and resolves it with `filepath.EvalSymlinks`, and the registry record, the merkle snapshot, and the Milvus collection name all key off that resolved real path. A symlink's own location or name may change without breaking the link, because any symlink that resolves to an indexed real directory resolves to that codebase. `get_indexing_status` appends `🔗 symlink resolved to: <real path>` when the queried path traverses a symlink.

For a codebase rooted at a symlink the shared-collection invariant does not hold: the upstream TS adapter names its collection from `path.resolve` (`packages/core/src/context.ts`), which does not follow symlinks, so the Go name (md5 of the real path) and the TS name (md5 of the symlink path) differ and the two tools do not share that collection. Non-symlinked roots resolve identically in both tools and stay shared.

## Embedding

The Go port supports exactly one embedding provider, an OpenAI-compatible HTTP adapter. `OPENAI_BASE_URL` points at any endpoint that speaks the OpenAI embeddings API.

Rules:

- Do not add provider-specific clients for VoyageAI, Gemini, or native Ollama. Anything that speaks OpenAI on a configurable base URL works without code changes.
- Do not assume an internet connection. Default config should let `claude-context-mcp` start and answer "not indexed" gracefully even when the embedding endpoint is unreachable.

## Splitter

- AST (default): `tree-sitter` parsers cover JavaScript, TypeScript, Python, Java, C, C++, Go, Rust, Scala, C#, PHP, Ruby, Bash, JSON, HTML, CSS, Kotlin, Objective-C, Dart, and Swift. Eighteen of these are pinned Go-module grammars in `go.mod`. Dart and Swift have no usable module against the pinned runtime, so each is a pinned git submodule under `internal/splitter/grammars/<language>/upstream` compiled through a hand-written cgo `binding.go` (with `grammar_parser.c` and `grammar_scanner.c` shims so the parser and scanner compile as separate translation units). The runtime `github.com/tree-sitter/go-tree-sitter` accepts grammar ABI versions 13 through 15; adding a grammar needs a `grammarForLanguage` case and an `extensionLanguages` entry, and nothing else, because the chunker is grammar-agnostic.
- The Swift submodule commits only its grammar definition, not the generated parser, so the parser is produced from the pinned submodule by the `make grammars` target, which `make build`, `make test`, and `make lint` run first. The generated parser stays inside the submodule working tree (gitignored there) and is never committed. A build host needs the `tree-sitter` CLI and initialized submodules (`git submodule update --init`).
- Chunking method (cAST): the AST path walks the parse tree and balances chunk size. A node within the budget becomes one chunk; a larger node splits into its children whose chunks are greedily merged back up to the budget; a larger node with no children is cut on language-aware separators. The budget counts non-whitespace bytes. `splitter.go` holds this walk; there is no per-language declaration list.
- Recursive separator splitter (`internal/splitter/langchain.go`): the automatic fallback when a file's language has no grammar or a parse fails, and the cut path the AST walk uses for an oversize leaf. Its tables mirror LangChain JS `RecursiveCharacterTextSplitter.fromLanguage` for the languages LangChain defines and a declaration-first chain for the rest. It is also selectable per index request via `splitter: "langchain"`, but AST is the default and produces better search results.

## Incremental sync

Per-codebase Merkle snapshots live under `XDG_STATE_HOME/lm-semantic-search/merkle/<codebase-id>.json`, fallback `~/.local/state/lm-semantic-search/merkle/<codebase-id>.json`. The sync flow lives in `internal/daemon/background_sync.go` and `internal/daemon/manager.go` (`runDeltaSync`).

On every sync request:

1. The daemon captures a new snapshot of the codebase.
2. `merkle.DiffSnapshots` computes `{Added, Modified, Removed}`.
3. An empty diff completes the job as a no-op tagged "already up to date" only when the live semantic collection is still present. If the collection is definitively missing, the daemon falls back to a full Replace instead of treating the codebase as current.
4. The indexer otherwise processes only added and modified files, and `semantic.Reindex` deletes existing Milvus rows for `relative_path in (removed + modified)` before upserting the new chunks.
5. The new snapshot is persisted only after success.

A missing previous snapshot or a missing semantic collection routes back to a full `Replace`.

## Idempotency

Concurrent MCP requests for the same codebase deduplicate against any in-flight job with a matching effective config, and that includes `force=true` requests. N parallel `index_codebase(force=true)` calls collapse to a single embedding pass instead of cancelling each other in sequence. This defensive shape prevents the machine-blowing-up failure mode where a client fan-out can otherwise launch arbitrary parallel work.

The mechanism is `Manager.dedupAgainstActiveJob` in `internal/daemon/manager.go`. Do not regress this contract when touching `StartIndex` or the job lifecycle.

## Blocking `index_codebase`

The MCP tool accepts `wait: true` and `wait_timeout_seconds` (default 300). When `wait=true`, the handler in `internal/mcpserver/server.go` polls `GetJob` every 1.5 seconds until the job reaches `completed`, `failed`, or `cancelled`, then returns the corresponding `GetIndex` response. On timeout the daemon job keeps running and the tool returns the latest progress.

Rules:

- When `wait=true`, always poll through the daemon. Do not subscribe directly to manager state.
- Concurrent waiters dedupe at the daemon's StartIndex path, so adding extra retry logic in the MCP handler is unnecessary.

## Orphan guard

The MCP adapter must exit when its parent process (Claude Code, Cursor, an editor) exits. Three independent defenses make sure that happens:

1. **stdin EOF**: the parent closes its end of the pipe, the read loop returns, and the process unwinds.
2. **PPID watcher**: `internal/mcpserver/orphan_guard.go` polls `os.Getppid()` every 2 seconds and cancels the run context when it returns `1` (reparented to init).
3. **Panic recovery**: `cmd/claude-context-mcp/main.go` wraps `mcpserver.Run` in a deferred recover that forces `os.Exit(1)` so a panic in any goroutine takes the whole process down rather than leaving a half-dead orphan.

`make kill-orphans` is the only sanctioned cleanup tool. The Makefile target checks `PPID == 1` and leaves live adapters with real parents untouched. Never run `pkill -9 -f "claude-context-mcp"` or any equivalent broad pattern match; the pattern matches every running adapter on the host and disconnects active sessions.

## Time and timezone

The daemon stores every timestamp in UTC (see `internal/clock/clock.go`). Human-facing MCP and CLI output renders timestamps in the daemon host's local time zone with the abbreviation appended, through `formatLocalTime` in `internal/daemon/render.go`. Machine-facing JSON output preserves UTC via `protojson` so consumers can parse with RFC3339.

When adding new timestamp output, use `formatLocalTime` for any string headed at a human-facing display path. Anything that flows into JSON or the gRPC wire should stay UTC.

## Style and structural conventions

JavaScript and TypeScript style does not apply here; this is Go.

- Top-level functions declared with `func`, not assigned to variables.
- Every function, method, parameter, and return type has a concrete Go type. Avoid `any` and `interface{}` unless a real upstream union requires it.
- Test files live alongside the package source.
- Wrap errors with operation context and the relevant identifier.
- Add package doc comments and exported type comments. Add field comments only when the field's meaning is not obvious from its name.
- Match the existing comment density. The codebase favors comments on the why, not the what.

## Testing and verification

Run all of the following before claiming a change is complete:

- `go test ./...`
- `make lint` (golangci-lint, gofumpt, gocyclo, deadcode, staticcheck-extra; all five gates must pass)
- `make build` (also runs vet and govulncheck)
- For changes that touch the live indexing path, run a live smoke against the user's local Milvus and `lmd-serve` on port 5400.

### Failing make steps

If any `make` step fails, fix the underlying code, test, configuration, or documentation honestly. Do not disable, silence, weaken, baseline, or otherwise circumvent the check. Do not add `|| true`, ignore exit codes, narrow target scopes, raise thresholds, or remove checks unless the user explicitly asks for that exact policy change.

## Deliberately not supported

The Go port is local- and self-hosted-only. The following upstream surfaces are intentionally absent:

- Zilliz Cloud auto-provisioning, `ClusterManager`, free-cluster creation.
- `checkCollectionLimit()` and the Zilliz pricing surface.
- `syncIndexedCodebasesFromCloud()` and description-based recovery.
- `MILVUS_TOKEN`-based address auto-resolution.
- The `MilvusRestfulVectorDatabase` REST client.
- VS Code and Chrome extension packages.
- Telemetry and hosted-service hooks.
- Dedicated VoyageAI, Gemini, and Ollama embedding clients. Use an OpenAI-compatible proxy with `OPENAI_BASE_URL` instead.
