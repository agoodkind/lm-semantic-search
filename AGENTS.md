# AGENTS

Durable instructions for coding agents working in this repository.

Keep this file short, current, and focused on rules that should affect day-to-day code changes. Move long runbooks, dated audits, generated examples, and machine-specific workflows into `docs/` if they ever appear.

## Project purpose

`lm-semantic-search` is a local Go rewrite of `zilliztech/claude-context`.

The repo builds three binaries:

- `lm-semantic-search-daemon`, the long-lived gRPC daemon
- `lm-semantic-search`, the operator CLI
- `lm-semantic-search-mcp`, the stdio MCP adapter

VS Code and Chrome extension clients are intentionally out of scope here.

The daemon's service identity is `io.goodkind.lm-semantic-search-daemon` on macOS launchd and `lm-semantic-search-daemon.service` for a Linux systemd user unit.

The only compatibility requirement with the upstream TypeScript tool is the shared Milvus data store: collection names, schema, and chunk ids. The service identity is not shared with the upstream tool.

## Transport contract

The daemon transport is gRPC only, with protobuf definitions managed by `buf`. The repo does not define or accept a JSON-RPC control plane.

Sources of truth:

- `proto/*`
- `buf.yaml`
- `buf.gen.yaml`

## Typescript upstream drop-in compatibility

The Milvus collection is the portable index, shared byte-for-byte with the upstream Typescript adapter. Both tools name it `hybrid_code_chunks_<md5(path)[:8]>`, use the same schema, and compute the same deterministic chunk id `chunk_<sha256(path:start:end:content)[:16]>`.

Each tool keeps its own private bookkeeping outside Milvus:

- Typescript at `~/.context/mcp-codebase-snapshot.json` and `~/.context/merkle/<md5(path)>.json`
- Go daemon at `XDG_STATE_HOME/lm-semantic-search/registry.json`, and `XDG_STATE_HOME/lm-semantic-search/merkle/<codebase-id>.json`

The `~/.context` directory stays read-only for daemon compatibility inputs such as `.env`, `.sync-trigger`, and TS snapshot or merkle adoption.

The contract is two-way and reduces to one rule the daemon fully controls: Never silently drop or rename a shared collection, and never write TS's bookkeeping files (the Go daemon only reads the TS merkle to seed adoption).

### Adoption (TS to Go)

When `lm-semantic-search` resolves a path with a Milvus collection but no registry entry, the daemon adopts that path as a Go-managed codebase.

Adoption persists a registry record with a stable id through `adoptUnregisteredCodebase`. During adoption, the daemon calls `LoadTSMerkle` and tries to seed the Go merkle snapshot from the TypeScript merkle. If that merkle data is missing, empty, unreadable, or cannot be written, the daemon keeps the adopted codebase and continues without the seed.

After the registry update, the daemon notifies the codebase lifecycle hook. When file watching is enabled, the watcher uses that notification to add the codebase root.

To reconcile the adopted collection with the current files on disk, the daemon enqueues a deferred refresh sync. If `LoadTSMerkle` produced a usable seed, the daemon diffs that seed against current files and re-embeds only changed paths. Without a usable seed, the sync treats every current file as changed.

### Switch-back (Go to TS)

When the Go daemon writes or updates a Milvus collection, it keeps that collection compatible with the TypeScript adapter.

A compatible collection uses the same Milvus format as the TypeScript adapter. That format includes the collection name, the collection schema, and the chunk ids.

Both tools name the collection `hybrid_code_chunks_<md5(path)[:8]>`. This shared name lets the TypeScript adapter find a collection that the Go daemon created or updated.

Snapshot files stay outside Milvus. The TypeScript adapter keeps its own snapshot files and can rebuild them when it reads an existing shared collection.

After that rebuild, the TypeScript adapter can reuse the existing Milvus collection and re-embed only changed files.

Only explicit `clear_index` may drop the shared collection. The Go daemon must not otherwise drop or rename it.

- Code that constructs collection names, the schema, or chunk ids must preserve the shared invariant
- The only collection deletion is explicit `clear_index`. There is no orphan-collection garbage collection; a collection without a registry entry is adopted, never dropped.

## Path identity

A codebase's identity is the real directory its path resolves to. `canonicalizePath` (`internal/daemon/manager_paths.go`) makes the requested path absolute and resolves it with `filepath.EvalSymlinks`, and the registry record, the merkle snapshot, and the Milvus collection name all key off that resolved real path. A symlink's own location or name may change without breaking the link, because any symlink that resolves to an indexed real directory resolves to that codebase. `get_indexing_status` appends `🔗 symlink resolved to: <real path>` when the queried path traverses a symlink.

For a codebase rooted at a symlink the shared-collection invariant does not hold: the upstream TS adapter names its collection from `path.resolve` (`packages/core/src/context.ts`), which does not follow symlinks, so the Go name (md5 of the real path) and the TS name (md5 of the symlink path) differ and the two tools do not share that collection. Non-symlinked roots resolve identically in both tools and stay shared.

## Embedding

The Go port supports exactly one embedding provider, an OpenAI-compatible HTTP adapter. `OPENAI_BASE_URL` points at any endpoint that speaks the OpenAI embeddings API.

- Do not add provider-specific clients for VoyageAI, Gemini, or native Ollama. Anything that speaks OpenAI on a configurable base URL works without code changes.
- Do not assume an internet connection. Default config should let `claude-context-mcp` start and answer "not indexed" gracefully even when the embedding endpoint is unreachable.

## Splitter

- The AST splitter and the tree-sitter grammars live in `goodkind.io/gksyntax`, a git submodule under `third_party/gksyntax`. The indexer imports `goodkind.io/gksyntax/chunk`; lms holds no tree-sitter or chunking code of its own. gksyntax is consumed through `go.work` rather than a module `require`, because it vendors the dart and swift grammars as its own submodules whose C sources are absent from a Go module zip. The grammar registry (`GrammarForLanguage`) and the extension-to-language map live in `goodkind.io/gksyntax/treesitter`; adding a grammar is a `GrammarForLanguage` case and an `extensionLanguages` entry there, and nothing else, because the chunker is grammar-agnostic.
- AST (default): `tree-sitter` parsers cover JavaScript, TypeScript, Python, Java, C, C++, Go, Rust, Scala, C#, PHP, Ruby, Bash, JSON, HTML, CSS, Kotlin, Objective-C, Dart, and Swift. Eighteen are pinned Go-module grammars in `gksyntax`'s `go.mod`. Dart and Swift have no usable module against the pinned runtime, so each is a pinned git submodule under `gksyntax`'s `treesitter/grammars/<language>/upstream`, compiled through a hand-written cgo `binding.go` with `grammar_parser.c` and `grammar_scanner.c` shims. The runtime `github.com/tree-sitter/go-tree-sitter` accepts grammar ABI versions 13 through 15.
- The Swift submodule commits only its grammar definition, not the generated parser, so the parser is produced from the pinned submodule by the `make gksyntax-grammars` target, which `make build`, `make test`, and `make lint` run first. The generated parser stays inside the submodule working tree (gitignored there) and is never committed. A build host needs the `tree-sitter` CLI and `git submodule update --init --recursive`.
- Chunking method (cAST): the AST path walks the parse tree and balances chunk size. A node within the budget becomes one chunk; a larger node splits into its children whose chunks are greedily merged back up to the budget; a larger node with no children is cut on language-aware separators. The budget counts non-whitespace bytes. `gksyntax`'s `chunk` package holds this walk; there is no per-language declaration list.
- Recursive separator splitter (`gksyntax`'s `chunk/langchain.go`): the automatic fallback when a file's language has no grammar or a parse fails, and the cut path the AST walk uses for an oversize leaf. Its tables mirror LangChain JS `RecursiveCharacterTextSplitter.fromLanguage` for the languages LangChain defines and a declaration-first chain for the rest. It is also selectable per index request via `splitter: "langchain"`, but AST is the default and produces better search results.

## Incremental sync

Per-codebase Merkle snapshots live under `XDG_STATE_HOME/lm-semantic-search/merkle/<codebase-id>.json`. The sync flow lives in `internal/daemon/background_sync.go` and `internal/daemon/manager.go` (`runDeltaSync`).

On every sync request:

1. The daemon captures a new snapshot of the codebase.
2. `merkle.DiffSnapshots` computes `{Added, Modified, Removed}`.
3. An empty diff completes the job as a no-op tagged "already up to date" only when the live semantic collection is still present. If the collection is definitively missing, the daemon falls back to a full Replace instead of treating the codebase as current.
4. The indexer otherwise processes only added and modified files, and `semantic.Reindex` deletes existing Milvus rows for `relative_path in (removed + modified)` before upserting the new chunks.
5. The new snapshot is persisted only after success.

A missing previous snapshot or a missing semantic collection routes back to a full `Replace`.

Conversation ingest (manifest cap, message-level delta, seed and reuse invariants, bootstrap reasons) is documented at [docs/conversationingest/overview.md](docs/conversationingest/overview.md).

The per-codebase code graph (the cbm engine integration, its lifecycle, query tools, build wiring, and submodule auto-bump) is documented at [docs/cbm/overview.md](docs/cbm/overview.md).

## Idempotency

Concurrent MCP requests for the same codebase deduplicate against any in-flight job with a matching effective config, and that includes `force=true` requests. N parallel `index_codebase(force=true)` calls collapse to a single embedding pass instead of cancelling each other in sequence. This defensive shape prevents the machine-blowing-up failure mode where a client fan-out can otherwise launch arbitrary parallel work.

The mechanism is `Manager.dedupAgainstActiveJob` in `internal/daemon/manager.go`. Do not regress this contract when touching `StartIndex` or the job lifecycle.

## Blocking `index_codebase`

The MCP tool accepts `wait: true` and `wait_timeout_seconds` (default 300). When `wait=true`, the handler in `internal/mcpserver/server.go` polls `GetJob` every 1.5 seconds until the job reaches `completed`, `failed`, or `cancelled`, then returns the corresponding `GetIndex` response. On timeout the daemon job keeps running and the tool returns the latest progress.

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

### Failing make steps

If any `make` step fails, fix the underlying code, test, configuration, or documentation honestly. Do not disable, silence, weaken, baseline, or otherwise circumvent the check. Do not add `|| true`, ignore exit codes, narrow target scopes, raise thresholds, or remove checks unless the user explicitly asks for that exact policy change.
