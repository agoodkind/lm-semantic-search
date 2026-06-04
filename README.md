# claude-context-go

A ground-up Go rewrite of the Claude Context runtime, owning the daemon, the operator CLI, and the MCP adapter. VS Code and Chrome extension clients are out of scope here.

Original TypeScript implementation: `github.com/zilliztech/claude-context`. This Go port is independent of, and not affiliated with or endorsed by, Zilliz.

Provided AS IS under the MIT License with no warranty. See [LICENSE](LICENSE).

Coding agents working in this repository should read [AGENTS.md](AGENTS.md) for architectural rules, conventions, and the testing contract. `CLAUDE.md` is a symlink to `AGENTS.md`.

## Binaries

- `claude-contextd`: long-lived daemon.
- `claude-context`: operator CLI.
- `claude-context-mcp`: MCP stdio adapter that forwards every tool call to the daemon over its unix socket.

## TS adapter drop-in

The Go daemon and the upstream TypeScript adapter share one Milvus index per codebase, so a codebase indexed by either tool is searchable through the other with no migration step and neither tool modified. The shared-index contract and the adoption flow are defined in the "TS upstream drop-in compatibility" section of [AGENTS.md](AGENTS.md).

## Build

```sh
make test
make build
```

## Install and deploy

This repo follows the same local deploy shape as `~/Sites/agent-gate`:

- `make install` installs the daemon binary.
- `make install-clients` installs `claude-context` and `claude-context-mcp`.
- `make deploy` performs local install, restarts or installs the user service, waits for readiness, then reports daemon status.
- `make deploy-service` uses the shared `go-service.mk` restart-first, install-on-failure pattern.
- `make daemon-status` checks the daemon through the installed CLI.
- `make daemon-wait` polls the installed CLI until the daemon is reachable.
- `make kill-orphans` SIGKILLs any `claude-context-mcp` process whose PPID is `1`. Live sessions with a real parent stay untouched.

## Configuration

The daemon reads configuration in this order (highest precedence first):

1. Process environment variables.
2. `~/.context/.env` (KEY=VALUE pairs, comments via `#`, respects already-set env vars).
3. `XDG_CONFIG_HOME/lm-semantic-search/config.json`, fallback `~/.config/lm-semantic-search/config.json` (persisted daemon defaults).

The daemon owns its local registry, jobs journal, chunk cache, Merkle snapshots, sockets, locks, and logs under `XDG_STATE_HOME/lm-semantic-search`, fallback `~/.local/state/lm-semantic-search`. The `~/.context` directory remains a compatibility input root for `.env`, `.sync-trigger`, and upstream TS snapshot or merkle adoption.

Required keys:

- `EMBEDDING_PROVIDER=OpenAI`
- `OPENAI_API_KEY=<key>`
- `OPENAI_BASE_URL=<base URL>` (omit when using the public OpenAI endpoint).
- `EMBEDDING_MODEL=<model id served by the upstream>`
- `MILVUS_ADDRESS=<host:port>` for the vector backend.

Optional:

- `EMBEDDING_DIMENSION=<int>` requests a non-default output dimension.
- `EMBEDDING_BATCH_SIZE=<int>` (default 32) is the number of chunks per embedding HTTP call.
- `CUSTOM_EXTENSIONS=.ext1,.ext2,...` adds extra file extensions to discovery.
- `CUSTOM_IGNORE_PATTERNS=glob1,glob2,...` adds extra ignore patterns.
- `HYBRID_MODE=true|false` (default true) toggles Milvus BM25 + dense hybrid.
- `CODE_CHUNKS_COLLECTION_NAME_OVERRIDE` adds a readable infix to collection names.
- `CLAUDE_CONTEXT_BACKGROUND_SYNC=true|false` (default true) enables periodic sync.
- `CLAUDE_CONTEXT_TRIGGER_WATCHER=true|false` (default true) watches `~/.context/.sync-trigger`.
- `CLAUDE_CONTEXT_SYNC_INTERVAL_MS` and `CLAUDE_CONTEXT_SYNC_LOCK_STALE_MS` tune sync cadence and lock recovery.
