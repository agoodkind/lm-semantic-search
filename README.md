# lm-semantic-search

A fork and Go rewrite of [zilliztech/claude-context](https://github.com/zilliztech/claude-context) that keeps backward compatibility with the Milvus data store used by Claude Context while adding local improvements and features on top.

> [!WARNING]
> This daemon does not coordinate with the upstream TypeScript `claude-context` tool. Running both against the same index at the same time is unsafe: they embed into the same shared Milvus collection, they track their progress in separate checkpoints, and neither one excludes the other. The daemon takes a kernel file lock on `mcp-sync.flock` in its context root, `~/.context` by default and wherever `CLAUDE_CONTEXTD_CONTEXT_ROOT` points otherwise. The upstream tool does not take that lock. A `mcp-sync.lock` directory in the same root is the retired lock protocol's leftover, which the daemon ignores and leaves untouched.

## Where Current Truth Lives

CLI behavior lives in the current help output, starting with `lm-semantic-search --help` and the grouped subcommand help. The daemon binary takes no help output, so its two commands are described here: `lm-semantic-search-daemon version` and [`lm-semantic-search-daemon sandbox`](docs/sandbox.md).

[`docs/metrics.md`](docs/metrics.md) explains the historical `status --since` report.

## Configuration

The daemon reads `config.json` from `$XDG_CONFIG_HOME/lm-semantic-search/`, or from `~/.config/lm-semantic-search/` when `XDG_CONFIG_HOME` is unset.

Use `OPENAI_API_KEY` and `MILVUS_TOKEN` as environment variables or in `~/.context/.env`. Do not put secret values in checked-in files.

Example `config.json`:

```json
{
  "embeddingProvider": "OpenAI",
  "embeddingModel": "text-embedding-3-small",
  "embeddingBatchSize": 32,
  "embeddingBatchTokenBudget": 6000,
  "openaiBaseUrl": "http://localhost:5400/v1",
  "milvusAddress": "localhost:19530",
  "hybridMode": true
}
```

If both `openaiBaseUrl` and `OPENAI_BASE_URL` are unset, the OpenAI SDK uses its default endpoint. `OPENAI_BASE_URL` overrides `openaiBaseUrl` when both are set.

### Where the daemon keeps its files

Each root below has a default and an environment variable that moves it. A variable that is set wins over the default.

| Variable | Default | Holds |
| --- | --- | --- |
| `CLAUDE_CONTEXTD_CONFIG_ROOT` | `$XDG_CONFIG_HOME/lm-semantic-search` | `config.json` |
| `CLAUDE_CONTEXTD_STATE_ROOT` | `$XDG_STATE_HOME/lm-semantic-search` | the registry, job ledger, merkle snapshots, chunks, and the code graph |
| `CLAUDE_CONTEXTD_CONTEXT_ROOT` | `~/.context` | `mcp-sync.flock`, the file whose kernel lock serializes this daemon's embeds |
| `CLAUDE_CONTEXTD_MODEL_CACHE_ROOT` | the state root | downloaded offline embedding models |
| `CLAUDE_CONTEXTD_SOCKET_PATH` | `<state root>/sockets/lm-semantic-search-daemon.sock` | the gRPC socket clients dial |
| `CLAUDE_CONTEXTD_LOG_PATH` | `<state root>/logs/lm-semantic-search-daemon.log` | the combined log |

The model cache is separate from the state root because the two have different lifetimes. State belongs to one daemon, while a downloaded model is checksum-verified and reusable by every daemon on the machine, so a short-lived daemon can point its state somewhere temporary and still reuse the model.

Moving these by hand is rarely necessary. To run a second daemon that touches none of the operator's files, use [`lm-semantic-search-daemon sandbox`](docs/sandbox.md), which sets them all.

## Offline profile

The `offline` profile runs indexing and search entirely on the local machine, so it needs no Docker, GPU, or hosted model server. It uses an on-disk approximate-nearest-neighbor vector index and an in-process ONNX embedding model that the daemon downloads and caches on first use. The local model provides lower retrieval precision than the default profile. See [docs/offline.md](docs/offline.md) for how it works, its limits, and switching back.

Enable the profile with the Go CLI:

```bash
lm-semantic-search profile offline
```

Select an offline embedding model with `--model`:

```bash
lm-semantic-search profile offline --model bge-small
```

Valid model names are `embeddinggemma` (the default) and `bge-small`. The daemon fetches and caches the selected model on first use.

The command writes `"profile": "offline"` to the daemon `config.json`. You can set the same value directly:

```json
{
  "profile": "offline"
}
```

`CLAUDE_CONTEXT_PROFILE=offline` overrides the file setting. Restart the daemon after changing the profile.

To try the offline profile without changing the installed daemon, run [`lm-semantic-search-daemon sandbox`](docs/sandbox.md). It defaults to offline, writes no `config.json`, and needs no restart.

Offline search is dense-only. It does not include the default profile's BM25 sparse search and hybrid reranking.

Offline collections are stored separately from the default Milvus collections. After switching an offline-indexed codebase back to `standard`, force a reindex:

```bash
lm-semantic-search profile standard
```

Restart the daemon, then run:

```bash
lm-semantic-search codebase index /absolute/path/to/repo --force
```

## MCP Installation

Install the release binaries and user service:

```bash
curl -fsSL https://raw.githubusercontent.com/agoodkind/lm-semantic-search/main/install.sh | bash
```

From source, build and install the daemon, CLI, and MCP adapter:

```bash
make install
```

Install or restart the user service:

```bash
make deploy-service
lm-semantic-search daemon status
```

Add the MCP adapter as a stdio server in the MCP client:

```json
{
  "mcpServers": {
    "lm-semantic-search": {
      "command": "lm-semantic-search-mcp"
    }
  }
}
```

If `lm-semantic-search-mcp` is not on the client process `PATH`, set `command` to the installed binary's absolute path.

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

This fork is independent of and not affiliated with Zilliz.
