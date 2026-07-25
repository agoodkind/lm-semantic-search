# lm-semantic-search

A fork and Go rewrite of [zilliztech/claude-context](https://github.com/zilliztech/claude-context) that keeps backward compatibility with the Milvus data store used by Claude Context while adding local improvements and features on top.

## Where Current Truth Lives

CLI behavior lives in the current help output, starting with `lm-semantic-search --help` and the grouped subcommand help.

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
