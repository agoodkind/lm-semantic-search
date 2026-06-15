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

## MCP Installation

Build and install the daemon, CLI, and MCP adapter:

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
