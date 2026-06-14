# lm-semantic-search

A fork and Go rewrite of [zilliztech/claude-context](https://github.com/zilliztech/claude-context) that keeps backward compatibility with the Milvus data store used by Claude Context while adding local improvements and features on top.

## Where Current Truth Lives

- CLI behavior lives in the current help output, starting with `lm-semantic-search --help` and the grouped subcommand help.

- TODO: write config example somewhere + install instructions for MCP

## For Coding Agents

Read [AGENTS.md](AGENTS.md) before changing this repository. It is the durable project guide and owns the rules that should affect code changes.

This fork is independent of and not affiliated with Zilliz.

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
