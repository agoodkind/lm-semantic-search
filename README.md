# lm-semantic-search

`lm-semantic-search` provides a local semantic search daemon, operator CLI, and MCP stdio adapter for codebases. VS Code and Chrome extension clients are out of scope here.

This project is independent of `zilliztech/claude-context` and is not affiliated with or endorsed by Zilliz. It is provided under the MIT License; see [LICENSE](LICENSE).

## Where Current Truth Lives

- Architecture, compatibility rules, indexing behavior, service identity, and the verification contract live in [AGENTS.md](AGENTS.md).
- CLI behavior lives in the current help output, starting with `lm-semantic-search --help` and the grouped subcommand help.
- Daemon API and transport rules live in [service.proto](proto/lmsemanticsearch/v1/service.proto), [buf.yaml](buf.yaml), and [buf.gen.yaml](buf.gen.yaml).
- Runtime configuration behavior lives in [config.go](internal/config/config.go) and [config_test.go](internal/config/config_test.go).
- Build and deploy behavior lives in [Makefile](Makefile), [bootstrap.mk](bootstrap.mk), and the testing section of [AGENTS.md](AGENTS.md).
- Status wording and display rules live in [internal/status/](internal/status/).

## For Coding Agents

Read [AGENTS.md](AGENTS.md) before changing this repository. It is the durable project guide and owns the rules that should affect code changes.
