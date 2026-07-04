# Code graph (cbm)

The daemon builds a per-codebase structural graph next to the semantic index, so an agent can ask structural questions about a codebase rather than only semantic ones. The graph is produced by a vendored C engine (codebase-memory-mcp, "cbm") linked into the daemon, and it is exposed through a small set of read-only graph tools.

## Overview

The code graph is a derived side-index. The semantic index stays the primary artifact, and the graph is built from the same files on the same sync passes. A graph never being present, being stale, or failing to build is not fatal: the semantic index still works and still answers "not indexed" or "ready" on its own. This area covers what the graph is, when it is built, how it is queried, how it is packaged, and how it stays current with upstream.

Related jumping-off points:

- The incremental sync flow that also drives graph refresh lives in the AGENTS incremental-sync section.
- The shared Milvus collection contract, which the graph deliberately sits outside of, is in the AGENTS drop-in-compatibility section.
- Conversation ingest, a sibling derived surface with its own overview, is at [conversation ingest overview](../conversationingest/overview.md).

## What the graph is, and where it is stored

The graph is a local per-codebase SQLite database, kept under the daemon's graph directory and keyed by the daemon's own codebase id. It is not a Milvus collection, so it is not part of the shared drop-in contract with the upstream TypeScript tool. The upstream tool neither reads nor writes it.

Because the graph is local and derived, it can be removed and rebuilt at any time without affecting the shared semantic collection. A codebase's graph database, together with its SQLite side files, is removed only by an explicit `clear_index` for that codebase. The daemon does not otherwise drop it.

## When the graph is built and refreshed

Graph building rides the semantic index and sync flow. When the daemon finishes indexing or syncing a code codebase, it reconciles the graph as a follow-on step. Reconciliation happens when the graph has not been built, when a previous build did not finish, or when the codebase's files have changed since the graph was last built. The effect is that a not-current graph heals itself on the next sync of that codebase, with no separate action required, since the periodic sweep and the file watcher both drive that sync (see the AGENTS incremental-sync section).

Graph building applies only to code codebases. Conversation and other document codebases do not get a graph.

The graph store is reconciled against the same content signature the semantic index uses, so the graph tracks the indexed file set rather than a separate notion of freshness.

## Concurrency and lifecycle contracts

Three contracts keep the graph safe under concurrent daemon work, and each has an enforcing test in the daemon graph test suite:

- One graph build per codebase at a time. A second build trigger for a codebase that is already building is skipped rather than run in parallel, so the same tree is not parsed twice into the same database.
- A build slot is held until the underlying engine call actually finishes. If a build is cancelled or times out, the daemon detaches from the still-running engine call but keeps the codebase's build slot held until that call returns, so a new trigger cannot start a second build alongside the first.
- Clearing waits for in-flight graph work. When a codebase is being cleared, the daemon waits for active graph operations on it to drain, refuses to open the engine for it during the clear, and then removes the database and its side files together. A graph operation that races a clear is rejected with a retryable conflict rather than resurrecting the cleared database.

The daemon runs the engine with a single internal worker, which keeps engine work serialized per process.

## Querying the graph

The graph is queried through four read-only tools, each addressed by an absolute codebase path:

- `query_graph` runs a graph query and returns matching rows.
- `trace_path` traces call relationships from a named function, in a chosen direction and depth.
- `get_architecture` summarizes structure, optionally scoped to a path.
- `manage_adr` reads or updates the codebase's architecture decision records.

The daemon owns the project identity and injects it into every graph tool call, so callers never pass it. Only these four tools are accepted. The engine's own repository-indexing entry point is not exposed as a tool, because indexing is driven by the daemon's own index and sync flow, not by ad hoc tool calls.

## Freshness in status and doctor

`get_indexing_status` and `doctor` report the graph's freshness for a code codebase in plain language, alongside the semantic status. The exact wording is intentionally left to the presentation layer and is not restated here, so this document does not drift when that wording changes. The behavioral contract is that the human surfaces never leak internal storage terms, and that a not-current graph reads as something that heals on its own rather than as an error the reader must fix.

## How the engine is packaged

The cbm engine is vendored as a git submodule under `third_party/cbm` and is compiled into the daemon as a localized static archive through the go-makefile cgo-dependency hook, so a normal `make build` produces it. Two properties matter for correctness:

- Only a small set of the engine's `cbm_` entry-point symbols is exported from the archive; everything else is localized. This keeps the engine's internals from colliding with the daemon's other C dependencies.
- The engine's bundled allocator is compiled so that it does not override the Go runtime's allocator.

Cross-compiled release builds resolve the archiver and related tools from the active toolchain, including the case where a build wrapper shadows the compiler on PATH. This lets the darwin cross build find its own archiver rather than falling back to one that cannot index the object files.

## Staying current with upstream

A scheduled workflow proposes submodule pin bumps for the cbm engine. It searches the newest upstream commits, and it accepts a commit only if that commit both builds and passes the engine round-trip test. It then opens an auto-merge pull request for the bump. The workflow only ever moves the pin forward: it refuses a candidate that would move the pin backward or sideways relative to the current pin. The build-and-test search runs unprivileged and without any token, and a separate privileged step opens the pull request after re-checking the chosen commit, so an untrusted upstream build cannot reach a token or move the pin on its own.

## Incremental rebuild and pruning

Re-indexing a codebase over its existing graph database converges the graph to the current file set. A file that has been removed has its symbols pruned from the graph on the next build, and surviving files keep theirs. This is why the daemon reuses a codebase's graph database across builds rather than clearing and rebuilding from scratch, and it is verified directly by the engine's incremental pruning test.
