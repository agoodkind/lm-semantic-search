# Unified CLI + MCP surface (daemon-owned surface model)

Date: 2026-06-07
Status: approved design, pending implementation plan

## Context

The daemon already renders a composable `display_text` envelope (dependency
banner + correlation header + body) for every read RPC, and both the CLI and the
MCP adapter print it through one shared formatter, `internal/response.FormatProto`
(human prints `display_text`, single-line prints its first line, `--json` prints
the full proto). So most surfaces are already identical across CLI and MCP.

Three problems remain:

1. **Two renderers diverge from the shared text.** The CLI `codebase list` TUI
   (`cmd/lm-semantic-search/codebase_list_tui.go`) rebuilds its table from proto
   fields and owns its own `statusGlyphs` / `statusColors` / `statusLabel` maps.
   The MCP `doctor_indexing` tool appends a "dropped codebases" section the
   daemon's `display_text` does not have. Adding the `waiting` status in the
   prior change had to be done in two places (daemon renderer and the TUI maps),
   which is the drift this design removes.

2. **The shared text is sometimes ambiguous because the body fields are
   incomplete.** During a reindex the status shows a bare `📈 N chunks so far`.
   On the wire, `files_embedded`, `chunks_total`, and the change breakdown come
   across as zero, so `📥 848 of 2103 files · 🧩 472 chunks` does not form a
   coherent picture and cannot answer "is my prior work being reused or redone."

3. **The heading names the internal job path, not the trigger.** A forced rebuild
   that resumes from a failed run's checkpoint reads "Reindexing changed files"
   even though the codebase never completed an initial index.

## Principle

The daemon is the single source of every surface's vocabulary and content. Per
surface it produces, from one builder, both the `display_text` envelope and a set
of structured fields carrying the same values. They cannot drift because they
come from the same code. The CLI human mode prints the text; the CLI list TUI and
`--json` read the structured fields; MCP returns the text. A new status or
heading is one edit in the daemon.

## Surface vocabulary (one place)

These live only in the daemon and are exposed as tokens on the wire:

- **status token**: `preparing`, `indexing`, `waiting`, `indexed`, `stale`,
  `failed`, `missing` (the existing `displayStatus`, already on the wire as
  `display_status`).
- **glyph token**: the shape per status (`◌ ◐ ⋯ ● △ ✗ ⊘`), today duplicated in the
  CLI TUI.
- **status label**: the human word per status (e.g. `not indexed` spelled as two
  words), today in the CLI TUI's `statusLabel`.
- **trigger**: `initial_build`, `forced_reindex`, or `changed_files`. This names
  what started the run and selects the heading.

Color stays a client concern (terminal palette), keyed off the status token. The
daemon does not emit colors.

## Heading logic (daemon)

The heading derives from whether a completed index exists and from the trigger:

- no `LastSuccessfulRun` → `Building initial index` (even when resuming a failed
  run's checkpoint).
- `LastSuccessfulRun` present and trigger `forced_reindex` → `Forced reindex`.
- `LastSuccessfulRun` present and trigger `changed_files` → `Indexing new changes`.

This requires the job to carry its trigger, because a first index and a forced
reindex are both `operation=index` today and are only told apart by the `force`
flag plus whether a successful run exists.

## Breakdown surface (in-progress build / reindex)

```
📁 claude-code-src
🔄 Building initial index: 41%
📥 848 of 2103 files processed
🧩 7461 chunks total
├─ ♻️ 6989 reused
└─ ➕ 472 embedded this run
🕐 Updated 11:34 PM PDT
```

`total = reused + embedded this run`, so the numbers are self-evident and the
reuse-vs-redo question is answerable from the screen: genuine reuse shows a large
`reused` and a small `embedded this run`; a silent full re-embed shows `reused`
at 0. The `files` line shows `processed of total`; "changed" and "processed" are
the same dimension at different times, so only the progressing ratio is shown.

## Proto additions (minimal)

- `Codebase`: add `glyph_token` and `status_label` next to the existing
  `display_status`, for list rows.
- `Job`: add `trigger` (derived from operation plus the `force` flag), for the
  heading.
- `Progress`: add `chunks_reused` and populate the existing `chunks_total` (the
  live collection count). The chunks tree reads `chunks_total`, `chunks_reused`,
  and `chunks_generated` (embedded this run).
- `dependency_health` stays as already shipped; the banner reads it.

These are additive fields. The shared Milvus contract and the transport are
unchanged.

## Instrumentation fix (daemon + semantic)

- The semantic reuse/copy path reports the count of reused (copied) chunks into
  `Progress.chunks_reused`.
- The live whole-collection count populates `Progress.chunks_total` for an
  in-flight run (the field exists but is currently zero on the wire).
- The job records its trigger so the heading is honest.

This closes the ambiguity in problem 2 and the mislabel in problem 3.

## Client consumption

- **CLI human**: unchanged, prints `display_text`.
- **CLI `--json`**: unchanged mechanism, now richer because the structured fields
  are first-class.
- **CLI `codebase list` TUI**: renders rows from the daemon tokens
  (`status token`, `glyph_token`, `status_label`, file count) instead of its own
  maps. It keeps interactivity (navigation, sizing) and the local color palette,
  but stops owning glyph/label/breakdown logic.
- **MCP**: returns `display_text` verbatim (already does). The
  `doctor_indexing` "dropped codebases" section moves into the daemon's Doctor
  surface so MCP stops computing it client-side.

## Data flow

```
gRPC read RPC
  └─ daemon surface builder (one place)
       ├─ display_text  (envelope: banner + header + body)   → human text
       └─ structured fields (status/glyph/label/trigger/breakdown) → TUI + JSON
CLI: human prints display_text · TUI reads structured · json prints proto
MCP: returns display_text · json available
```

## De-duplication removed

- CLI `codebase_list_tui.go` `statusGlyphs` / `statusColors` / `statusLabel`
  logic moves to the daemon tokens (the color map stays, keyed off the token).
- MCP `callDaemonDoctor` dropped-section computation moves to the daemon Doctor
  surface.
- The remaining `fmt.Sprintf` body renderers (`failed`, `missing`, `stale`,
  `search`, `job`, `list`) become template partials that also populate the
  structured fields, so text and struct share one builder.

## Testing

- Golden tests that the rendered text and the structured fields agree for each
  surface and status.
- A table test asserting every status token has a glyph and a label, so a new
  status cannot be added in one place and forgotten in another.
- Instrumentation tests: `Progress` carries `chunks_reused` and `chunks_total`,
  and the chunks tree text matches those fields; the heading matches the trigger
  and `LastSuccessfulRun` state.
- A CLI TUI test that rows render from the daemon tokens, not local maps.
- An MCP test that the doctor dropped-section comes from the daemon.

## Out of scope (YAGNI)

- No generic templating DSL; the structured fields plus the existing Go
  templates are enough.
- No transport changes and no change to the shared Milvus collection contract.
- Terminal color palette stays client-side.

## Verification

- `go test ./...`, `make lint`, `make build` all pass.
- Live smoke on the local daemon and lmd on `:5400`: a reindex shows the
  `total / reused / embedded this run` tree with coherent numbers; a never-
  completed build reads `Building initial index`; an explicit `--force` over a
  good index reads `Forced reindex`; the CLI `codebase list` and the MCP
  `list_indexing_statuses` show the same status vocabulary including `waiting`.
