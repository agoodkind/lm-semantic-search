# Presentation choke point

## Problem

The daemon shows information through many renderers. Each one reads raw model
structs and decides on its own how to present them. That freedom caused a
string of one-surface-at-a-time bugs: the job list showed raw `failed` while
the banner said paused, the job magnitude line never got the reuse split the
status view got, and the TUI falls back to the raw status string.

The rule "no surface re-derives presentation from raw records" exists today as
a comment and a partial guard test. Nothing stops the next formatter from
forking. The fix is structural: one resolver layer produces typed view models,
one render layer formats them, and the compiler forbids the render layer from
seeing raw models at all.

## Audit

The audit is closed-form. The proto is the only way information leaves the
daemon, so the surface list is: every RPC response field, every daemon
formatter, and every client-side formatter. Four independent audits produced
the inventory. Key facts:

- 18 RPCs. 15 carry `display_text`. Version, WatchJobs, and Shutdown do not.
- The envelope is inconsistent. 8 RPCs show the degraded banner. 7 text RPCs
  never do: StartIndex, SyncIndex, ClearIndex, CancelJob,
  RegisterConversationCollection, SyncConversationManifest,
  UpsertConversationDocuments, DeleteConversation.
- Two display strings are composed inside `grpc_server.go` itself:
  the manifest sync message (line 538) and the start-index merge note
  (line 221).
- About 30 render functions read raw `model.*` fields. About 17 already format
  resolved views. 6 status templates format the `statusView` struct.
- The search renderers read raw `StoredChunk` rows directly.
- The TUI (`cmd/lm-semantic-search/codebase_list_tui.go`) falls back to raw
  `GetStatus()` at line 352, falls back to a hardcoded glyph and the raw
  status label at lines 357-364, and keys colors off its own local map.
- Already-clean choke points that stay as they are: `adapterr` for error text,
  `correlation.HeaderLine` for trace refs, the `status` vocabulary for glyphs,
  labels, banner headlines, and search notes, and `response.FormatProto` plus
  the MCP server as pure relays of `display_text`.

## Design

Three layers with a compile-time wall between the last two.

```
model.* records
      |
      v
boundary resolvers   (internal/daemon, the ONLY presentation code
      |               allowed to read model.*)
      v
view models          (internal/view, pure data)
      |
      v
render layer         (internal/render, imports view ONLY,
                      cannot import internal/model)
```

### internal/view

One view model per output. Pure data. No methods that read models.

- `BannerView`: headline, detail.
- `Surface`: codebase display status, glyph, label. The shape moves to
  `view`; the `status` package keeps the resolution rules and vocabulary.
- `JobSurface`: state label, error line, superseded fields (exists).
- `FailureSurface`: the failure detail view (today `codebaseFailureView`).
- `ProgressSurface`: the new progress view. See below.
- `StatusView`: the template view (moved from the daemon).
- `SearchView`, `ConversationSearchView`: results reduced from `StoredChunk`
  at the boundary. Render never sees a raw chunk.
- `ListSummary`: the job list counts (completed, failed, superseded,
  canceled, queued, running, canceling).
- `StartIndexView`: start acknowledgment plus the merge note inputs.
- `MutationAckView`: clear, cancel, sync, and conversation acks.
- `DoctorView`: diagnostics plus the dropped-codebases section.
- `TimingView`: started, updated, completed, duration strings.

The `status` package keeps the vocabulary (glyphs, labels, headlines, notes)
and the pure resolution rules. `view` holds the shapes the render layer
formats.

### ProgressSurface

This view fixes every number ambiguity found in the incident review.

Fields:

- `Denominator`: count plus scope label. The scope label is derived from the
  run mode: "changed documents" for a delta or ingest, "documents (full
  build)" for a first build, "documents (forced reindex)" for a forced run.
  A bare unlabeled total can no longer render.
- `RunMode`: one word for the pass: first build, changed files, forced
  reindex, or resuming. Resume is detected at plan time (checkpoint seed or
  daemon-resume client) and carried here, so a resume pass no longer renders
  like starting over.
- `Checked`, `Embedded`, `AlreadyIndexed`: AlreadyIndexed = checked minus
  embedded minus removed/skipped. The fast-forward part of a pass is visible.
- `ChunksThisRun`, `ChunksReused`, `ChunksInCollection`: per-run chunk counts
  never render without the collection total next to them.
- `ScopeLine`: the added/modified/removed counts with their own unit
  (conversations for a manifest diff), shown as classification, separate from
  the progress counter.

Corpus totals become a tracked fact. The daemon updates the stored collection
totals on every successful write, not only at end of run, and serves them into
`ProgressSurface`. A failed run no longer leaves the totals frozen at the last
success.

Example rendering for a resumed ingest:

```
Resuming after restart: checking changed work, embedding only what's new
📄 238 of 1,011 changed documents checked · 12 embedded · 226 already indexed
🧩 29 chunks added this run · 33,240 in collection
Changed since last sync: 1,004 conversations added · 7 modified
```

### internal/render

All formatters and the 6 templates move here. The package imports
`internal/view` and never `internal/model`. A formatter that tries to read a
raw record fails to compile. A small import-list test also asserts the
package's import set, so nobody re-adds the dependency.

The two stray compositions in `grpc_server.go` move behind view models.

### Envelope

Every text-bearing RPC goes through `envelopeText`. The degraded banner shows
on all of them, not just 8. Version, WatchJobs, and Shutdown stay text-free by
design.

### Clients

- The TUI's raw `GetStatus()` fallback and label fallbacks are deleted. The
  daemon always sends resolved tokens.
- TUI colors key off `display_status` values only.
- A `go/ast` guard over `cmd/` forbids `GetStatus()` and `GetState()` in
  display code, so the client cannot fork again.
- `response.FormatProto` and the MCP server stay pure relays. No change.

### Riders (same plan, shipped together)

1. `ResumeOrphanedJobs` skips `Kind == Document` codebases. The boot resume
   pass created jobs against `chat://` URIs as filesystem paths.
2. One-time cleanup deletes the ghost registry record
   `/chat:/clyde-conversations` (kind "code", status missing) that the resume
   bug created.
3. `canonicalizePath` rejects arguments containing a URI scheme instead of
   resolving them as filesystem paths.

## Testing

- The existing guard test stays through the migration. After the wall lands it
  shrinks to the `cmd/` client guard plus the render import-list test.
- Every view model gets table tests in its resolver: progress denominators per
  run mode, resume detection, corpus totals freshness, list summary
  reconciliation, search view reduction.
- Render tests assert formatting from view fixtures only.
- Existing render output expectations stay green or change deliberately with
  the new wording, never silently.
- Each migration step lands with `go test ./...`, `make lint`, and
  `make build` green.

## Migration order

1. Create `internal/view`. Move or alias the existing resolved types
   (`JobSurface`, `codebaseFailureView`, `statusView`, `bannerView`,
   `searchView`).
2. Consolidate boundary resolvers in the daemon. Add the `ProgressSurface`
   resolver and the corpus-total tracking.
3. Create `internal/render`. Move formatters and templates. Cut the
   `internal/model` import. Compile.
4. Uniform envelope across all text RPCs.
5. TUI fallback removal plus `cmd/` guard.
6. Riders: resume kind-skip, ghost record cleanup, URI rejection in
   `canonicalizePath`.

## Out of scope

- Logs and the debug listener (operator surfaces, not product surfaces).
- The MCP playbook static text.
- TS upstream compatibility surfaces (collection names, chunk ids) are data
  contracts, not presentation, and do not change.
- Color palettes stay client-side, keyed off resolved display values.
