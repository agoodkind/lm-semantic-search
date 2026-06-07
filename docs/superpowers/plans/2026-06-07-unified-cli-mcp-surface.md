# Unified CLI + MCP surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the daemon the single source of every status surface's vocabulary and content, add trigger-driven headings and a coherent reuse-aware chunk breakdown, and have the CLI table and MCP doctor derive from that one source.

**Architecture:** The daemon already renders a `display_text` envelope that both clients print via `internal/response.FormatProto`. This plan adds structured surface fields (glyph token, status label, trigger, reused/total chunks) produced by the same daemon builders, fixes the chunk/file instrumentation in the semantic and delta paths, and points the CLI list TUI and MCP doctor at the daemon tokens instead of their own logic.

**Tech Stack:** Go, protobuf via buf, text/template status views, bubbletea TUI, gRPC over a unix socket.

**Baseline already on this branch (do not redo):** infra-failure classification, dependency-health record, `waiting` display fold, the banner and envelope, the `dependency_health` proto field, the relabeled failed-job diagnostics line, and search/job emoji. This plan builds on those.

---

### Task 1: Reuse-aware chunk counters in the semantic progress

**Files:**
- Modify: the file defining `type Progress` in `internal/semantic` (confirm the exact filename while implementing).
- Modify: the `embedChunkBatch` and `insertChunksBatched` paths where progress is reported.
- Test: `internal/semantic/reuse_test.go`

- [ ] **Step 1: Write the failing test** in `internal/semantic/reuse_test.go`. It asserts that `embedChunkBatch` over a 3-chunk slice with 2 reuse hits reports `ChunksReused: 2, ChunksEmbedded: 1` through the progress callback. Extend the existing `countingEmbedder` test to capture the last `Progress`.

```go
func TestProgressReportsReusedVsEmbedded(t *testing.T) {
	embedder := &countingEmbedder{}
	service := &Service{embedder: embedder}
	chunks := []model.StoredChunk{{Content: "reused-A"}, {Content: "fresh-B"}, {Content: "reused-C"}}
	reuse := map[string][]float32{contentVectorKey("reused-A"): {7}, contentVectorKey("reused-C"): {9}}
	// drive the batch insert with a progress sink; assert last.ChunksReused==2 && last.ChunksEmbedded==1
}
```

- [ ] **Step 2: Run it, expect FAIL.** `go test ./internal/semantic/ -run TestProgressReportsReusedVsEmbedded -v` fails to compile because `ChunksReused` is undefined.

- [ ] **Step 3: Add fields to `semantic.Progress`.** Add `ChunksReused int` and `ChunksEmbedded int` next to the existing chunk fields. In `embedChunkBatch`, count reuse-map hits as reused and embedder results as embedded, then thread both into the `Progress` the callback receives in `insertChunksBatched`.

- [ ] **Step 4: Run it, expect PASS.** `go test ./internal/semantic/ -run TestProgressReportsReusedVsEmbedded -v`.

- [ ] **Step 5: Commit.** `git add internal/semantic && git commit -m "Report reused vs embedded chunk counts from the semantic progress callback"`.

---

### Task 2: Thread reused/total chunks into job Progress

**Files:**
- Modify: `internal/model/types.go` `Progress` struct, adding `ChunksReused int32`.
- Modify: `internal/daemon/manager_delta.go` `reportDeltaProgress` and the bootstrap adapter that maps `semantic.Progress`/`indexer.Progress` to `model.Progress`.
- Modify: `internal/daemon/grpc_server.go` `fillLiveChunkTotal` so an in-flight run populates `Progress.ChunksTotal` from the live collection `Count`.
- Test: a new `internal/daemon/manager_progress_test.go`.

- [ ] **Step 1: Write the failing test.** After a reindex pass reporting `ChunksReused: 6989, ChunksEmbedded: 472`, the job's `model.Progress` carries `ChunksReused==6989` and `ChunksGenerated==472`.

- [ ] **Step 2: Run it, expect FAIL.** `ChunksReused` is undefined on `model.Progress`.

- [ ] **Step 3: Add `ChunksReused int32` to `model.Progress`.** Map it through the progress adapters in `reportDeltaProgress` and the bootstrap adapter. In `fillLiveChunkTotal`, set `ChunksTotal` from `semantic.Count` when a job is active and the collection exists.

- [ ] **Step 4: Run it, expect PASS.**

- [ ] **Step 5: Commit.** `git commit -am "Carry reused and live-total chunk counts into job progress"`.

---

### Task 3: Job trigger (operation + forced)

**Files:**
- Modify: `internal/model/types.go` `Job` struct, adding `Forced bool`.
- Modify: the `StartIndex` job-creation path in `internal/daemon`, setting `Forced` from the request force flag.
- Test: `internal/daemon/manager_test.go`.

- [ ] **Step 1: Write the failing test.** `StartIndex(..., force=true)` produces a job with `Forced==true`; `force=false` yields `Forced==false`.

- [ ] **Step 2: Run it, expect FAIL.**

- [ ] **Step 3: Add `Forced bool` to `model.Job`.** Set it in the job-creation path from the force flag.

- [ ] **Step 4: Run it, expect PASS.**

- [ ] **Step 5: Commit.** `git commit -am "Record the force flag on the job for trigger-aware headings"`.

---

### Task 4: Trigger-driven headings and the reuse-aware breakdown template

**Files:**
- Modify: `internal/daemon/status_render.go` `statusView`, adding `Heading`, `ChunksReused`, `ChunksEmbeddedThisRun`.
- Modify: `internal/daemon/render.go` `renderIndexingActive`, computing the heading and filling the new fields.
- Replace: the building view template content with one that reads `.Heading` and the chunk tree:

```
📁 {{ .Name }}
🔄 {{ .Heading }}: {{ .Percent }}%
📥 {{ .FilesProcessed }} of {{ .FilesTotal }} files processed
🧩 {{ .ChunksTotal }} chunks total
├─ ♻️ {{ .ChunksReused }} reused
└─ ➕ {{ .ChunksEmbeddedThisRun }} embedded this run
🕐 Updated {{ .UpdatedAt }}
```

- Test: `internal/daemon/render_test.go`.

- [ ] **Step 1: Write failing tests** for `headingFor`:

```go
func TestHeadingFor(t *testing.T) {
	cases := []struct{ name string; cb model.Codebase; job model.Job; want string }{
		{"initial", model.Codebase{}, model.Job{Operation: "index", Forced: true}, "Building initial index"},
		{"forced", model.Codebase{LastSuccessfulRun: &model.IndexRunSummary{}}, model.Job{Operation: "index", Forced: true}, "Forced reindex"},
		{"changed", model.Codebase{LastSuccessfulRun: &model.IndexRunSummary{}}, model.Job{Operation: "sync"}, "Indexing new changes"},
	}
	// assert headingFor(cb, &job) == want
}
```

- [ ] **Step 2: Run it, expect FAIL.** `headingFor` is undefined.

- [ ] **Step 3: Implement `headingFor(codebase, job)`** and wire it into `renderIndexingActive`. Set `view.Heading`, `view.ChunksReused = progress.ChunksReused`, `view.ChunksEmbeddedThisRun = progress.ChunksGenerated`, and `view.ChunksTotal = max(progress.ChunksTotal, progress.ChunksReused + progress.ChunksGenerated)`. Render the new template.

- [ ] **Step 4: Adjust the building/incremental tests** to the new heading and tree lines. Run `go test ./internal/daemon/ -run 'Heading|Building|Incremental|StatusTemplate' -v`, expect PASS.

- [ ] **Step 5: Commit.** `git commit -am "Render trigger-driven heading and reuse-aware chunk tree in the indexing view"`.

---

### Task 5: Glyph and label tokens as a daemon single source

**Files:**
- Create: `internal/daemon/status_vocab.go` with `glyphForDisplay(displayStatus) string` and `labelForDisplay(displayStatus) string`, covering every `displayStatus` including `waiting`.
- Modify: `internal/daemon/grpc_server.go` to set `GlyphToken` and `StatusLabel` next to `DisplayStatus`.
- Test: `internal/daemon/status_vocab_test.go`.

- [ ] **Step 1: Write the failing test.** A table over every `displayStatus` constant asserts each has a non-empty glyph and label, so adding a status without a glyph fails.

- [ ] **Step 2: Run it, expect FAIL.** `glyphForDisplay` is undefined.

- [ ] **Step 3: Implement the two functions in `status_vocab.go`,** moving the glyph/label vocabulary out of the CLI TUI into the daemon.

- [ ] **Step 4: Run it, expect PASS.**

- [ ] **Step 5: Commit.** `git commit -am "Add daemon glyph and label vocabulary for display statuses"`.

---

### Task 6: Proto surface fields and boundary population

**Files:**
- Modify: `proto/lmsemanticsearch/v1/service.proto`. `Codebase` gains `string glyph_token` and `string status_label`; `Job` gains `bool forced` and `string trigger`; `Progress` gains `int32 chunks_reused`.
- Regenerate: `buf generate`.
- Modify: `internal/pbconv/pbconv.go` and `internal/daemon/grpc_server.go` to populate the new fields.
- Test: `internal/daemon` boundary test asserting `ListIndexes` rows carry the tokens.

- [ ] **Step 1: Write the failing test.** A `ListIndexes` response row for a `waiting` codebase has `GlyphToken == "⋯"` and `StatusLabel == "waiting"`.

- [ ] **Step 2: Run it, expect FAIL.** The field is undefined.

- [ ] **Step 3: Edit the proto, run `buf generate`, populate the fields.** Set glyph/label in `grpc_server.go` after `pbconv.ToCodebase` (pbconv cannot import daemon, so the daemon vocab is applied at the boundary). Map `forced`, `trigger`, and `chunks_reused` through `ToJob` and the progress mapping.

- [ ] **Step 4: Run it, expect PASS.** `go test ./internal/daemon/ -run 'Surface|ListIndexes|GetIndex' -v`.

- [ ] **Step 5: Commit.** `git add proto gen internal && git commit -m "Add structured surface fields to proto and populate at the gRPC boundary"`.

---

### Task 7: CLI list TUI renders from daemon tokens

**Files:**
- Modify: `cmd/lm-semantic-search/codebase_list_tui.go`. Replace `statusGlyphs` and `statusLabel` with reads of `codebase.GetGlyphToken()` and `codebase.GetStatusLabel()`. Keep `statusColors` (color is client-side) keyed off `GetDisplayStatus()`.
- Test: `cmd/lm-semantic-search/codebase_list_tui_test.go`.

- [ ] **Step 1: Write the failing test.** A row for a codebase with `DisplayStatus: "waiting", GlyphToken: "⋯", StatusLabel: "waiting"` renders `⋯` and `waiting` from those fields.

- [ ] **Step 2: Run it, expect FAIL.** Remove the local maps so the field path is forced.

- [ ] **Step 3: Rewrite `renderRow` glyph/label to read the proto fields,** falling back to the raw status only when the token is empty.

- [ ] **Step 4: Run it, expect PASS.** `go test ./cmd/... -v`.

- [ ] **Step 5: Commit.** `git commit -am "Render CLI list rows from daemon glyph and label tokens"`.

---

### Task 8: Move the MCP doctor dropped-section into the daemon

**Files:**
- Modify: `internal/daemon/render.go` `renderDoctor` and the Doctor manager method to compute and include the dropped-codebases section.
- Modify: `internal/mcpserver/server.go` `callDaemonDoctor` to stop computing the section and just return `display_text`.
- Test: `internal/daemon/render_test.go` and an `internal/mcpserver` test.

- [ ] **Step 1: Write the failing test.** The daemon Doctor `display_text` includes a dropped-codebases line when a tracked codebase's source dir is gone.

- [ ] **Step 2: Run it, expect FAIL.**

- [ ] **Step 3: Move `computeDroppedCodebases` and `renderDroppedSection` into the daemon Doctor path; delete the MCP-side computation.**

- [ ] **Step 4: Run it, expect PASS.** `go test ./internal/daemon/ ./internal/mcpserver/ -v`.

- [ ] **Step 5: Commit.** `git commit -am "Move doctor dropped-codebases section into the daemon surface"`.

---

### Task 9: Full verification

- [ ] **Step 1:** `go test ./...`, expect ALL PASS.
- [ ] **Step 2:** `make lint`, expect exit 0.
- [ ] **Step 3:** `make build`, expect exit 0.
- [ ] **Step 4: Live smoke** on the deployed daemon and lmd `:5400`. A reindex shows the `total / reused / embedded this run` tree with coherent numbers. A never-completed build reads `Building initial index`. A `--force` over a good index reads `Forced reindex`. `codebase list` and MCP `list_indexing_statuses` show the same vocabulary including `waiting`.
- [ ] **Step 5: Commit** any fixture updates and push.

---

## Self-review notes

- Spec coverage: trigger headings (Tasks 3, 4); reuse breakdown and instrumentation (Tasks 1, 2, 4); glyph/label single source (Tasks 5, 6, 7); MCP doctor unification (Task 8); JSON structured fields (Task 6). Banner, waiting, no-echo, and classification are the committed baseline.
- The exact filenames for `semantic.Progress` and the bootstrap progress adapter get confirmed at implementation time by reading the package; the responsibilities are fixed even if a path differs.
