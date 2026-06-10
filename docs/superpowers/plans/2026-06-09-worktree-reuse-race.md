# Worktree Sibling-Reuse Race Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a worktree build reuse an already-indexed sibling's vectors even when that sibling is mid-sync, by keying reuse eligibility on durable facts (a sibling that has been indexed and has a collection built with the matching model) instead of the transient `ActiveJobID` flag.

**Architecture:** The reuse mechanism already exists and works (`worktreeSiblingReuseCollections` feeds collection names to `LoadReuseVectors`, seeded into the build in `runBootstrap`). This change is a one-predicate fix to the eligibility gate plus tests. No new mechanism, no schema change, no proto change.

**Tech Stack:** Go, `internal/daemon` (`manager_worktree.go`), `internal/semantic` reuse (unchanged).

---

## Background: what is already true

- A worktree is its own codebase with its own Milvus collection, so it always walks every file. The optimization that should spare re-embedding is sibling-vector reuse: for each chunk, if the parent's collection already has a vector for that exact content, reuse it instead of calling the embedder. Reuse is keyed by `contentVectorKey` = sha256 of chunk content (`internal/semantic/reuse.go`), so a reused vector is correct for matching content regardless of which collection it came from.
- `runBootstrap` (`internal/daemon/manager_delta.go`) samples reuse once at build start: it calls `worktreeSiblingReuseCollections`, then `LoadReuseVectors`, then seeds `state.reuse`.
- `LoadReuseVectors` (`internal/semantic/reuse.go`) calls Milvus `HasCollection` per source and skips a missing one, so passing a collection name that is absent or mid-rename is safe (graceful no-op, not an error).
- The chunk id is relativePath-based (`internal/semantic/service.go:795`, `sha256(RelativePath:StartLine:EndLine:Content)`), locked by `portability_test.go`. This fix reuses only the vector attached to the worktree's own natively-built chunk, so the produced collection is byte-identical to a normal index. No TS drop-in risk.

## The bug

`worktreeSiblingReuseCollections` (`internal/daemon/manager_worktree.go`) excludes a sibling when `ActiveJobID != ""` or `Status != Indexed`:

```go
if codebase.CollectionName == "" || codebase.ActiveJobID != "" {
    continue
}
if codebase.Status != model.CodebaseStatusIndexed {
    continue
}
```

The auto-create trigger `hasIndexedSiblingWorktreeLocked` is looser: it fires when `Status == Indexed || LastSuccessfulRun != nil`. So during a routine background sync of the parent (which sets `ActiveJobID` and flips `Status` to `indexing`), the worktree auto-creates and builds but finds no eligible reuse source, and re-embeds everything. `ActiveJobID` is a transient flag, not a fact about whether a usable collection exists.

## The fix

Change the reuse eligibility to match the trigger's durable eligibility and drop the transient `ActiveJobID` condition:

```go
if codebase.Kind == model.CodebaseKindDocument {
    continue
}
if codebase.CollectionName == "" {
    continue
}
if codebase.Status != model.CodebaseStatusIndexed && codebase.LastSuccessfulRun == nil {
    continue
}
if !reuseModelMatches(codebase.EffectiveConfig, indexConfig) {
    continue
}
collections = append(collections, codebase.CollectionName)
```

This is not loosening a threshold. It re-bases the predicate on durable facts: a sibling that currently is indexed or has at least one past successful run, and has a named collection built with the matching model. An in-flight sync no longer disables reuse. The trigger and the reuse gate now share the same eligibility shape (`Status == Indexed || LastSuccessfulRun != nil`).

## Edge cases (researched)

1. **Busy sibling, previously indexed (the bug).** `ActiveJobID` set, `Status == indexing`, `LastSuccessfulRun != nil`, collection present. Now eligible. This is the race we are fixing. Covered by Task 1 Test A.
2. **Sibling mid streaming-reindex (in-place delete then insert on the live collection).** Reuse is content-hash keyed, so any row read gives a valid vector for its content; rows mid-delete just miss and get re-embedded. No incorrect vectors. Safe.
3. **Sibling mid staging-rebuild (writes a `_stg` collection, then promotes by rename).** The live collection during staging is the prior full version, so reuse gets the full prior index. The brief rename window resolves through `LoadReuseVectors`' `HasCollection` check (graceful skip). Safe.
4. **Collection missing despite `LastSuccessfulRun != nil` (stale registry).** `LoadReuseVectors` `HasCollection` returns false and skips it. Graceful; the worktree just embeds from scratch that run.
5. **Adopted sibling (`Status == Indexed`, `LastSuccessfulRun == nil`).** Adoption sets `Status = Indexed` without a run record (`manager_adopt.go`). The `Status == Indexed ||` branch keeps it eligible, matching the trigger. Covered by Task 1 Test D.
6. **Sibling never succeeded (first-ever build in progress, no live collection yet).** `Status != Indexed` and `LastSuccessfulRun == nil`, so not eligible. Correct: a first-ever build writes only a staging collection, so there is nothing live to reuse.
7. **Embedding-model mismatch.** `reuseModelMatches` fails, so the sibling is skipped. A vector cannot cross models. Covered by Task 1 Test C.
8. **Multiple eligible siblings.** All eligible collections are returned and loaded, yielding more hits. Cost is more client-side vector loading; acceptable, and unchanged in shape from today. Noted, not optimized here.
9. **Document/conversation codebases are never siblings** (siblings come from `SiblingWorktreeRoots`, which returns filesystem roots, while a conversation codebase has a `chat://` path). The explicit `Kind == Document` skip is defensive and matches the pattern in `manager_paths.go`.
10. **Self-reuse.** The worktree's own root is excluded from `siblings`, so it never reuses from itself.
11. **TS-built sibling.** A sibling collection written by the upstream TS adapter is content-hash compatible, so reuse works across tools.
12. **Files added or removed on the branch.** The worktree's from-scratch bootstrap walks the worktree's current files, so added files are embedded (no reuse hit) and removed files are simply absent (no phantom rows). Reuse only affects the embed-versus-reuse decision per chunk.
13. **Embedder contention (slow NV-EmbedCode-7b).** Out of scope, but the fix reduces the worktree's embed calls to the branch diff, which lowers contention with a concurrent parent sync.
14. **Single-sample timing.** Reuse is still sampled once at bootstrap start. After this fix the only remaining miss is a sibling that has genuinely never produced a usable collection, which is correct.

---

### Task 1: Re-base the reuse eligibility predicate

**Files:**
- Modify: `internal/daemon/manager_worktree.go` (`worktreeSiblingReuseCollections`)
- Test: `internal/daemon/manager_worktree_test.go`

- [ ] **Step 1: Write the failing tests**

Add to `internal/daemon/manager_worktree_test.go`:

```go
// registerSiblingCodebase records a main-worktree codebase for the repo rooted at
// mainRoot with the given liveness fields, so the reuse gate can be unit-tested
// without driving a full concurrent build.
func registerSiblingCodebase(t *testing.T, manager *Manager, mainRoot string, mutate func(*model.Codebase)) {
	t.Helper()
	codebase := newCodebaseRecord(evalSym(t, mainRoot))
	codebase.CollectionName = "cc_repo"
	codebase.EffectiveConfig = defaultIndexConfig()
	mutate(&codebase)
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()
}

// TestWorktreeReuseAcceptsBusyIndexedSibling proves the race fix: a sibling that
// was indexed once but is currently mid-sync (ActiveJobID set, Status indexing)
// is still an eligible reuse source.
func TestWorktreeReuseAcceptsBusyIndexedSibling(t *testing.T) {
	manager, _, _ := newTestManager(t)
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

	registerSiblingCodebase(t, manager, mainRoot, func(c *model.Codebase) {
		c.Status = model.CodebaseStatusIndexing // busy
		c.ActiveJobID = "job-sync-inflight"     // busy
		c.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed"}
	})

	got := manager.worktreeSiblingReuseCollections(evalSym(t, worktreeDir), defaultIndexConfig())
	if len(got) != 1 || got[0] != "cc_repo" {
		t.Fatalf("busy-but-indexed sibling not reused: got %v, want [cc_repo]", got)
	}
}

// TestWorktreeReuseAcceptsAdoptedSibling proves an adopted sibling (Status
// Indexed, no run record) is eligible, matching the auto-create trigger.
func TestWorktreeReuseAcceptsAdoptedSibling(t *testing.T) {
	manager, _, _ := newTestManager(t)
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

	registerSiblingCodebase(t, manager, mainRoot, func(c *model.Codebase) {
		c.Status = model.CodebaseStatusIndexed
		c.LastSuccessfulRun = nil
	})

	got := manager.worktreeSiblingReuseCollections(evalSym(t, worktreeDir), defaultIndexConfig())
	if len(got) != 1 || got[0] != "cc_repo" {
		t.Fatalf("adopted sibling not reused: got %v, want [cc_repo]", got)
	}
}

// TestWorktreeReuseSkipsNeverIndexedSibling proves a sibling that never produced
// a usable collection (not indexed, no run) is not an eligible source.
func TestWorktreeReuseSkipsNeverIndexedSibling(t *testing.T) {
	manager, _, _ := newTestManager(t)
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

	registerSiblingCodebase(t, manager, mainRoot, func(c *model.Codebase) {
		c.Status = model.CodebaseStatusIndexing
		c.LastSuccessfulRun = nil
	})

	got := manager.worktreeSiblingReuseCollections(evalSym(t, worktreeDir), defaultIndexConfig())
	if len(got) != 0 {
		t.Fatalf("never-indexed sibling should not be reused: got %v", got)
	}
}

// TestWorktreeReuseSkipsModelMismatch proves a sibling indexed with a different
// embedding model is not an eligible source.
func TestWorktreeReuseSkipsModelMismatch(t *testing.T) {
	manager, _, _ := newTestManager(t)
	base := t.TempDir()
	mainRoot := filepath.Join(base, "repo")
	makeMainRepo(t, mainRoot)
	worktreeDir := filepath.Join(base, "feature")
	makeLinkedWorktree(t, mainRoot, "feature", worktreeDir, "feature")

	mismatched := defaultIndexConfig()
	mismatched.EmbeddingModel = "some-other-model"
	registerSiblingCodebase(t, manager, mainRoot, func(c *model.Codebase) {
		c.Status = model.CodebaseStatusIndexed
		c.LastSuccessfulRun = &model.IndexRunSummary{IndexedFiles: 1, TotalChunks: 1, Status: "completed"}
		c.EffectiveConfig = mismatched
	})

	got := manager.worktreeSiblingReuseCollections(evalSym(t, worktreeDir), defaultIndexConfig())
	if len(got) != 0 {
		t.Fatalf("model-mismatched sibling should not be reused: got %v", got)
	}
}
```

- [ ] **Step 2: Run the tests to verify the busy/adopted ones fail**

Run: `go test ./internal/daemon/ -run 'TestWorktreeReuse' -v`
Expected: `TestWorktreeReuseAcceptsBusyIndexedSibling` and `TestWorktreeReuseAcceptsAdoptedSibling` FAIL (current gate excludes `ActiveJobID != ""` and requires `Status == Indexed`); the two skip tests PASS.

Note: if `defaultIndexConfig()` leaves embedding fields empty, `TestWorktreeReuseSkipsModelMismatch` still works because it sets a distinct `EmbeddingModel`; the others match because both sides use `defaultIndexConfig()`.

- [ ] **Step 3: Apply the predicate fix**

In `internal/daemon/manager_worktree.go`, replace the body of the per-codebase loop in `worktreeSiblingReuseCollections`:

Replace:

```go
		if codebase.CollectionName == "" || codebase.ActiveJobID != "" {
			continue
		}
		if codebase.Status != model.CodebaseStatusIndexed {
			continue
		}
		if !reuseModelMatches(codebase.EffectiveConfig, indexConfig) {
			continue
		}
		collections = append(collections, codebase.CollectionName)
```

with:

```go
		if codebase.Kind == model.CodebaseKindDocument {
			continue
		}
		if codebase.CollectionName == "" {
			continue
		}
		// Reuse keys on durable facts, not the transient ActiveJobID: a sibling
		// that is currently indexed or has at least one past successful run has a
		// usable collection. An in-flight sync does not drop the live collection,
		// and reuse is content-hash keyed, so reading a mid-sync sibling is safe.
		// This mirrors the auto-create trigger's eligibility so the two agree.
		if codebase.Status != model.CodebaseStatusIndexed && codebase.LastSuccessfulRun == nil {
			continue
		}
		if !reuseModelMatches(codebase.EffectiveConfig, indexConfig) {
			continue
		}
		collections = append(collections, codebase.CollectionName)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/daemon/ -run 'TestWorktreeReuse|TestWorktreeBuildReusesSiblingCollection' -v`
Expected: all PASS, including the existing `TestWorktreeBuildReusesSiblingCollection` integration test.

- [ ] **Step 5: Commit**

```bash
git add internal/daemon/manager_worktree.go internal/daemon/manager_worktree_test.go
git commit -m "Reuse a busy sibling worktree's vectors by keying reuse on durable index facts"
```

---

### Task 2: Full verification

**Files:** none (verification only).

- [ ] **Step 1: Whole suite**

Run: `go test ./...`
Expected: every package `ok`, zero failures. If the daemon package times out under a cold `./...` run, re-run `go test ./internal/daemon/ -count=1` to confirm it passes in isolation (a known cold-build contention effect, not a logic failure).

- [ ] **Step 2: Lint**

Run: `make lint`
Expected: `All checks passed.` If `exhaustruct` flags a `model.Codebase` literal in the new test helper, set the missing fields, or build the record with `newCodebaseRecord` (which the helper already does) and only mutate via the callback.

- [ ] **Step 3: Build**

Run: `make build`
Expected: `All checks passed.` plus `codesign ok`.

- [ ] **Step 4: Commit any verification fixes**

```bash
git add -A
git commit -m "Finalize worktree reuse race fix: lint and build clean"
```

If steps 1-3 needed no changes, skip this commit.

---

## Self-Review

**Spec coverage:** The single behavior (reuse from a busy-but-indexed sibling) is implemented in Task 1 Step 3 and locked by Task 1 Test A; adoption, never-indexed, and model-mismatch are locked by Tests D, the skip test, and C; the existing integration test confirms the settled-sibling path still works.

**Placeholder scan:** No TBD or TODO. Every step shows complete code or an exact command with expected output.

**Type consistency:** `worktreeSiblingReuseCollections(canonicalPath string, indexConfig model.IndexConfig) []string` is unchanged in signature; only its loop body changes. `registerSiblingCodebase` uses `newCodebaseRecord` (returns `model.Codebase`, `Kind = Code`) and `model.IndexRunSummary` fields (`IndexedFiles`, `TotalChunks`, `Status`) consistent with their use in `status_present_test.go`. `reuseModelMatches` and `CodebaseKindDocument` exist (`manager_merge.go`, `model`).

**Scope:** One predicate change plus four unit tests. No proto, schema, or mechanism change. The reuse loader, merkle seeding, and `runBootstrap` wiring are untouched.
