# Implementation Plan: Caller-Relative Paths and a Single Discovery Walk

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `codebase index` resolve relative paths against the caller's cwd, reply in milliseconds, scan each tree exactly once per job, and behave sanely under `--wait` and Ctrl-C.

**Architecture:** The CLI and MCP adapter send the caller's working directory in `ClientInfo.caller_cwd` (already on the wire; commit `212843d`). The daemon joins relative paths against it at the gRPC boundary and refuses relative paths without it. Registration stops scanning the tree. The background job's single-pass discovery walk produces both the file list and the ignore-rule tree, and persists the rules onto the codebase record. The watcher reuses persisted rules. The CLI gains `--wait <duration>` (bounded, never indefinite) that streams progress from `WatchJobs`.

**Tech Stack:** Go, gRPC + protobuf (buf, already regenerated), cobra, rjeczalik/notify.

**Spec:** `docs/superpowers/specs/2026-06-10-index-cwd-and-single-walk-design.md`

**Worktree:** `/Users/agoodkind/Sites/lm-semantic-search/.claude/worktrees/relative-path-cwd`, branch `fix/relative-path-cwd`. Run all commands from this directory.

**Verification gates for every task:** the named test command, then at the end `go test ./...`, `make lint`, `make build`.

---

### Task 1: `canonicalizePath` rejects relative paths and URIs

The daemon's `filepath.Abs` resolves relative paths against the daemon's own cwd (`/` under launchd). That is never the caller's cwd, so a relative path reaching this function is a bug upstream. The `://` URI rejection exists in the user's main checkout working tree only; this task brings the same check onto this branch.

**Files:**
- Modify: `internal/daemon/manager_paths.go:134-158` (`canonicalizePath`)
- Test: `internal/daemon/required_args_test.go` (existing `TestCanonicalizePath*` tests live here)

- [ ] **Step 1: Write the failing tests**

Add to `internal/daemon/required_args_test.go`:

```go
func TestCanonicalizePathRejectsRelative(t *testing.T) {
	t.Parallel()

	for _, requested := range []string{".", "..", "some/relative/dir", "./x"} {
		if _, err := canonicalizePath(requested); err == nil {
			t.Fatalf("canonicalizePath(%q) returned nil error; a relative path must not resolve against the daemon cwd", requested)
		}
	}
}

func TestCanonicalizePathRejectsURI(t *testing.T) {
	t.Parallel()

	if _, err := canonicalizePath("chat:///clyde-conversations"); err == nil {
		t.Fatal("canonicalizePath accepted a URI-shaped path")
	}
}
```

- [ ] **Step 2: Run the tests, confirm they fail**

Run: `go test ./internal/daemon/ -run 'TestCanonicalizePathRejects' -v`
Expected: FAIL (both new tests; relative paths currently resolve via `filepath.Abs`).

- [ ] **Step 3: Implement**

In `canonicalizePath`, replace the `filepath.Abs` call with rejection plus `filepath.Clean`:

```go
func canonicalizePath(requestedPath string) (string, error) {
	// Reject an empty or whitespace path early; "" must never silently
	// resolve to any directory.
	if strings.TrimSpace(requestedPath) == "" {
		return "", errors.New("codebase path is required")
	}
	if strings.Contains(requestedPath, "://") {
		return "", fmt.Errorf("path %q looks like a URI; pass a filesystem directory instead", requestedPath)
	}
	// A relative path reaching the daemon is unresolvable here: the daemon's
	// working directory is never the caller's. resolveRequestPath at the gRPC
	// boundary joins relative paths against the caller's cwd before this point.
	if !filepath.IsAbs(requestedPath) {
		return "", fmt.Errorf("path %q is relative; pass an absolute path or send caller_cwd", requestedPath)
	}
	absolutePath := filepath.Clean(requestedPath)
	canonicalPath, err := filepath.EvalSymlinks(absolutePath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return absolutePath, nil
		}
		slog.Error("resolve symlinks failed", "path", absolutePath, "err", err)
		return "", fmt.Errorf("resolve symlinks for %s: %w", absolutePath, err)
	}
	return canonicalPath, nil
}
```

Note: the existing `TestCanonicalizePathAcceptsNonEmpty` passes a `t.TempDir()`, which is absolute, so it keeps passing.

- [ ] **Step 4: Run the daemon package tests**

Run: `go test ./internal/daemon/ -run 'TestCanonicalizePath' -v`
Expected: PASS (all canonicalizePath tests).

Then run the whole package to catch callers that fed relative paths in tests:
Run: `go test ./internal/daemon/`
Expected: PASS. If a test fails because it passed a relative path, fix the test to pass an absolute path (that was the bug this guards against).

- [ ] **Step 5: Commit**

```bash
git add internal/daemon/manager_paths.go internal/daemon/required_args_test.go
git commit -m "Reject relative and URI paths in canonicalizePath"
```

---

### Task 2: Join relative request paths against `caller_cwd` at the gRPC boundary

**Files:**
- Modify: `internal/daemon/manager_paths.go` (new function `resolveRequestPath`)
- Modify: `internal/daemon/grpc_server.go` (handlers `StartIndex`, `SyncIndex`, `ClearIndex`, `GetIndex`, `SearchCode`)
- Test: Create `internal/daemon/manager_paths_test.go`

- [ ] **Step 1: Write the failing test**

Create `internal/daemon/manager_paths_test.go`:

```go
package daemon

import (
	"path/filepath"
	"testing"
)

func TestResolveRequestPathJoinsRelativeAgainstCallerCwd(t *testing.T) {
	t.Parallel()

	got, err := resolveRequestPath("sub/dir", "/Users/example/repo")
	if err != nil {
		t.Fatalf("resolveRequestPath returned error: %v", err)
	}
	want := filepath.Join("/Users/example/repo", "sub/dir")
	if got != want {
		t.Fatalf("resolveRequestPath = %q, want %q", got, want)
	}
}

func TestResolveRequestPathResolvesDotToCallerCwd(t *testing.T) {
	t.Parallel()

	got, err := resolveRequestPath(".", "/Users/example/repo")
	if err != nil {
		t.Fatalf("resolveRequestPath returned error: %v", err)
	}
	if got != "/Users/example/repo" {
		t.Fatalf("resolveRequestPath = %q, want /Users/example/repo", got)
	}
}

func TestResolveRequestPathRejectsRelativeWithoutCallerCwd(t *testing.T) {
	t.Parallel()

	for _, callerCwd := range []string{"", "   ", "not/absolute"} {
		if _, err := resolveRequestPath(".", callerCwd); err == nil {
			t.Fatalf("resolveRequestPath(%q, %q) returned nil error", ".", callerCwd)
		}
	}
}

func TestResolveRequestPathPassesThroughAbsoluteIDAndURI(t *testing.T) {
	t.Parallel()

	cases := []string{"/abs/path", "cb_123_abc", "chat:///clyde-conversations"}
	for _, requested := range cases {
		got, err := resolveRequestPath(requested, "/Users/example/repo")
		if err != nil {
			t.Fatalf("resolveRequestPath(%q) returned error: %v", requested, err)
		}
		if got != requested {
			t.Fatalf("resolveRequestPath(%q) = %q, want pass-through", requested, got)
		}
	}
}
```

- [ ] **Step 2: Run, confirm compile failure**

Run: `go test ./internal/daemon/ -run 'TestResolveRequestPath' -v`
Expected: FAIL with `undefined: resolveRequestPath`.

- [ ] **Step 3: Implement the function**

Add to `internal/daemon/manager_paths.go`, directly above `canonicalizePath`:

```go
// resolveRequestPath makes a relative request path absolute using the
// caller's working directory, carried in ClientInfo.caller_cwd. Absolute
// paths, codebase ids, and URI-shaped arguments pass through unchanged for
// the later resolution stages to classify. A relative path with no absolute
// caller cwd is rejected: the daemon's own working directory is never the
// caller's, so resolving against it silently is never correct.
func resolveRequestPath(requestedPath string, callerCwd string) (string, error) {
	trimmed := strings.TrimSpace(requestedPath)
	if trimmed == "" || looksLikeCodebaseID(trimmed) || strings.Contains(trimmed, "://") {
		return requestedPath, nil
	}
	if filepath.IsAbs(trimmed) {
		return trimmed, nil
	}
	cwd := strings.TrimSpace(callerCwd)
	if !filepath.IsAbs(cwd) {
		return "", fmt.Errorf("path %q is relative and the request carries no absolute caller working directory; pass an absolute path or upgrade the client", requestedPath)
	}
	return filepath.Join(cwd, trimmed), nil
}
```

- [ ] **Step 4: Run the unit tests**

Run: `go test ./internal/daemon/ -run 'TestResolveRequestPath' -v`
Expected: PASS (all four).

- [ ] **Step 5: Wire the five path-carrying handlers**

In `internal/daemon/grpc_server.go`, each of `StartIndex`, `SyncIndex`, `ClearIndex`, `GetIndex`, and `SearchCode` currently passes `request.GetPath()` into the manager. After the existing `requireNonEmpty` check, insert the join and use its result everywhere the handler used `request.GetPath()` (including `startIndexMergeNote` and `DisplayText` rendering in `StartIndex`):

```go
requestedPath, pathErr := resolveRequestPath(request.GetPath(), request.GetClient().GetCallerCwd())
if pathErr != nil {
	return nil, status.Error(adapterr.Respond(ctx, adapterr.NewInvalidPath(pathErr.Error(), pathErr)))
}
```

`GetIndexRequest` and `SearchCodeRequest` have a `client` field as of commit `212843d`, so `request.GetClient()` compiles on all five. `GetClient()` returns nil for old clients and `GetCallerCwd()` on a nil receiver returns "". That empty value only matters in the reject-relative branch, so absolute paths from old clients pass untouched.

- [ ] **Step 6: Build and run the package tests**

Run: `go build ./... && go test ./internal/daemon/`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add internal/daemon/manager_paths.go internal/daemon/manager_paths_test.go internal/daemon/grpc_server.go
git commit -m "Resolve relative request paths against caller_cwd at the gRPC boundary"
```

---

### Task 3: Refuse to register the filesystem root

**Files:**
- Modify: `internal/daemon/manager_guards.go` (new guard)
- Modify: `internal/daemon/manager.go:398-403` (wire into `StartIndex` next to `guardStateRoot`)
- Test: Create `internal/daemon/manager_guards_test.go`

- [ ] **Step 1: Write the failing test**

Create `internal/daemon/manager_guards_test.go`:

```go
package daemon

import "testing"

func TestGuardFilesystemRootRejectsRoot(t *testing.T) {
	t.Parallel()

	manager := &Manager{}
	if err := manager.guardFilesystemRoot("/"); err == nil {
		t.Fatal("guardFilesystemRoot accepted the filesystem root")
	}
	if err := manager.guardFilesystemRoot("/Users/example/repo"); err != nil {
		t.Fatalf("guardFilesystemRoot rejected a normal path: %v", err)
	}
}
```

- [ ] **Step 2: Run, confirm compile failure**

Run: `go test ./internal/daemon/ -run 'TestGuardFilesystemRoot' -v`
Expected: FAIL with `undefined` method.

- [ ] **Step 3: Implement**

Add to `internal/daemon/manager_guards.go` after `guardStateRoot`:

```go
// guardFilesystemRoot rejects a registration rooted at the filesystem root.
// Indexing "/" swallows every mount on the host and is never intentional;
// the daemon-resolved-relative-path incident registered "/" exactly this way.
func (manager *Manager) guardFilesystemRoot(canonicalPath string) error {
	_ = manager
	if filepath.Clean(canonicalPath) == string(filepath.Separator) {
		err := fmt.Errorf("refusing to index filesystem root %s", canonicalPath)
		slog.Error("filesystem-root guard rejected registration", "path", canonicalPath, "err", err)
		return err
	}
	return nil
}
```

In `internal/daemon/manager.go` `StartIndex`, after the `guardStateRoot` call (line 398) add:

```go
if err := manager.guardFilesystemRoot(canonicalPath); err != nil {
	return emptyJob, emptyCodebase, false, "", err
}
```

- [ ] **Step 4: Run the tests**

Run: `go test ./internal/daemon/ -run 'TestGuardFilesystemRoot' -v && go test ./internal/daemon/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/daemon/manager_guards.go internal/daemon/manager_guards_test.go internal/daemon/manager.go
git commit -m "Add filesystem-root guard to StartIndex registration"
```

---

### Task 4: Single-pass discovery walk that returns the rule tree

Today `Discover` (`internal/discovery/discovery.go:234`) calls `EffectiveIgnorePatterns` (one full tree walk via `walkGitignore`) and then `walkFiles` (a second full walk). This task merges them: one walk reads each directory's ignore files on entry, evaluates children against the rules collected so far, and returns files plus the rule tree.

Ordering invariant: the root node's pattern order must stay `builtin, override, root ignore files, global`, because `PathIgnored` is last-match-wins within a node and `EffectiveIgnorePatterns` produces that order. All three root segments are known at walk start, so the combined walk assembles the root node when it enters the root directory, before judging any child.

**Files:**
- Modify: `internal/discovery/discovery.go` (`Result`, `Discover`, new `combinedWalker`; delete `walkFiles`; keep `walkGitignore` for `EffectiveIgnorePatterns`)
- Test: `internal/discovery/discovery_test.go`

- [ ] **Step 1: Write the failing tests**

Add to `internal/discovery/discovery_test.go`:

```go
// TestDiscoverReturnsRulesMatchingEffectiveIgnorePatterns proves the
// single-pass walk produces verdicts identical to the standalone rules
// resolver for the same tree.
func TestDiscoverReturnsRulesMatchingEffectiveIgnorePatterns(t *testing.T) {
	t.Parallel()
	tempDir := t.TempDir()
	writeFile(t, tempDir, ".gitignore", "ignored-dir/\n*.tmp\n")
	mkdir(t, tempDir, "ignored-dir")
	writeFile(t, tempDir, "ignored-dir/inside.go", "package x\n")
	mkdir(t, tempDir, "nested")
	writeFile(t, tempDir, "nested/.gitignore", "local-only.go\n")
	writeFile(t, tempDir, "nested/local-only.go", "package x\n")
	writeFile(t, tempDir, "nested/kept.go", "package x\n")
	writeFile(t, tempDir, "kept.go", "package x\n")
	writeFile(t, tempDir, "scratch.tmp", "x\n")

	result, err := Discover(context.Background(), tempDir, nil, nil)
	if err != nil {
		t.Fatalf("Discover returned error: %v", err)
	}
	standalone, err := EffectiveIgnorePatterns(context.Background(), tempDir, nil)
	if err != nil {
		t.Fatalf("EffectiveIgnorePatterns returned error: %v", err)
	}

	candidates := []string{
		"kept.go", "scratch.tmp", "ignored-dir/inside.go",
		"nested/local-only.go", "nested/kept.go",
	}
	for _, candidate := range candidates {
		fromWalk, _, _ := PathIgnored(candidate, result.Rules)
		fromStandalone, _, _ := PathIgnored(candidate, standalone)
		if fromWalk != fromStandalone {
			t.Fatalf("verdict for %q diverged: walk=%v standalone=%v", candidate, fromWalk, fromStandalone)
		}
	}

	wantFiles := []string{
		filepath.Join(tempDir, ".gitignore"),
		filepath.Join(tempDir, "kept.go"),
		filepath.Join(tempDir, "nested", ".gitignore"),
		filepath.Join(tempDir, "nested", "kept.go"),
	}
	if !slices.Equal(result.Files, wantFiles) {
		t.Fatalf("Files = %v, want %v", result.Files, wantFiles)
	}
}

// TestDiscoverDoesNotDescendIntoIgnoredDirectories proves the walk prunes:
// an unreadable directory that the rules exclude must not fail discovery.
func TestDiscoverDoesNotDescendIntoIgnoredDirectories(t *testing.T) {
	t.Parallel()
	tempDir := t.TempDir()
	writeFile(t, tempDir, ".gitignore", "sealed/\n")
	mkdir(t, tempDir, "sealed")
	writeFile(t, tempDir, "sealed/file.go", "package x\n")
	if err := os.Chmod(filepath.Join(tempDir, "sealed"), 0o000); err != nil {
		t.Fatalf("chmod: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(filepath.Join(tempDir, "sealed"), 0o755) })

	if _, err := Discover(context.Background(), tempDir, nil, nil); err != nil {
		t.Fatalf("Discover failed on a pruned unreadable directory: %v", err)
	}
}
```

If `writeFile`/`mkdir` helpers do not exist in `discovery_test.go`, add them:

```go
func writeFile(t *testing.T, root string, relative string, content string) {
	t.Helper()
	full := filepath.Join(root, relative)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatalf("mkdir for %s: %v", relative, err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", relative, err)
	}
}

func mkdir(t *testing.T, root string, relative string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(root, relative), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", relative, err)
	}
}
```

(Check the existing test file first; it may already have equivalents with different names. Reuse what exists.)

- [ ] **Step 2: Run, confirm failure**

Run: `go test ./internal/discovery/ -run 'TestDiscover' -v`
Expected: FAIL. `result.Rules` does not compile because `Result` has no `Rules` field. The compile error comes before any assertion runs.

- [ ] **Step 3: Implement**

In `internal/discovery/discovery.go`:

1. Add `Rules IgnoreRules` to `Result` (line 160).

2. Replace the body of `Discover` (lines 234-265):

```go
// Discover walks a codebase root once. The walk reads each directory's
// ignore files as it enters the directory and applies the rules collected
// so far, so it never descends into an ignored directory. It returns the
// surviving files together with the finished rule tree, which callers
// persist so no second walk is needed.
func Discover(ctx context.Context, root string, additionalIgnorePatterns []string, additionalExtensions []string) (Result, error) {
	absoluteRoot, err := filepath.Abs(root)
	if err != nil {
		slog.ErrorContext(ctx, "resolve absolute root failed", "root", root, "err", err)
		return Result{}, fmt.Errorf("resolve absolute root %s: %w", root, err)
	}

	rootPatterns := make([]ignorePattern, 0, len(defaultIgnorePatterns)+len(additionalIgnorePatterns))
	for _, pattern := range defaultIgnorePatterns {
		rootPatterns = append(rootPatterns, parsePattern(pattern, patternSourceBuiltin))
	}
	for _, pattern := range additionalIgnorePatterns {
		rootPatterns = append(rootPatterns, parsePattern(pattern, patternSourceOverride))
	}
	globalPatterns, globalErr := loadGlobalContextIgnore(ctx)
	if globalErr != nil {
		globalPatterns = nil
	}

	rootCommonDir, _ := gitworktree.CommonDirAt(absoluteRoot)

	files := []string{}
	walker := &combinedWalker{
		root:           absoluteRoot,
		rootCommonDir:  rootCommonDir,
		rootPatterns:   rootPatterns,
		globalPatterns: globalPatterns,
		rules:          IgnoreRules{Nodes: map[string][]ignorePattern{}},
		files:          &files,
	}
	if err := walker.walk(ctx, ""); err != nil {
		return Result{}, err
	}
	slices.Sort(files)

	return Result{
		Files:          files,
		IgnorePatterns: walker.rules.Flatten(),
		Extensions:     normalizeExtensions(additionalExtensions),
		Rules:          walker.rules,
	}, nil
}
```

3. Add the walker where `walkFiles` was (lines 375-418), replacing it:

```go
// combinedWalker is the one-pass descent that powers Discover: rules are
// collected and applied in the same walk that lists files.
type combinedWalker struct {
	root           string
	rootCommonDir  string
	rootPatterns   []ignorePattern
	globalPatterns []ignorePattern
	rules          IgnoreRules
	files          *[]string
}

func (walker *combinedWalker) walk(ctx context.Context, relativeDir string) error {
	if err := ctx.Err(); err != nil {
		slog.ErrorContext(ctx, "walk cancelled", "path", relativeDir, "err", err)
		return fmt.Errorf("walk cancelled at %s: %w", relativeDir, err)
	}
	currentDir := filepath.Join(walker.root, relativeDir)
	entries, err := os.ReadDir(currentDir)
	if err != nil {
		slog.ErrorContext(ctx, "read directory failed", "path", currentDir, "err", err)
		return fmt.Errorf("read directory %s: %w", currentDir, err)
	}

	// Read this directory's ignore files before judging any child, so the
	// rules that govern the children exist when the children are evaluated.
	// The root node keeps EffectiveIgnorePatterns's order (builtin,
	// override, root ignore files, global) because PathIgnored is
	// last-match-wins within a node.
	parsed, err := walker.readIgnoreEntries(ctx, currentDir, relativeDir, entries)
	if err != nil {
		return err
	}
	if relativeDir == "" {
		merged := make([]ignorePattern, 0, len(walker.rootPatterns)+len(parsed)+len(walker.globalPatterns))
		merged = append(merged, walker.rootPatterns...)
		merged = append(merged, parsed...)
		merged = append(merged, walker.globalPatterns...)
		walker.rules.Nodes[""] = merged
	} else if len(parsed) > 0 {
		walker.rules.Nodes[relativeDir] = parsed
	}

	for _, entry := range entries {
		// The codebase root's own .git entry is metadata, never content. The
		// .git/ pattern already excludes the main worktree's .git directory,
		// but a linked worktree roots at a .git file that the directory
		// pattern does not match, so it is dropped explicitly here.
		if entry.Name() == ".git" && relativeDir == "" {
			continue
		}
		childRelative := entry.Name()
		if relativeDir != "" {
			childRelative = relativeDir + "/" + entry.Name()
		}
		if excluded, _, _ := PathIgnored(childRelative, walker.rules); excluded {
			continue
		}
		fullPath := filepath.Join(currentDir, entry.Name())
		if entry.IsDir() {
			if isSameRepoWorktree(fullPath, walker.rootCommonDir) {
				continue
			}
			if err := walker.walk(ctx, childRelative); err != nil {
				return err
			}
			continue
		}
		*walker.files = append(*walker.files, fullPath)
	}
	return nil
}

// readIgnoreEntries parses every dot-prefixed *ignore file in the directory
// listing, mirroring walkGitignore's name predicate and source labeling.
func (walker *combinedWalker) readIgnoreEntries(ctx context.Context, currentDir string, relativeDir string, entries []os.DirEntry) ([]ignorePattern, error) {
	parsed := []ignorePattern{}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasPrefix(name, ".") || !strings.HasSuffix(name, "ignore") {
			continue
		}
		raw, readErr := readIgnoreFile(ctx, filepath.Join(currentDir, name))
		if readErr != nil {
			return nil, readErr
		}
		source := name
		if relativeDir != "" {
			source = filepath.ToSlash(filepath.Join(relativeDir, name))
		}
		for _, pattern := range raw {
			parsed = append(parsed, parsePattern(pattern, source))
		}
	}
	return parsed, nil
}
```

4. Delete the old `walkFiles` function. Keep `walkGitignore`, `isDefaultExcludedDir`, and `EffectiveIgnorePatterns` untouched; the status path still uses them for rules-only resolution.

- [ ] **Step 4: Run the discovery tests**

Run: `go test ./internal/discovery/ -v`
Expected: PASS, including every pre-existing `Discover` and `EffectiveIgnorePatterns` test. A pre-existing test failure here means a behavior regression in the walk; fix the walk, not the test.

- [ ] **Step 5: Commit**

```bash
git add internal/discovery/discovery.go internal/discovery/discovery_test.go
git commit -m "Merge ignore-rule collection into the discovery file walk"
```

---

### Task 5: The job persists the walk's rule tree onto the codebase record

**Files:**
- Modify: `internal/merkle/snapshot.go:126-139` (`Capture` returns the rule tree)
- Modify: `internal/daemon/item_source.go:41-58` (`codeItemSource` gains an `onRules` callback)
- Modify: `internal/daemon/manager_runner.go:85` (wire the callback to `cacheResolvedRules`)
- Modify: `internal/daemon/background_sync.go:333` (interval sync also persists)
- Modify: `internal/daemon/converge_concurrency_test.go:407`, `internal/daemon/manager_bootstrap_test.go:93` (new return value)
- Test: `internal/daemon/item_source_rules_test.go` (new)

- [ ] **Step 1: Write the failing test**

Create `internal/daemon/item_source_rules_test.go`:

```go
package daemon

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/discovery"
	"goodkind.io/lm-semantic-search/internal/model"
)

// TestCodeItemSourceCaptureReportsRules proves one capture both snapshots the
// tree and hands the resolved ignore rules to the registered callback, so the
// daemon can persist them without a second walk.
func TestCodeItemSourceCaptureReportsRules(t *testing.T) {
	t.Parallel()
	tempDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(tempDir, ".gitignore"), []byte("skipped/\n"), 0o644); err != nil {
		t.Fatalf("write .gitignore: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tempDir, "kept.go"), []byte("package x\n"), 0o644); err != nil {
		t.Fatalf("write kept.go: %v", err)
	}

	var reported discovery.IgnoreRules
	source := newCodeItemSource(nil, tempDir, model.IndexConfig{}, func(rules discovery.IgnoreRules) {
		reported = rules
	})

	if _, err := source.capture(context.Background()); err != nil {
		t.Fatalf("capture returned error: %v", err)
	}
	if reported.IsEmpty() {
		t.Fatal("capture did not report the resolved ignore rules")
	}
	if excluded, _, _ := discovery.PathIgnored("skipped/file.go", reported); !excluded {
		t.Fatal("reported rules do not contain the root .gitignore pattern")
	}
}
```

- [ ] **Step 2: Run, confirm compile failure**

Run: `go test ./internal/daemon/ -run 'TestCodeItemSourceCaptureReportsRules' -v`
Expected: FAIL: `newCodeItemSource` takes three arguments today.

- [ ] **Step 3: Implement**

1. In `internal/merkle/snapshot.go`, change `Capture` to return the rule tree:

```go
// Capture walks a codebase once and records content hashes for the tracked
// files. The returned rule tree is the walk's resolved ignore rules, handed
// back so the caller can persist them instead of re-walking.
func Capture(
	ctx context.Context,
	root string,
	indexConfig model.IndexConfig,
) (Snapshot, discovery.IgnoreRules, error) {
	discoveryResult, err := discovery.Discover(
		ctx,
		root,
		indexConfig.IgnorePatterns,
		indexConfig.Extensions,
	)
	if err != nil {
		return Snapshot{}, discovery.IgnoreRules{}, fmt.Errorf("discover sync files under %s: %w", root, err)
	}
	...
```

Every `return Snapshot{}, err` inside becomes `return Snapshot{}, discovery.IgnoreRules{}, err`. The final success return becomes `return snapshot, discoveryResult.Rules, nil` (adapt to the actual local variable names at the end of the function). Add the `discovery` import.

2. In `internal/daemon/item_source.go`, extend `codeItemSource`:

```go
type codeItemSource struct {
	runner        indexingRunner
	canonicalPath string
	config        model.IndexConfig
	// onRules receives the walk's resolved ignore rules each capture, so the
	// manager can persist them without a second walk. Nil disables reporting.
	onRules func(discovery.IgnoreRules)
}

func newCodeItemSource(runner indexingRunner, canonicalPath string, config model.IndexConfig, onRules func(discovery.IgnoreRules)) codeItemSource {
	return codeItemSource{runner: runner, canonicalPath: canonicalPath, config: config, onRules: onRules}
}

func (source codeItemSource) capture(ctx context.Context) (merkle.Snapshot, error) {
	snapshot, rules, err := merkle.Capture(ctx, source.canonicalPath, source.config)
	if err != nil {
		slog.ErrorContext(ctx, "capture code snapshot failed", "path", source.canonicalPath, "err", err)
		return merkle.Snapshot{}, fmt.Errorf("capture code snapshot for %s: %w", source.canonicalPath, err)
	}
	if source.onRules != nil {
		source.onRules(rules)
	}
	return snapshot, nil
}
```

Add the `discovery` import.

3. In `internal/daemon/manager_runner.go:85`, wire the callback:

```go
codeSource := newCodeItemSource(manager.runner, job.CanonicalPath, job.Config, func(rules discovery.IgnoreRules) {
	manager.cacheResolvedRules(job.CodebaseID, rules)
})
```

`cacheResolvedRules` already exists at `internal/daemon/manager_status.go:158`. Add the `discovery` import.

4. In `internal/daemon/background_sync.go:333`, the interval sync's `merkle.Capture` call gains the rules return. Persist it the same way, using the codebase variable in scope there:

```go
currentSnapshot, rules, err := merkle.Capture(...)
...
manager.cacheResolvedRules(codebase.ID, rules)
```

(Read the surrounding function for the exact codebase variable name; persist after the error check.)

5. Update the two test callers (`converge_concurrency_test.go:407`, `manager_bootstrap_test.go:93`) to `captured, _, err := merkle.Capture(...)`.

- [ ] **Step 4: Run the tests**

Run: `go test ./internal/daemon/ ./internal/merkle/ ./internal/indexer/`
Expected: PASS, including the new test.

- [ ] **Step 5: Commit**

```bash
git add internal/merkle/snapshot.go internal/daemon/item_source.go internal/daemon/item_source_rules_test.go internal/daemon/manager_runner.go internal/daemon/background_sync.go internal/daemon/converge_concurrency_test.go internal/daemon/manager_bootstrap_test.go
git commit -m "Persist the discovery walk's ignore rules from each capture"
```

---

### Task 6: Registration, adoption, and the watcher stop walking the tree

**Files:**
- Modify: `internal/daemon/manager.go:491-492` (StartIndex commit), `manager.go:556` (SyncIndex), `manager.go:697-708` (delete `resolveIgnoreRulesOrLog`)
- Modify: `internal/daemon/manager_adopt.go:48`
- Modify: `internal/daemon/watcher.go:87-104`
- Test: `internal/daemon/watcher_test.go`

- [ ] **Step 1: Write the failing test**

Add to `internal/daemon/watcher_test.go` (mirror the setup of `TestWatcherAddCodebaseIsIdempotent` at line 30):

```go
// TestWatcherAddCodebaseDoesNotResolveRules proves AddCodebase never scans
// the tree itself: a codebase whose record holds no rules is registered with
// empty rules even when a .gitignore exists on disk.
func TestWatcherAddCodebaseDoesNotResolveRules(t *testing.T) {
	t.Parallel()
	manager, _, _ := newTestManager(t)
	queue := NewEventQueue(time.Millisecond, func(_ string, _ []string) {})
	watcher := NewWatcher(manager, queue)

	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte("ignored/\n"), 0o644); err != nil {
		t.Fatalf("write .gitignore: %v", err)
	}
	codebase := model.Codebase{
		ID:                  "cb_test_no_resolve",
		CanonicalPath:       root,
		ResolvedIgnoreRules: discovery.IgnoreRules{Nodes: nil},
	}
	watcher.AddCodebase(context.Background(), codebase)

	for _, watchRoot := range watcher.snapshotRoots() {
		if watchRoot.codebaseID == codebase.ID && !watchRoot.rules.IsEmpty() {
			t.Fatal("AddCodebase resolved rules from disk; it must reuse the record")
		}
	}
}
```

- [ ] **Step 2: Run, confirm failure**

Run: `go test ./internal/daemon/ -run 'TestWatcherAddCodebaseDoesNotResolveRules' -v`
Expected: FAIL: the watcher's lazy re-resolve fills rules from the on-disk `.gitignore`.

- [ ] **Step 3: Implement**

1. In `internal/daemon/watcher.go`, delete the re-resolve block (lines 92-104) so `AddCodebase` reads:

```go
func (watcher *Watcher) AddCodebase(ctx context.Context, codebase model.Codebase) {
	if codebase.Kind == model.CodebaseKindDocument {
		return
	}

	// The record's rules are the walk-resolved cache the jobs maintain.
	// Empty rules mean no capture has run yet; dispatch passes a few extra
	// events until the first capture persists rules, and converge drops them.
	rules := codebase.ResolvedIgnoreRules

	watcher.mu.Lock()
	...
```

Remove the now-unused `discovery` import if nothing else in the file uses it.

2. In `internal/daemon/manager.go`, delete line 492 (`codebase.ResolvedIgnoreRules = resolveIgnoreRulesOrLog(...)`) in `commitStartIndexLocked` and line 556 (same call) in `SyncIndex`. The codebase record keeps whatever rules it already carries.

3. In `internal/daemon/manager_adopt.go`, delete line 48 (`record.ResolvedIgnoreRules = resolveIgnoreRulesOrLog(...)`). Adoption already enqueues a refresh sync whose capture persists rules.

4. In `internal/daemon/manager.go:697-708`, delete `resolveIgnoreRulesOrLog`; nothing calls it now and the deadcode gate would flag it.

- [ ] **Step 4: Run the tests**

Run: `go test ./internal/daemon/`
Expected: PASS, including the new watcher test and all existing watcher and registration tests.

- [ ] **Step 5: Commit**

```bash
git add internal/daemon/watcher.go internal/daemon/watcher_test.go internal/daemon/manager.go internal/daemon/manager_adopt.go
git commit -m "Drop registration, adoption, and watcher ignore-rule walks"
```

---

### Task 7: Watcher registration leaves the reply path

**Files:**
- Modify: `internal/daemon/manager.go:437` (StartIndex), `internal/daemon/manager_adopt.go:63` (adoption)

- [ ] **Step 1: Implement**

Both call sites currently run `manager.notifyCodebaseAdded(ctx, ...)` synchronously inside an RPC handler. Detach them with the same context pattern `runJobAsync` uses (`internal/daemon/manager_runner.go:14-16`).

In `manager.go` `StartIndex` (line 437):

```go
notifyCtx := correlation.WithContext(context.WithoutCancel(ctx), correlation.FromContext(ctx).Child())
go manager.notifyCodebaseAdded(notifyCtx, codebase)
```

In `manager_adopt.go` (line 63), add the same two lines with that function's context and record variable.

`AddCodebase` is idempotent per codebase id (locked check in `watcher.go:107`), so a duplicate fire is harmless.

- [ ] **Step 2: Run the daemon tests with the race detector**

Run: `go test -race ./internal/daemon/`
Expected: PASS, no data races. The existing idempotency and adoption tests cover the behavior; the race detector covers the new concurrency.

- [ ] **Step 3: Commit**

```bash
git add internal/daemon/manager.go internal/daemon/manager_adopt.go
git commit -m "Run watcher lifecycle notification off the RPC reply path"
```

---

### Task 8: The CLI sends its working directory on every path command

**Files:**
- Modify: `cmd/lm-semantic-search/rpc.go:24-33` (`currentClientInfo`)
- Modify: `cmd/lm-semantic-search/codebase.go` (`status` and `search` RunE pass `ClientInfo`)
- Test: `cmd/lm-semantic-search/rpc_test.go` (new)

- [ ] **Step 1: Write the failing test**

Create `cmd/lm-semantic-search/rpc_test.go`:

```go
package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCurrentClientInfoCarriesCallerCwd(t *testing.T) {
	info, err := currentClientInfo()
	if err != nil {
		t.Fatalf("currentClientInfo returned error: %v", err)
	}
	if info.GetCallerCwd() == "" {
		t.Fatal("currentClientInfo did not set caller_cwd")
	}
	if !filepath.IsAbs(info.GetCallerCwd()) {
		t.Fatalf("caller_cwd %q is not absolute", info.GetCallerCwd())
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if info.GetCallerCwd() != wd {
		t.Fatalf("caller_cwd = %q, want %q", info.GetCallerCwd(), wd)
	}
}
```

- [ ] **Step 2: Run, confirm failure**

Run: `go test ./cmd/lm-semantic-search/ -run 'TestCurrentClientInfoCarriesCallerCwd' -v`
Expected: FAIL: `caller_cwd` is empty.

- [ ] **Step 3: Implement**

`cmd/lm-semantic-search/rpc.go`:

```go
func currentClientInfo() (*pb.ClientInfo, error) {
	pid := os.Getpid()
	if pid < 0 || pid > math.MaxInt32 {
		return nil, fmt.Errorf("process id %d does not fit in int32", pid)
	}
	workingDir, err := os.Getwd()
	if err != nil {
		return nil, fmt.Errorf("resolve working directory: %w", err)
	}
	return &pb.ClientInfo{
		Name:      "cli",
		Pid:       int32(pid),
		CallerCwd: workingDir,
	}, nil
}
```

In `cmd/lm-semantic-search/codebase.go`, `newCodebaseStatusCmd` and `newCodebaseSearchCmd` RunE bodies currently build requests without `ClientInfo`. Add it, following the exact shape `index`/`sync`/`clear` already use:

```go
RunE: func(cmd *cobra.Command, args []string) error {
	clientInfo, err := currentClientInfo()
	if err != nil {
		return err
	}
	return callAndPrint(options.cliOptions(), func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
		return client.GetIndex(ctx, &pb.GetIndexRequest{Path: args[0], Client: clientInfo})
	})
},
```

For search, the request becomes `&pb.SearchCodeRequest{Path: args[0], Query: args[1], Limit: searchLimit, ExtensionFilter: extensions, Client: clientInfo}`.

- [ ] **Step 4: Run the CLI tests**

Run: `go test ./cmd/lm-semantic-search/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cmd/lm-semantic-search/rpc.go cmd/lm-semantic-search/rpc_test.go cmd/lm-semantic-search/codebase.go
git commit -m "Send caller_cwd from the CLI on every path command"
```

---

### Task 9: The MCP adapter sends its working directory

**Files:**
- Modify: `internal/mcpserver/server.go:136`, `server.go:167` (existing `ClientInfo` literals), `server.go:179-187` (status tool), `server.go:281-297` (search tool)

- [ ] **Step 1: Implement**

Add one helper near the other small helpers in `server.go`:

```go
// mcpClientInfo identifies this adapter to the daemon. caller_cwd lets the
// daemon resolve a relative tool path against the adapter's working
// directory, which the editor sets to the project root.
func mcpClientInfo() *pb.ClientInfo {
	workingDir, err := os.Getwd()
	if err != nil {
		workingDir = ""
	}
	return &pb.ClientInfo{Name: "mcp", CallerCwd: workingDir}
}
```

Replace both `&pb.ClientInfo{Name: "mcp"}` literals (lines 136 and 167) with `mcpClientInfo()`. Add `Client: mcpClientInfo()` to the `GetIndexRequest` (line ~187) and `SearchCodeRequest` (line ~297) the tools build.

- [ ] **Step 2: Build and test**

Run: `go test ./internal/mcpserver/ && go build ./...`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add internal/mcpserver/server.go
git commit -m "Send caller_cwd from the MCP adapter and on get and search requests"
```

---

### Task 10: `--wait <duration>`, clean Ctrl-C, no usage dump on runtime errors

**Files:**
- Modify: `cmd/lm-semantic-search/root.go` (silence usage after parse)
- Modify: `cmd/lm-semantic-search/codebase.go` (`index` and `sync` commands)
- Modify: `cmd/lm-semantic-search/rpc.go` (extract `printResponse` from `callAndPrint`)
- Create: `cmd/lm-semantic-search/job_watch.go`
- Test: `cmd/lm-semantic-search/main_test.go`

- [ ] **Step 1: Write the failing tests**

Add to `cmd/lm-semantic-search/main_test.go`:

```go
// TestIndexRejectsWaitOutsideHumanMode proves --wait cannot interleave
// progress rendering with machine output.
func TestIndexRejectsWaitOutsideHumanMode(t *testing.T) {
	root, _, _ := testRoot()
	root.SetArgs([]string{"--json", "codebase", "index", "/tmp/x", "--wait"})

	err := root.Execute()
	if err == nil {
		t.Fatal("expected --wait with --json to error")
	}
	if !strings.Contains(err.Error(), "--wait requires human output") {
		t.Fatalf("error = %q, want --wait requires human output", err.Error())
	}
}

// TestRuntimeErrorsDoNotPrintUsage proves a post-parse failure prints no
// usage block (the Ctrl-C dump from the incident).
func TestRuntimeErrorsDoNotPrintUsage(t *testing.T) {
	root, stdout, stderr := testRoot()
	root.SetArgs([]string{"daemon", "status", "--socket", "/nonexistent/socket.sock"})

	_ = root.Execute()
	combined := stdout.String() + stderr.String()
	if strings.Contains(combined, "Usage:") {
		t.Fatalf("runtime error printed usage:\n%s", combined)
	}
}
```

(Adapt the `testRoot()` return shape to the existing helper in `main_test.go`; it already returns the command plus stdout and stderr buffers.)

- [ ] **Step 2: Run, confirm failure**

Run: `go test ./cmd/lm-semantic-search/ -run 'TestIndexRejectsWait|TestRuntimeErrors' -v`
Expected: FAIL: unknown flag `--wait`, and the usage assertion fails.

- [ ] **Step 3: Implement usage silencing**

In `cmd/lm-semantic-search/root.go` `newRoot`, add a persistent pre-run. Cobra validates arguments before this hook runs, so argument mistakes still print usage while runtime failures do not:

```go
root.PersistentPreRun = func(cmd *cobra.Command, _ []string) {
	cmd.SilenceUsage = true
}
```

- [ ] **Step 4: Implement the watch helper**

Extract the formatting tail of `callAndPrint` (`rpc.go:57-73`) into a reusable function so the index path can print the registration reply and then keep the typed response:

```go
func printResponse(options cliOptions, result protoMessage) error {
	formatted, err := response.FormatProto(options.outputMode, result)
	if err != nil {
		slog.Error("format response failed", "err", err)
		return fmt.Errorf("format response: %w", err)
	}
	if _, err := fmt.Fprintf(os.Stdout, "%s\n", formatted); err != nil {
		return fmt.Errorf("write response output: %w", err)
	}
	return nil
}

func callAndPrint(options cliOptions, call rpcCall) error {
	result, err := callDaemon(options, call)
	if err != nil {
		return err
	}
	return printResponse(options, result)
}
```

Create `cmd/lm-semantic-search/job_watch.go`:

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	daemonclient "goodkind.io/lm-semantic-search/client"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
)

// watchJob attaches to one daemon job and renders progress lines until the
// job reaches a terminal state, the timeout expires, or the user interrupts.
// Timeout and interrupt both detach without cancelling the job: the daemon
// owns the job, so the command reports it as sent to the background.
func watchJob(options cliOptions, jobID string, timeout time.Duration) error {
	if jobID == "" {
		// A deduplicated or already-indexed registration has no job to watch.
		return nil
	}
	signalCtx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	ctx, cancel := context.WithTimeout(signalCtx, timeout)
	defer cancel()

	connection, client, err := daemonclient.DialDaemon(ctx, options.socketPath)
	if err != nil {
		return fmt.Errorf("dial daemon: %w", err)
	}
	defer connection.Close()

	stream, err := client.WatchJobs(ctx, &pb.WatchJobsRequest{JobIds: []string{jobID}})
	if err != nil {
		return formatCallError(err)
	}

	// The stream pushes transitions; fetch the current state once so a job
	// that finished before the attach does not wait out the whole timeout.
	if current, getErr := client.GetJob(ctx, &pb.GetJobRequest{JobId: jobID}); getErr == nil {
		if done, exitErr := renderJobUpdate(current.GetJob()); done {
			return exitErr
		}
	}

	for {
		update, recvErr := stream.Recv()
		if recvErr != nil {
			if ctx.Err() != nil {
				fmt.Fprintf(os.Stderr, "\nsent to background: job %s keeps running in the daemon; check it with `lm-semantic-search job get %s`\n", jobID, jobID)
				return nil
			}
			return formatCallError(recvErr)
		}
		if done, exitErr := renderJobUpdate(update.GetJob()); done {
			return exitErr
		}
	}
}

// renderJobUpdate prints one progress line and reports whether the job
// reached a terminal state, with the error the command should exit with.
func renderJobUpdate(job *pb.Job) (bool, error) {
	if job == nil {
		return false, nil
	}
	progress := job.GetProgress()
	unit := progress.GetUnit()
	if unit == "" {
		unit = "file"
	}
	fmt.Fprintf(os.Stderr, "\r\033[K%s %.1f%% (%d/%d %ss)",
		progress.GetPhase(), progress.GetOverallPercent(),
		progress.GetFilesProcessed(), progress.GetFilesTotal(), unit)

	switch model.JobState(job.GetState()) {
	case model.JobStateCompleted:
		fmt.Fprintf(os.Stderr, "\njob %s completed\n", job.GetId())
		return true, nil
	case model.JobStateFailed:
		fmt.Fprintln(os.Stderr)
		message := job.GetDisplayError()
		if message == "" {
			message = "job failed; see `lm-semantic-search job get " + job.GetId() + "`"
		}
		return true, errors.New(message)
	case model.JobStateCancelled:
		fmt.Fprintln(os.Stderr)
		return true, errors.New("job " + job.GetId() + " was cancelled")
	case model.JobStateQueued, model.JobStateRunning, model.JobStateCancelling:
		return false, nil
	default:
		return false, nil
	}
}
```

- [ ] **Step 5: Implement the flag on `index` and `sync`**

In `cmd/lm-semantic-search/codebase.go`, `newCodebaseIndexCmd` gains:

```go
var waitTimeout time.Duration
```

and after the existing flag registrations:

```go
cmd.Flags().DurationVar(&waitTimeout, "wait", 0, "attach to the job and render progress for up to this long; bare --wait uses 5m")
cmd.Flags().Lookup("wait").NoOptDefVal = "5m"
```

The RunE body changes from `callAndPrint` to:

```go
RunE: func(cmd *cobra.Command, args []string) error {
	clientInfo, err := currentClientInfo()
	if err != nil {
		return err
	}
	cliOpts := options.cliOptions()
	if waitTimeout > 0 && cliOpts.outputMode != response.ModeHuman {
		return errors.New("--wait requires human output mode")
	}
	request := &pb.StartIndexRequest{
		Path:             args[0],
		Force:            force,
		CustomExtensions: customExtensions,
		IgnorePatterns:   ignorePatterns,
		Client:           clientInfo,
	}
	if splitterType != "" {
		request.Splitter = &pb.SplitterConfig{Type: splitterType}
	}
	result, err := callDaemon(cliOpts, func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
		return client.StartIndex(ctx, request)
	})
	if err != nil {
		return err
	}
	if err := printResponse(cliOpts, result); err != nil {
		return err
	}
	if waitTimeout > 0 {
		started, ok := result.(*pb.StartIndexResponse)
		if !ok {
			return nil
		}
		return watchJob(cliOpts, started.GetJobId(), waitTimeout)
	}
	return nil
},
```

`newCodebaseSyncCmd` gets the same flag and the same tail using `*pb.SyncIndexResponse` and its `GetJobId()`.

- [ ] **Step 6: Run the CLI tests**

Run: `go test ./cmd/lm-semantic-search/ -v`
Expected: PASS, including both new tests and all pre-existing root/arg tests (arg-validation errors must still show usage; `TestCodebaseStatusRequiresPath` covers that).

- [ ] **Step 7: Commit**

```bash
git add cmd/lm-semantic-search/root.go cmd/lm-semantic-search/rpc.go cmd/lm-semantic-search/codebase.go cmd/lm-semantic-search/job_watch.go cmd/lm-semantic-search/main_test.go
git commit -m "Add bounded --wait progress attachment and silence usage on runtime errors"
```

---

### Task 11: Full gates, daemon restart, live smoke

- [ ] **Step 1: Run every gate**

```bash
go test ./...
make lint
make build
```

Expected: all PASS. Fix honestly anything that fails; do not weaken a gate.

- [ ] **Step 2: Install and restart the daemon**

The running daemon still holds the recursive FSEvents watch on `/` from the incident; the restart drops it and loads the new binaries. Check the repo's Makefile for the install target (`rg -n "install" Makefile`) and use it; then:

```bash
lm-semantic-search daemon stop
lm-semantic-search daemon status
```

Expected: launchd relaunches the daemon and `daemon status` reports the new build commit.

- [ ] **Step 3: Live smoke (interactive TTY via tmux)**

The Ctrl-C and progress-rendering checks need a real TTY; run them inside a tmux session. From any real repository directory (for example `~/Sites/agent-gate`, the directory from the incident):

```bash
lm-semantic-search codebase index .        # prints job id immediately, returns
lm-semantic-search codebase status .       # resolves . to this repo
lm-semantic-search codebase index . --wait # renders progress, exits on completion
lm-semantic-search codebase index /        # refused by the filesystem-root guard
```

Also confirm Ctrl-C during `--wait` prints the single sent-to-background line, and `lm-semantic-search --json codebase index . --wait` errors with `--wait requires human output mode`.

- [ ] **Step 4: Commit any smoke-test fixes**

The branch is then ready for validation, merge, push, and deploy.

---

## Self-review notes

- Spec coverage: registration latency (Tasks 3, 6, 7), single walk (Tasks 4, 5), watcher reuse (Task 6), caller_cwd resolution and refusals (Tasks 1, 2, 8, 9), `--wait` and Ctrl-C UX (Task 10), gates and smoke (Task 11). The proto field itself landed earlier as commit `212843d`.
- `model.JobState*` constants are referenced from `internal/model` in Task 10; they exist (used in `internal/mcpserver/server.go:371-380`).
- The `newCodeItemSource` signature change (Task 5) has one production call site at `manager_runner.go:85`; run `rg -n "newCodeItemSource"` before editing to confirm.
- After Task 6, `resolveIgnoreRulesOrLog` must have zero references or the deadcode lint gate fails; run `rg -n "resolveIgnoreRulesOrLog"` to confirm before committing.
