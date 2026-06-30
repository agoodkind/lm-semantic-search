// Package indexability is the single source of truth for whether a path should
// be indexed. It resolves git-style ignore rules with a git-pkgs matcher and
// then applies the content and size gates that decide a file's eligibility.
package indexability

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/git-pkgs/gitignore"
	"goodkind.io/lm-semantic-search/internal/gitworktree"
)

// Reason names the gate that decided a path's indexability. The empty value
// Keep means the path passed every gate.
type Reason string

const (
	// Keep marks a path that passes every gate and stays indexable.
	Keep Reason = ""
	// ReasonOutOfScope marks a path that belongs to a nested same-repo worktree.
	ReasonOutOfScope Reason = "out-of-scope"
	// ReasonIgnored marks a path excluded by a git-style ignore rule.
	ReasonIgnored Reason = "ignored"
	// ReasonNotRegular marks directories, symlinks, and other non-regular files.
	ReasonNotRegular Reason = "not-regular"
	// ReasonOversize marks regular files above the configured byte cap.
	ReasonOversize Reason = "oversize"
	// ReasonNonUTF8 marks files whose content is not valid UTF-8, the post-read
	// content gate DecideContent applies.
	ReasonNonUTF8 Reason = "non-utf-8"
)

// Decision is the indexability verdict for one path.
type Decision struct {
	// Indexed is true when the path passes every gate.
	Indexed bool
	// Reason names the gate that declined the path, or Keep when indexed.
	Reason Reason
}

// IgnoreOverrides supplies the per-codebase custom ignore patterns the daemon
// merged from --ignore, the MCP ignorePatterns argument, and the
// CUSTOM_IGNORE_PATTERNS env var. The resolver calls it while building a
// codebase's matcher and appends the returned patterns last, so a custom
// pattern wins over a repository re-include. A nil func means no overrides.
type IgnoreOverrides func(codebaseID string) []string

// Resolver answers indexability questions for many codebases. It caches one
// built ignore matcher per codebase id and rebuilds it lazily after
// InvalidateRules. Each codebase builds under its own once, so a slow first
// build for one codebase never blocks Decide for another.
type Resolver struct {
	mu        sync.Mutex
	entries   map[string]*ruleEntry
	overrides IgnoreOverrides
}

// ruleEntry holds one codebase's lazily built rules. once guards the build so a
// codebase resolves exactly once until it is invalidated, without holding the
// resolver mutex across the filesystem walk.
type ruleEntry struct {
	once  sync.Once
	built *builtRules
}

// NewResolver returns a Resolver with an empty per-codebase rule cache. The
// overrides func supplies each codebase's custom ignore patterns; pass nil when
// there are none.
func NewResolver(overrides IgnoreOverrides) *Resolver {
	return &Resolver{
		mu:        sync.Mutex{},
		entries:   map[string]*ruleEntry{},
		overrides: overrides,
	}
}

// builtRules is the cached, resolved ignore state for one codebase.
type builtRules struct {
	matcher   *gitignore.Matcher
	commonDir string
	maxBytes  int64
}

// Decide applies the pre-read gates to relPath in this order: a path inside a
// nested same-repo worktree is out of scope, a path matched by a git-style
// ignore rule is ignored, a non-regular file is rejected, and an oversize file
// is rejected. relPath is slash-separated and relative to root, and info is the
// path's [os.FileInfo].
func (r *Resolver) Decide(ctx context.Context, codebaseID string, root string, relPath string, info os.FileInfo) Decision {
	rules := r.rulesFor(ctx, codebaseID, root)

	if reason := rules.scopeOrIgnore(root, relPath, info.IsDir()); reason != Keep {
		return Decision{Indexed: false, Reason: reason}
	}

	if keep, skip := eligibleByStat(info, rules.maxBytes); !keep {
		return Decision{Indexed: false, Reason: reasonForSkip(skip)}
	}

	return Decision{Indexed: true, Reason: Keep}
}

// DecideContent applies the post-read content gate to data: a file whose bytes
// are not valid UTF-8 is rejected, because Milvus requires every VarChar field
// to be valid UTF-8 on the gRPC wire. Decide already covers the pre-read
// size, scope, and not-regular gates; DecideContent is the second half of the
// file-set choke for the bytes only the caller can read, so merkle.Capture and
// the indexer route their content decision here instead of running the gate
// themselves.
func (r *Resolver) DecideContent(data []byte) Decision {
	if keep, skip := eligibleContent(data); !keep {
		return Decision{Indexed: false, Reason: reasonForSkip(skip)}
	}
	return Decision{Indexed: true, Reason: Keep}
}

// Ignored reports whether relPath is out of scope or git-ignored for the
// codebase, without needing the path's [os.FileInfo]. The watcher uses it on
// every event, including deletes and renames where the file is already gone, so
// structurally out-of-scope paths such as .git metadata never enqueue. Pass
// isDir when known; deleted paths can pass false, because the ancestor-directory
// check still catches a path under an ignored directory.
func (r *Resolver) Ignored(ctx context.Context, codebaseID string, root string, relPath string, isDir bool) bool {
	return r.rulesFor(ctx, codebaseID, root).scopeOrIgnore(root, relPath, isDir) != Keep
}

// scopeOrIgnore returns the scope or ignore reason for relPath, or Keep. It
// needs no file info: a path in a nested same-repo worktree is out of scope, the
// git directory is out of scope, and a path matched by a git-style rule is
// ignored.
//
// The git directory is metadata, never worktree content. This is structural
// scope, not an ignore rule: ".git" is never listed in .gitignore because git
// excludes its own directory implicitly, so the matcher alone would let .git/*
// churn through on every git operation. A linked worktree's ".git" is a file and
// is excluded the same way. ".gitignore" and ".gitattributes" stay indexable
// because the component is compared exactly to ".git".
func (rules *builtRules) scopeOrIgnore(root string, relPath string, isDir bool) Reason {
	if rules.commonDir != "" {
		absPath := filepath.Join(root, filepath.FromSlash(relPath))
		if gitworktree.PathInsideNestedWorktree(root, rules.commonDir, absPath) {
			return ReasonOutOfScope
		}
	}
	if hasGitComponent(relPath) {
		return ReasonOutOfScope
	}
	if pathIgnored(rules.matcher, relPath, isDir) {
		return ReasonIgnored
	}
	return Keep
}

// IgnoreDetail reports whether relPath is excluded for the codebase and names
// the git-style rule that excluded it. It mirrors Ignored's verdict and adds
// the matched-pattern detail the status surface shows, so no caller runs its
// own matcher. The returned [gitignore.MatchResult] carries the matched pattern
// text and its source file. An out-of-scope path (a nested same-repo worktree
// or the .git directory) reports excluded with an empty pattern, because no
// .gitignore line names it. relPath is slash-separated and relative to root;
// pass isDir for a directory so a directory-only pattern matches.
func (r *Resolver) IgnoreDetail(ctx context.Context, codebaseID string, root string, relPath string, isDir bool) (gitignore.MatchResult, bool) {
	rules := r.rulesFor(ctx, codebaseID, root)
	reason := rules.scopeOrIgnore(root, relPath, isDir)
	if reason == Keep {
		return gitignore.MatchResult{}, false
	}
	if reason != ReasonIgnored {
		return gitignore.MatchResult{Ignored: true}, true
	}
	return ignoreDetail(rules.matcher, relPath, isDir), true
}

// ignoreDetail returns the matched-pattern detail for relPath, honoring git's
// directory-exclusion rule by reporting the first excluded ancestor directory's
// pattern before the leaf's own. It mirrors pathIgnored's ancestor walk so the
// detail names the same rule that decided the verdict. The matcher uses the
// trailing-slash convention to recognize directories, so each ancestor and an
// isDir leaf carry a trailing slash.
func ignoreDetail(matcher *gitignore.Matcher, relPath string, isDir bool) gitignore.MatchResult {
	normalized := strings.Trim(strings.ReplaceAll(relPath, "\\", "/"), "/")
	if normalized == "" {
		return gitignore.MatchResult{}
	}
	parts := strings.Split(normalized, "/")
	for index := 1; index < len(parts); index++ {
		ancestor := strings.Join(parts[:index], "/") + "/"
		if detail := matcher.MatchDetail(ancestor); detail.Ignored {
			return detail
		}
	}
	leaf := normalized
	if isDir {
		leaf += "/"
	}
	return matcher.MatchDetail(leaf)
}

// InvalidateRules drops the cached matcher for codebaseID so the next Decide
// rebuilds it. Callers invalidate after a .gitignore or other ignore source
// changes on disk.
func (r *Resolver) InvalidateRules(codebaseID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.entries, codebaseID)
}

// rulesFor returns the cached rules for codebaseID, building them on first use.
// The resolver mutex guards only the entry map; the filesystem walk runs under
// the entry's own once, so building one codebase never blocks Decide for
// another.
func (r *Resolver) rulesFor(ctx context.Context, codebaseID string, root string) *builtRules {
	r.mu.Lock()
	entry, found := r.entries[codebaseID]
	if !found {
		entry = &ruleEntry{once: sync.Once{}, built: nil}
		r.entries[codebaseID] = entry
	}
	r.mu.Unlock()

	entry.once.Do(func() {
		entry.built = r.buildRules(ctx, codebaseID, root)
	})
	return entry.built
}

// pathIgnored reports whether relPath is excluded by matcher. It honors git's
// rule that a file under an excluded directory stays excluded even when a later
// pattern tries to re-include it, by checking every ancestor directory first.
// git-pkgs leaves that rule to the caller: its MatchPath re-includes a file
// under an excluded directory, so the ancestor walk here enforces it.
func pathIgnored(matcher *gitignore.Matcher, relPath string, isDir bool) bool {
	normalized := strings.Trim(strings.ReplaceAll(relPath, "\\", "/"), "/")
	if normalized == "" {
		return false
	}
	parts := strings.Split(normalized, "/")
	for index := 1; index < len(parts); index++ {
		if matcher.MatchPath(strings.Join(parts[:index], "/"), true) {
			return true
		}
	}
	return matcher.MatchPath(normalized, isDir)
}

// hasGitComponent reports whether relPath has a path component exactly equal to
// ".git", meaning it is the git directory, a linked worktree's ".git" pointer
// file, or content beneath either. git tracks nothing there, so the resolver
// treats it as out of scope rather than relying on a .gitignore pattern that
// never exists.
func hasGitComponent(relPath string) bool {
	normalized := strings.Trim(strings.ReplaceAll(relPath, "\\", "/"), "/")
	if normalized == "" {
		return false
	}
	return slices.Contains(strings.Split(normalized, "/"), ".git")
}

// buildRules resolves the ignore matcher and size cap for the codebase rooted at
// root. Sources are added in order of increasing priority, which is the order
// git-pkgs matches in (last match wins): global excludes, the project ignore,
// the built-in content denylist, the repo info/exclude, each .gitignore from the
// root down the tree, then the per-codebase custom override patterns. The
// content denylist stays below .gitignore so a repository's own re-include rule
// can still index a denylisted name; the custom overrides stay above .gitignore
// so a user-supplied pattern wins over a repository re-include.
func (r *Resolver) buildRules(ctx context.Context, codebaseID string, root string) *builtRules {
	commonDir, _ := gitworktree.CommonDirAt(root)

	matcher := gitignore.New("")
	addGlobalExcludes(ctx, matcher)
	matcher.AddFromFile(contextIgnorePath(ctx), "")
	matcher.AddPatterns([]byte(strings.Join(contentBuiltinPatterns, "\n")), "")
	if commonDir != "" {
		matcher.AddFromFile(filepath.Join(commonDir, "info", "exclude"), "")
	}
	addGitignoreFiles(ctx, matcher, root, commonDir)
	if r.overrides != nil {
		if extra := r.overrides(codebaseID); len(extra) > 0 {
			matcher.AddPatterns([]byte(strings.Join(extra, "\n")), "")
		}
	}

	return &builtRules{
		matcher:   matcher,
		commonDir: commonDir,
		maxBytes:  maxFileBytes(),
	}
}

// addGitignoreFiles walks root top-down, adding each directory's .gitignore to
// matcher scoped to that directory. It prunes a child directory when the rules
// gathered so far already exclude it, mirroring git's refusal to descend into an
// ignored directory, and it stops at a nested same-repo worktree boundary. The
// codebase root's own .git entry is skipped because it is never content.
func addGitignoreFiles(ctx context.Context, matcher *gitignore.Matcher, root string, commonDir string) {
	var walk func(relDir string)
	walk = func(relDir string) {
		if ctx.Err() != nil {
			return
		}
		currentDir := filepath.Join(root, filepath.FromSlash(relDir))
		entries, err := os.ReadDir(currentDir)
		if err != nil {
			slog.WarnContext(ctx, "indexability: read directory failed", "path", currentDir, "err", err)
			return
		}
		matcher.AddFromFile(filepath.Join(currentDir, ".gitignore"), relDir)
		for _, entry := range entries {
			if !entry.IsDir() || entry.Name() == ".git" {
				continue
			}
			childRel := entry.Name()
			if relDir != "" {
				childRel = relDir + "/" + entry.Name()
			}
			childAbs := filepath.Join(currentDir, entry.Name())
			if gitworktree.WorktreeOfRepo(childAbs, commonDir) {
				continue
			}
			if matcher.MatchPath(childRel, true) {
				continue
			}
			walk(childRel)
		}
	}
	walk("")
}

// contextIgnorePath returns the project-global ignore file at
// ~/.context/.contextignore, matching the discovery package's loader. It returns
// an empty path when the home directory cannot be resolved; AddFromFile treats
// an empty or missing path as a no-op.
func contextIgnorePath(ctx context.Context) string {
	home, err := os.UserHomeDir()
	if err != nil {
		slog.WarnContext(ctx, "indexability: resolve home directory failed", "err", err)
		return ""
	}
	return filepath.Join(home, ".context", ".contextignore")
}

// addGlobalExcludes adds git's global excludes to matcher. It honors a
// core.excludesFile configured in $GIT_CONFIG_GLOBAL or ~/.gitconfig and also
// reads the default ~/.config/git/ignore location, lowest priority first.
func addGlobalExcludes(ctx context.Context, matcher *gitignore.Matcher) {
	for _, path := range globalExcludePaths(ctx) {
		matcher.AddFromFile(path, "")
	}
}

// globalExcludePaths returns the candidate global excludes files in increasing
// priority: the default XDG location first, then any configured excludesFile.
// The default honors git's order: $XDG_CONFIG_HOME/git/ignore when XDG_CONFIG_HOME
// is set, otherwise ~/.config/git/ignore.
func globalExcludePaths(ctx context.Context) []string {
	paths := make([]string, 0, 2)
	if xdg := os.Getenv("XDG_CONFIG_HOME"); xdg != "" {
		paths = append(paths, filepath.Join(xdg, "git", "ignore"))
	} else if home, err := os.UserHomeDir(); err == nil {
		paths = append(paths, filepath.Join(home, ".config", "git", "ignore"))
	}
	if configured := excludesFileFromConfig(ctx); configured != "" {
		paths = append(paths, configured)
	}
	return paths
}

// excludesFileFromConfig returns the resolved core.excludesFile path from the
// first git config file that declares it, or empty when none does.
func excludesFileFromConfig(ctx context.Context) string {
	for _, configPath := range gitConfigPaths(ctx) {
		if value := readExcludesFileKey(configPath); value != "" {
			return expandHome(value)
		}
	}
	return ""
}

// gitConfigPaths returns the global git config files to inspect, in priority
// order: $GIT_CONFIG_GLOBAL first, then ~/.gitconfig.
func gitConfigPaths(ctx context.Context) []string {
	paths := make([]string, 0, 2)
	if env := os.Getenv("GIT_CONFIG_GLOBAL"); env != "" {
		paths = append(paths, env)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		slog.WarnContext(ctx, "indexability: resolve home directory failed", "err", err)
		return paths
	}
	return append(paths, filepath.Join(home, ".gitconfig"))
}

// readExcludesFileKey returns the core.excludesFile value from a git config
// file, or empty when the file or key is absent.
func readExcludesFileKey(configPath string) string {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}
	inCore := false
	for raw := range strings.SplitSeq(string(data), "\n") {
		line := strings.TrimSpace(raw)
		if strings.HasPrefix(line, "[") {
			inCore = strings.EqualFold(sectionName(line), "core")
			continue
		}
		if !inCore {
			continue
		}
		key, value, found := strings.Cut(line, "=")
		if !found {
			continue
		}
		if strings.EqualFold(strings.TrimSpace(key), "excludesfile") {
			return unquoteConfigValue(strings.TrimSpace(value))
		}
	}
	return ""
}

// sectionName extracts the section name from a git config "[section ...]"
// header line.
func sectionName(line string) string {
	trimmed := strings.TrimSpace(strings.Trim(line, "[]"))
	if name, _, found := strings.Cut(trimmed, " "); found {
		return name
	}
	return trimmed
}

// unquoteConfigValue strips one matching pair of surrounding double or single
// quotes from a git config value, so excludesfile = "~/path" resolves to ~/path.
func unquoteConfigValue(value string) string {
	if len(value) >= 2 {
		first := value[0]
		last := value[len(value)-1]
		if (first == '"' || first == '\'') && last == first {
			return value[1 : len(value)-1]
		}
	}
	return value
}

// tildeMarker is the leading byte git config uses to abbreviate the user's home
// directory in a path value.
const tildeMarker = '~'

// expandHome expands a leading tilde in path to the user's home directory, since
// Go's file APIs do not expand it. A bare "~user" form is left untouched.
func expandHome(path string) string {
	if len(path) == 0 || path[0] != tildeMarker {
		return path
	}
	if len(path) > 1 && path[1] != '/' {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return path
	}
	return filepath.Join(home, path[1:])
}

// contentBuiltinPatterns is the built-in content denylist: file-extension,
// lockfile, and minified-bundle globs that are never useful to index. It
// deliberately omits directory ignores such as node_modules/ or dist/, which
// git's own ignore files now own.
var contentBuiltinPatterns = []string{
	".DS_Store",
	".env",
	".env.*",
	"*.swp",
	"*.swo",
	"*.log",
	"*.local",
	"*.min.js",
	"*.min.css",
	"*.min.map",
	"*.bundle.js",
	"*.bundle.css",
	"*.chunk.js",
	"*.vendor.js",
	"*.polyfills.js",
	"*.runtime.js",
	"*.map",
	// Image formats.
	"*.png",
	"*.jpg",
	"*.jpeg",
	"*.gif",
	"*.ico",
	"*.webp",
	"*.bmp",
	"*.tiff",
	"*.tif",
	"*.heic",
	"*.avif",
	// Document and archive binaries.
	"*.pdf",
	"*.doc",
	"*.docx",
	"*.xls",
	"*.xlsx",
	"*.ppt",
	"*.pptx",
	"*.zip",
	"*.tar",
	"*.tgz",
	"*.gz",
	"*.bz2",
	"*.xz",
	"*.7z",
	"*.rar",
	"*.dmg",
	"*.iso",
	// Native, compiled, and packaged binaries.
	"*.exe",
	"*.dll",
	"*.so",
	"*.dylib",
	"*.o",
	"*.obj",
	"*.a",
	"*.lib",
	"*.class",
	"*.jar",
	"*.war",
	"*.ear",
	"*.wasm",
	"*.pyc",
	"*.pyo",
	"*.pyd",
	// Fonts.
	"*.ttf",
	"*.otf",
	"*.woff",
	"*.woff2",
	"*.eot",
	// Audio and video.
	"*.mp3",
	"*.mp4",
	"*.mov",
	"*.avi",
	"*.mkv",
	"*.webm",
	"*.wav",
	"*.flac",
	"*.ogg",
	// Databases and embedded blobs.
	"*.sqlite",
	"*.sqlite3",
	"*.db",
	"*.bin",
	// Package lockfiles.
	"package-lock.json",
	"yarn.lock",
	"pnpm-lock.yaml",
	"Cargo.lock",
	"Gemfile.lock",
	"composer.lock",
	"poetry.lock",
	"go.sum",
}
