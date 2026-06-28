// Package indexability is the single source of truth for whether a path should
// be indexed. It resolves git-style ignore rules with a go-git matcher and then
// applies the content and size gates that decide a file's eligibility.
package indexability

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/go-git/go-git/v5/plumbing/format/gitignore"
	"goodkind.io/lm-semantic-search/internal/fileset"
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
)

// Decision is the indexability verdict for one path.
type Decision struct {
	// Indexed is true when the path passes every gate.
	Indexed bool
	// Reason names the gate that declined the path, or Keep when indexed.
	Reason Reason
}

// Resolver answers indexability questions for many codebases. It caches one
// built ignore matcher per codebase id and rebuilds it lazily after
// InvalidateRules. Each codebase builds under its own once, so a slow first
// build for one codebase never blocks Decide for another.
type Resolver struct {
	mu      sync.Mutex
	entries map[string]*ruleEntry
}

// ruleEntry holds one codebase's lazily built rules. once guards the build so a
// codebase resolves exactly once until it is invalidated, without holding the
// resolver mutex across the filesystem walk.
type ruleEntry struct {
	once  sync.Once
	built *builtRules
}

// NewResolver returns a Resolver with an empty per-codebase rule cache.
func NewResolver() *Resolver {
	return &Resolver{
		mu:      sync.Mutex{},
		entries: map[string]*ruleEntry{},
	}
}

// builtRules is the cached, resolved ignore state for one codebase.
type builtRules struct {
	matcher   gitignore.Matcher
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

	if !info.Mode().IsRegular() {
		return Decision{Indexed: false, Reason: ReasonNotRegular}
	}

	if rules.maxBytes > 0 && info.Size() > rules.maxBytes {
		return Decision{Indexed: false, Reason: ReasonOversize}
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
		entry.built = buildRules(ctx, root)
	})
	return entry.built
}

// pathIgnored reports whether relPath is excluded by matcher. It honors git's
// rule that a file under an excluded directory stays excluded even when a later
// pattern tries to re-include it, by checking every ancestor directory first.
func pathIgnored(matcher gitignore.Matcher, relPath string, isDir bool) bool {
	normalized := strings.Trim(strings.ReplaceAll(relPath, "\\", "/"), "/")
	if normalized == "" {
		return false
	}
	parts := strings.Split(normalized, "/")
	for index := 1; index < len(parts); index++ {
		if matcher.Match(parts[:index], true) {
			return true
		}
	}
	return matcher.Match(parts, isDir)
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
// root. Patterns are appended in order of increasing priority, which is the
// order go-git's matcher expects: global excludes, the project ignore, the
// built-in content denylist, the repo info/exclude, then each .gitignore from
// the root down the tree.
func buildRules(ctx context.Context, root string) *builtRules {
	commonDir, _ := gitworktree.CommonDirAt(root)

	patterns := make([]gitignore.Pattern, 0)
	patterns = append(patterns, globalExcludePatterns(ctx)...)
	patterns = append(patterns, contextIgnorePatterns(ctx)...)
	patterns = append(patterns, contentDenylistPatterns()...)
	patterns = append(patterns, infoExcludePatterns(ctx, commonDir)...)
	patterns = gatherGitignorePatterns(ctx, root, commonDir, patterns)

	return &builtRules{
		matcher:   gitignore.NewMatcher(patterns),
		commonDir: commonDir,
		maxBytes:  fileset.MaxFileBytes(),
	}
}

// gatherGitignorePatterns walks root top-down, appending each directory's
// .gitignore patterns to the running list. It prunes a child directory when the
// patterns gathered so far already exclude it, mirroring git's refusal to
// descend into an ignored directory, and it stops at a nested same-repo worktree
// boundary. The codebase root's own .git entry is skipped because it is never
// content.
func gatherGitignorePatterns(ctx context.Context, root string, commonDir string, base []gitignore.Pattern) []gitignore.Pattern {
	patterns := base
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
		patterns = append(patterns, gitignoreInDir(ctx, currentDir, relDir)...)
		matcher := gitignore.NewMatcher(patterns)
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
			if matcher.Match(strings.Split(childRel, "/"), true) {
				continue
			}
			walk(childRel)
		}
	}
	walk("")
	return patterns
}

// gitignoreInDir parses the .gitignore file in currentDir, if present, with the
// directory's relative path as the pattern domain so the patterns apply only
// under that directory.
func gitignoreInDir(ctx context.Context, currentDir string, relDir string) []gitignore.Pattern {
	lines := readIgnoreLines(ctx, filepath.Join(currentDir, ".gitignore"))
	return parsePatterns(lines, domainFor(relDir))
}

// domainFor splits a slash-separated relative directory into the path-component
// domain go-git expects, returning nil for the root.
func domainFor(relDir string) []string {
	if relDir == "" {
		return nil
	}
	return strings.Split(relDir, "/")
}

// infoExcludePatterns reads the repository's info/exclude file. For a linked
// worktree the file lives under the shared common dir, which commonDir already
// resolves to.
func infoExcludePatterns(ctx context.Context, commonDir string) []gitignore.Pattern {
	if commonDir == "" {
		return nil
	}
	lines := readIgnoreLines(ctx, filepath.Join(commonDir, "info", "exclude"))
	return parsePatterns(lines, nil)
}

// contextIgnorePatterns reads the project-global ignore file at
// ~/.context/.contextignore, matching the discovery package's loader.
func contextIgnorePatterns(ctx context.Context) []gitignore.Pattern {
	home, err := os.UserHomeDir()
	if err != nil {
		slog.WarnContext(ctx, "indexability: resolve home directory failed", "err", err)
		return nil
	}
	lines := readIgnoreLines(ctx, filepath.Join(home, ".context", ".contextignore"))
	return parsePatterns(lines, nil)
}

// globalExcludePatterns reads git's global excludes. It honors a
// core.excludesFile configured in $GIT_CONFIG_GLOBAL or ~/.gitconfig and also
// reads the default ~/.config/git/ignore location.
func globalExcludePatterns(ctx context.Context) []gitignore.Pattern {
	patterns := make([]gitignore.Pattern, 0)
	for _, path := range globalExcludePaths(ctx) {
		patterns = append(patterns, parsePatterns(readIgnoreLines(ctx, path), nil)...)
	}
	return patterns
}

// globalExcludePaths returns the candidate global excludes files in increasing
// priority: the default XDG location first, then any configured excludesFile.
func globalExcludePaths(ctx context.Context) []string {
	paths := make([]string, 0, 2)
	if home, err := os.UserHomeDir(); err == nil {
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
			return strings.TrimSpace(value)
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

// readIgnoreLines reads a .gitignore-style file and returns its significant
// lines, dropping blank lines and comments. A missing file yields no lines.
func readIgnoreLines(ctx context.Context, path string) []string {
	data, err := os.ReadFile(path)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.WarnContext(ctx, "indexability: read ignore file failed", "path", path, "err", err)
		}
		return nil
	}
	lines := make([]string, 0)
	for raw := range strings.SplitSeq(string(data), "\n") {
		trimmed := strings.TrimSpace(raw)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		lines = append(lines, trimmed)
	}
	return lines
}

// parsePatterns parses each line into a go-git pattern bound to domain.
func parsePatterns(lines []string, domain []string) []gitignore.Pattern {
	patterns := make([]gitignore.Pattern, 0, len(lines))
	for _, line := range lines {
		patterns = append(patterns, gitignore.ParsePattern(line, domain))
	}
	return patterns
}

// contentDenylistPatterns parses the built-in content denylist into root-domain
// patterns.
func contentDenylistPatterns() []gitignore.Pattern {
	return parsePatterns(contentBuiltinPatterns, nil)
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
