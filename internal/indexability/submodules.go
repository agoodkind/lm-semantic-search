package indexability

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"goodkind.io/lm-semantic-search/internal/gitworktree"
)

type submoduleRules struct {
	byPath     map[string]string
	allowedSet map[string]bool
}

type submoduleDecl struct {
	name string
	path string
}

func buildSubmoduleRules(ctx context.Context, root string, allowlist []string) submoduleRules {
	rules := submoduleRules{
		byPath:     map[string]string{},
		allowedSet: normalizeSubmoduleAllowlist(allowlist),
	}
	rules.addGitmodules(ctx, root, "")
	return rules
}

func (rules submoduleRules) addGitmodules(ctx context.Context, currentDir string, relDir string) {
	for _, decl := range parseGitmodules(ctx, filepath.Join(currentDir, ".gitmodules"), relDir) {
		rules.add(decl.path, decl.name)
	}
}

func (rules submoduleRules) add(relPath string, name string) {
	normalizedPath := normalizeSubmoduleToken(relPath)
	if normalizedPath == "" {
		return
	}
	if existingName := rules.byPath[normalizedPath]; existingName != "" {
		return
	}
	rules.byPath[normalizedPath] = strings.TrimSpace(name)
}

func (rules submoduleRules) allowed(relRoot string) bool {
	normalizedRoot := normalizeSubmoduleToken(relRoot)
	if normalizedRoot == "" {
		return false
	}
	if rules.allowedSet[normalizedRoot] {
		return true
	}
	name := rules.byPath[normalizedRoot]
	return name != "" && rules.allowedSet[name]
}

func (rules submoduleRules) isRoot(relPath string) bool {
	_, found := rules.byPath[normalizeSubmoduleToken(relPath)]
	return found
}

func (rules submoduleRules) containing(root string, commonDir string, relPath string, absPath string) (string, bool) {
	normalizedPath := normalizeSubmoduleToken(relPath)
	bestRoot := ""
	for relRoot := range rules.byPath {
		if normalizedPath == relRoot || strings.HasPrefix(normalizedPath, relRoot+"/") {
			if len(relRoot) > len(bestRoot) {
				bestRoot = relRoot
			}
		}
	}
	if fsRoot, ok := gitworktree.PathInsideSubmodule(root, commonDir, absPath); ok {
		normalizedFSRoot := normalizeSubmoduleToken(fsRoot)
		if len(normalizedFSRoot) >= len(bestRoot) {
			bestRoot = normalizedFSRoot
		}
	}
	if bestRoot == "" {
		return "", false
	}
	return bestRoot, true
}

func normalizeSubmoduleAllowlist(values []string) map[string]bool {
	allowed := make(map[string]bool, len(values))
	for _, value := range values {
		normalized := normalizeSubmoduleToken(value)
		if normalized == "" {
			continue
		}
		allowed[normalized] = true
	}
	return allowed
}

func normalizeSubmoduleToken(value string) string {
	trimmed := strings.TrimSpace(strings.ReplaceAll(value, "\\", "/"))
	if trimmed == "" || trimmed == "." {
		return ""
	}
	cleanedNative := filepath.Clean(filepath.FromSlash(trimmed))
	cleaned := filepath.ToSlash(cleanedNative)
	if cleaned == "." || cleaned == ".." || strings.HasPrefix(cleaned, "../") || filepath.IsAbs(cleanedNative) {
		return ""
	}
	return cleaned
}

func parseGitmodules(ctx context.Context, path string, baseRel string) []submoduleDecl {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	decls := make([]submoduleDecl, 0)
	currentName := ""
	currentPath := ""
	flush := func() {
		if currentName == "" || currentPath == "" {
			currentName = ""
			currentPath = ""
			return
		}
		fullPath := currentPath
		if baseRel != "" {
			fullPath = baseRel + "/" + currentPath
		}
		decls = append(decls, submoduleDecl{name: currentName, path: fullPath})
		currentName = ""
		currentPath = ""
	}
	for rawLine := range strings.SplitSeq(string(data), "\n") {
		line := strings.TrimSpace(rawLine)
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue
		}
		if strings.HasPrefix(line, "[") {
			flush()
			currentName = submoduleSectionName(line)
			continue
		}
		key, value, found := strings.Cut(line, "=")
		if !found {
			continue
		}
		if currentName != "" && strings.EqualFold(strings.TrimSpace(key), "path") {
			currentPath = unquoteConfigValue(strings.TrimSpace(value))
		}
	}
	flush()
	if len(decls) == 0 && len(data) > 0 {
		slog.DebugContext(ctx, "indexability: .gitmodules declared no submodule paths", "path", path)
	}
	return decls
}

func submoduleSectionName(line string) string {
	trimmed := strings.TrimSpace(strings.Trim(line, "[]"))
	if !strings.HasPrefix(trimmed, "submodule") {
		return ""
	}
	_, rest, found := strings.Cut(trimmed, " ")
	if !found {
		return ""
	}
	return unquoteConfigValue(strings.TrimSpace(rest))
}
