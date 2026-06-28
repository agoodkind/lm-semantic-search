package archguard

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// resolverConstructionAllowedDirs are the only production homes allowed to
// construct an indexability.Resolver: the daemon, which owns the one shared
// resolver threaded into discovery, the indexer, and merkle capture, and the
// indexability package, which defines the constructor itself.
var resolverConstructionAllowedDirs = []string{
	"internal/daemon/",
	"internal/indexability/",
}

// TestResolverConstructedOnlyInDaemon fails when any production .go file outside
// internal/daemon and internal/indexability calls indexability.NewResolver. The
// daemon threads its one shared resolver (manager.indexability) into discovery,
// the indexer, and merkle capture, so a private resolver minted elsewhere would
// silently skip the per-codebase custom ignore overrides the shared resolver
// applies.
func TestResolverConstructedOnlyInDaemon(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string

	for _, rel := range productionGoFiles(t, root) {
		if dirAllowsResolverConstruction(rel) {
			continue
		}
		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			selector, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			pkg, ok := selector.X.(*ast.Ident)
			if !ok {
				return true
			}
			if pkg.Name == "indexability" && selector.Sel.Name == "NewResolver" {
				violations = append(violations, rel+": constructs indexability.NewResolver outside the daemon; thread manager.indexability instead")
			}
			return true
		})
	}

	if len(violations) > 0 {
		t.Fatalf("indexability.Resolver must be constructed only in internal/daemon:\n%s", strings.Join(violations, "\n"))
	}
}

func dirAllowsResolverConstruction(rel string) bool {
	for _, dir := range resolverConstructionAllowedDirs {
		if strings.HasPrefix(rel, dir) {
			return true
		}
	}
	return false
}
