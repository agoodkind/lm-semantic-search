package archguard

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// dependencyModeHome is the one production file allowed to assign the global
// dependency health Mode. Every other file must treat the dependency mode as a
// global-only fact and must never fold a per-path collection fact into it.
const dependencyModeHome = "internal/daemon/health.go"

// TestDependencyModeNotAssignedOutsideHealth fails if any production file other
// than health.go assigns to a `.Mode` field, for example `health.Mode = depMode`.
// That assignment is exactly how a per-path collection readiness once leaked into
// the global store banner: GetIndex folded a not-loaded collection into the
// dependency mode, and a codebase that was simply still indexing showed "Vector
// store unavailable". Per-path readiness is now a separate type
// (status.CollectionReadiness) carried on status.Inputs.Collection, so the global
// banner stays reserved for a real ProbeHealth failure. This guard, alongside the
// type separation the compiler already enforces, stops the conflation from
// returning under a future refactor.
func TestDependencyModeNotAssignedOutsideHealth(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string

	for _, rel := range productionGoFiles(t, root) {
		if rel == dependencyModeHome {
			continue
		}
		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			assign, ok := node.(*ast.AssignStmt)
			if !ok {
				return true
			}
			for _, lhs := range assign.Lhs {
				selector, ok := lhs.(*ast.SelectorExpr)
				if ok && selector.Sel.Name == "Mode" {
					violations = append(violations, rel+": assigns to a .Mode field; the dependency mode is global-only and must not carry a per-path collection fact")
				}
			}
			return true
		})
	}

	if len(violations) > 0 {
		t.Fatalf("dependency Mode assigned outside health.go; keep per-path readiness out of the global dependency channel:\n%s",
			strings.Join(violations, "\n"))
	}
}
