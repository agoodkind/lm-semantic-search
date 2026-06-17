package archguard

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// statusResolveFile is the one sanctioned home for the searchable fold, the
// single status chokepoint. status.ResolveSearchable is the only place that may
// combine the indexed precondition with the dependency health.
const statusResolveFile = "internal/status/status.go"

// TestSearchableNotComputedInline fails if any production file outside the status
// package combines the searchable preconditions inline, for example
// `Searchable: eligible && !health.Degraded()` in an RPC response literal or a
// `searchable := ...` assignment. The fold must live only in
// status.ResolveSearchable so the wire `searchable` field and the displayed
// status are derived from one resolution and cannot diverge.
func TestSearchableNotComputedInline(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string

	for _, rel := range productionGoFiles(t, root) {
		if rel == statusResolveFile {
			continue
		}
		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			switch typed := node.(type) {
			case *ast.KeyValueExpr:
				if isSearchableIdent(typed.Key) && isBoolExpr(typed.Value) {
					violations = append(violations, rel+": Searchable assigned an inline boolean expression")
				}
			case *ast.AssignStmt:
				for index, lhs := range typed.Lhs {
					if index >= len(typed.Rhs) {
						continue
					}
					if isSearchableIdent(lhs) && isBoolExpr(typed.Rhs[index]) {
						violations = append(violations, rel+": searchable assigned an inline boolean expression")
					}
				}
			}
			return true
		})
	}

	if len(violations) > 0 {
		t.Fatalf("searchable folded outside status.ResolveSearchable; route it through the status chokepoint:\n%s",
			strings.Join(violations, "\n"))
	}
}

// isSearchableIdent reports whether expr is the identifier `Searchable` (the
// response field) or `searchable` (a local), the names the fold would assign to.
func isSearchableIdent(expr ast.Expr) bool {
	ident, ok := expr.(*ast.Ident)
	return ok && strings.EqualFold(ident.Name, "searchable")
}

// isBoolExpr reports whether expr is a logical/comparison expression, which is
// what an inline searchable fold looks like. A call (computeSearchable, a getter)
// or a plain identifier is allowed; only an open-coded boolean formula fails.
func isBoolExpr(expr ast.Expr) bool {
	binary, ok := expr.(*ast.BinaryExpr)
	if !ok {
		return false
	}
	switch binary.Op {
	case token.LAND, token.LOR, token.EQL, token.NEQ:
		return true
	default:
		return false
	}
}
