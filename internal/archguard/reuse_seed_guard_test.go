package archguard

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

func TestBootstrapAndDeltaSyncSeedReuseThroughResolver(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()
	rel := "internal/daemon/manager_delta.go"
	parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", rel, err)
	}

	required := map[string]struct{}{
		"runBootstrap": {},
		"runDeltaSync": {},
	}
	var violations []string
	for _, decl := range parsed.Decls {
		functionDecl, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if _, found := required[functionDecl.Name.Name]; !found {
			continue
		}
		delete(required, functionDecl.Name.Name)
		captures, assignsReuse := functionSeedsReuseFromResolver(functionDecl)
		if !captures {
			violations = append(violations, rel+": "+functionDecl.Name.Name+" must capture the result of resolveReuseSeed in an assignment, not ignore it")
		}
		if !assignsReuse {
			violations = append(violations, rel+": "+functionDecl.Name.Name+" must assign state.reuse so the seeded vectors actually feed the build")
		}
	}
	for functionName := range required {
		violations = append(violations, rel+": "+functionName+" was not found")
	}
	if len(violations) > 0 {
		t.Fatalf("bootstrap and delta sync must seed reuse through resolveReuseSeed:\n%s", strings.Join(violations, "\n"))
	}
}

// functionSeedsReuseFromResolver reports two facts about a function body: it
// binds the result of a resolveReuseSeed call to at least one named variable,
// and it assigns state.reuse somewhere. Requiring both stops a regression where
// resolveReuseSeed is called but its result is discarded, or where state.reuse
// is never assigned at all. It does not verify that the state.reuse assignment
// reads from the captured resolveReuseSeed result; that data-flow link is
// covered by the behavioral reuse tests, not this structural guard.
func functionSeedsReuseFromResolver(functionDecl *ast.FuncDecl) (captures bool, assignsReuse bool) {
	if functionDecl.Body == nil {
		return false, false
	}
	ast.Inspect(functionDecl.Body, func(node ast.Node) bool {
		assign, ok := node.(*ast.AssignStmt)
		if !ok {
			return true
		}
		// Count the call only when its result is bound to at least one named
		// variable. `_, _, _ := resolveReuseSeed(...)` discards every return and
		// must not satisfy the guard.
		for _, rhs := range assign.Rhs {
			call, ok := rhs.(*ast.CallExpr)
			if ok && calledFunctionName(call.Fun) == "resolveReuseSeed" && assignBindsAnyValue(assign.Lhs) {
				captures = true
			}
		}
		// Require state.reuse specifically, not any field named reuse, so seeding
		// some unrelated .reuse field does not pass.
		for _, lhs := range assign.Lhs {
			selector, ok := lhs.(*ast.SelectorExpr)
			if !ok || selector.Sel.Name != "reuse" {
				continue
			}
			if receiver, ok := selector.X.(*ast.Ident); ok && receiver.Name == "state" {
				assignsReuse = true
			}
		}
		return true
	})
	return captures, assignsReuse
}

// assignBindsAnyValue reports whether at least one left-hand side of an
// assignment binds the value rather than discarding it. A blank identifier (_)
// discards; any other target (a named variable, or a selector or index
// expression) binds.
func assignBindsAnyValue(lhs []ast.Expr) bool {
	for _, expr := range lhs {
		ident, ok := expr.(*ast.Ident)
		if !ok || ident.Name != "_" {
			return true
		}
	}
	return false
}

func calledFunctionName(expr ast.Expr) string {
	switch typed := expr.(type) {
	case *ast.Ident:
		return typed.Name
	case *ast.SelectorExpr:
		return typed.Sel.Name
	default:
		return ""
	}
}
