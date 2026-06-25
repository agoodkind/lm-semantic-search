package archguard

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

func TestBootstrapAndDeltaSyncCallResolveReuseSeed(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()
	rel := "internal/daemon/manager_delta.go"
	parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", rel, err)
	}

	required := map[string]bool{
		"runBootstrap": false,
		"runDeltaSync": false,
	}
	for _, decl := range parsed.Decls {
		functionDecl, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if _, found := required[functionDecl.Name.Name]; !found {
			continue
		}
		required[functionDecl.Name.Name] = functionCallsResolveReuseSeed(functionDecl)
	}

	var violations []string
	for functionName, foundCall := range required {
		if !foundCall {
			violations = append(violations, rel+": "+functionName+" does not call resolveReuseSeed")
		}
	}
	if len(violations) > 0 {
		t.Fatalf("bootstrap and delta sync must seed reuse through resolveReuseSeed:\n%s", strings.Join(violations, "\n"))
	}
}

func functionCallsResolveReuseSeed(functionDecl *ast.FuncDecl) bool {
	if functionDecl.Body == nil {
		return false
	}
	found := false
	ast.Inspect(functionDecl.Body, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		if calledFunctionName(call.Fun) == "resolveReuseSeed" {
			found = true
			return false
		}
		return true
	})
	return found
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
