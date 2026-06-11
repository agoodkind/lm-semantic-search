package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// The CLI renders only resolved display fields from the daemon. Reading the
// raw lifecycle fields for display is how the TUI forked its own status
// vocabulary once; this guard makes that a test failure.
func TestCLIDisplayDoesNotReadRawStatusFields(t *testing.T) {
	t.Parallel()
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	fset := token.NewFileSet()
	var violations []string
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		parsed, parseErr := parser.ParseFile(fset, file, nil, 0)
		if parseErr != nil {
			t.Fatalf("parse %s: %v", file, parseErr)
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
			if selector.Sel.Name == "GetStatus" || selector.Sel.Name == "GetState" {
				position := fset.Position(node.Pos())
				violations = append(violations, file+":"+position.String()+": calls "+selector.Sel.Name)
			}
			return true
		})
	}
	if len(violations) > 0 {
		t.Fatalf("CLI reads raw status fields for display:\n%s", strings.Join(violations, "\n"))
	}
}
