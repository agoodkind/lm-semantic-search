package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// rawStatusFieldReads reports every call in source that reads a raw lifecycle
// status field off a wire record.
//
// A protobuf getter takes no arguments, which is what separates
// codebase.GetStatus() from client.GetStatus(ctx, request). The first reads a
// raw lifecycle field for display, which is the fork this guard exists to
// prevent. The second is the status RPC, whose reply carries values the daemon
// already resolved.
func rawStatusFieldReads(fset *token.FileSet, parsed *ast.File, label string) []string {
	var violations []string
	ast.Inspect(parsed, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if len(call.Args) > 0 {
			return true
		}
		if selector.Sel.Name == "GetStatus" || selector.Sel.Name == "GetState" {
			position := fset.Position(node.Pos())
			violations = append(violations, label+":"+position.String()+": calls "+selector.Sel.Name)
		}
		return true
	})
	return violations
}

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
		violations = append(violations, rawStatusFieldReads(fset, parsed, file)...)
	}
	if len(violations) > 0 {
		t.Fatalf("CLI reads raw status fields for display:\n%s", strings.Join(violations, "\n"))
	}
}

// TestRawStatusFieldReadsDiscriminates proves the guard still catches the read
// it was written for after being narrowed to zero-argument calls. A guard that
// cannot fail is worse than no guard, so the catching case is asserted here
// rather than inferred from the sweep above passing.
func TestRawStatusFieldReadsDiscriminates(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		source string
		want   int
	}{
		{
			name: "getter on a wire record is caught",
			source: `package main
func render(codebase *thing) string { return codebase.GetStatus() }`,
			want: 1,
		},
		{
			name: "raw state getter is caught",
			source: `package main
func render(job *thing) string { return job.GetState() }`,
			want: 1,
		},
		{
			name: "status RPC call is allowed",
			source: `package main
func read(ctx context, client svc) { client.GetStatus(ctx, request) }`,
			want: 0,
		},
		{
			name: "resolved display field is allowed",
			source: `package main
func render(codebase *thing) string { return codebase.GetDisplayStatus() }`,
			want: 0,
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			fset := token.NewFileSet()
			parsed, err := parser.ParseFile(fset, "sample.go", testCase.source, 0)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			got := rawStatusFieldReads(fset, parsed, "sample.go")
			if len(got) != testCase.want {
				t.Fatalf("violations = %d, want %d: %v", len(got), testCase.want, got)
			}
		})
	}
}
