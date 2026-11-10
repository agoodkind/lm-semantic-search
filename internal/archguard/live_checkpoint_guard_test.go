package archguard

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// liveCheckpointReader is the file that owns the one decision about whether an
// absent merkle checkpoint means the index lost state. Everything else reads a
// live checkpoint through it.
const liveCheckpointReader = "internal/daemon/run_artifacts.go"

// merkleHome is the package that implements the reads, so its own internal
// calls are not a bypass of the chokepoint above it.
const merkleHome = "internal/merkle/"

// liveCheckpointReadFunctions are the merkle entry points that hand back a live
// checkpoint's contents. LoadOptionalSnapshotForConfig is deliberately absent:
// it reads the staging bootstrap checkpoint, whose absence carries no
// information about lost state, so it is free to be called anywhere.
var liveCheckpointReadFunctions = map[string]bool{
	"ReadSnapshot":          true,
	"LoadSnapshotForConfig": true,
}

// The empty-repository defect returned twice because the rule for reading a
// checkpoint was written per call site, and each time a call site the fix had
// not visited kept reporting a healthy codebase as damaged. This guard is the
// standing half of that fix: a new production reader of a live checkpoint fails
// here until it goes through Manager.loadLiveCheckpoint, which is the only
// place the expectation is decided.
func TestLiveCheckpointReadsGoThroughOneChokepoint(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string

	for _, rel := range productionGoFiles(t, root) {
		slashed := filepath.ToSlash(rel)
		if slashed == liveCheckpointReader || strings.HasPrefix(slashed, merkleHome) {
			continue
		}
		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		for _, name := range merkleSnapshotReadCalls(parsed) {
			violations = append(violations, slashed+": calls merkle."+name+" directly; read the live checkpoint through Manager.loadLiveCheckpoint so the absent-file verdict stays in one place")
		}
	}

	if len(violations) > 0 {
		t.Fatalf("live merkle checkpoint reads must go through %s:\n%s", liveCheckpointReader, strings.Join(violations, "\n"))
	}
}

// merkleSnapshotReadCalls returns the names of the merkle read functions called
// in one parsed file. It matches on the merkle package qualifier so a
// same-named method on another type does not register.
func merkleSnapshotReadCalls(file *ast.File) []string {
	var called []string
	ast.Inspect(file, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || !liveCheckpointReadFunctions[selector.Sel.Name] {
			return true
		}
		pkg, ok := selector.X.(*ast.Ident)
		if ok && pkg.Name == "merkle" {
			called = append(called, selector.Sel.Name)
		}
		return true
	})
	return called
}
