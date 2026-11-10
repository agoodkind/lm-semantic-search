package archguard

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// milvusTransportPackage holds the per-call Milvus deadline policy. Its files
// are guarded as a unit because the invariant is about the whole package, not
// about one function: nothing in it may consult a clock or the caller's
// deadline.
const milvusTransportPackage = "internal/semantic/milvusgrpc/"

// milvusDeadlineSelector is the function that installs the per-call bound.
const milvusDeadlineSelector = "ClientReporter"

// clockReadNames are the calls that return a fresh reading of the machine's
// clock. internal/clock.Now returns time.Now().UTC(), and UTC strips Go's
// monotonic reading, so any comparison against it silently drops onto wall time.
var clockReadNames = map[string]bool{
	"Now":   true,
	"Since": true,
	"Until": true,
}

// clockPackageNames are the qualifiers those readings arrive through.
var clockPackageNames = map[string]bool{
	"time":  true,
	"clock": true,
}

// TestMilvusCallDeadlineSelectionReadsNoClock pins the shape of the Milvus
// per-call deadline bound, which no behavioral test in that package can pin on
// its own.
//
// The bound must be installed by handing the caller's context and the timeout to
// context.WithTimeout and letting it keep whichever deadline is earlier. That
// package compares the two deadlines on Go's monotonic timer, which no clock
// correction moves. Two earlier defects both came from doing the comparison by
// hand instead: branching on ctx.Deadline() to bound only callers that set none,
// which let a caller with a longer deadline keep it, and then comparing the
// caller's instant against a wall-clock reading, which let a forward clock
// correction read a live deadline as expired and skip the bound.
//
// Neither defect is reachable through a test of the package's behavior. The
// wall-clock one only appears when the wall and monotonic readings disagree, and
// Go's public time API cannot build a time.Time whose readings disagree, so a
// test cannot construct the input that separates the correct implementation from
// the broken one. What the two defects do share is a shape: both read something
// the correct implementation never reads. This guard bans those reads, so a
// restoration of either fails here rather than passing a green suite and waiting
// for a real clock correction in production.
func TestMilvusCallDeadlineSelectionReadsNoClock(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()

	var violations []string
	selectorFound := false
	guardedFiles := 0

	for _, rel := range productionGoFiles(t, root) {
		if !strings.HasPrefix(filepath.ToSlash(rel), milvusTransportPackage) {
			continue
		}
		guardedFiles++

		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		violations = append(violations, clockAndDeadlineReads(rel, parsed)...)

		for _, decl := range parsed.Decls {
			functionDecl, ok := decl.(*ast.FuncDecl)
			if !ok || functionDecl.Name.Name != milvusDeadlineSelector {
				continue
			}
			selectorFound = true
			if !callsContextWithTimeout(functionDecl) {
				violations = append(
					violations,
					rel+": "+milvusDeadlineSelector+" must install the bound with context.WithTimeout so the caller's deadline and the bound are compared on the monotonic timer",
				)
			}
		}
	}

	if guardedFiles == 0 {
		t.Fatalf("no production files found under %s; the guard would pass vacuously", milvusTransportPackage)
	}
	if !selectorFound {
		violations = append(violations, milvusTransportPackage+": "+milvusDeadlineSelector+" was not found")
	}
	if len(violations) > 0 {
		t.Fatalf(
			"the Milvus per-call bound must be chosen by context.WithTimeout alone:\n%s",
			strings.Join(violations, "\n"),
		)
	}
}

// clockAndDeadlineReads reports every clock reading and every Deadline call in
// one file. A clock reading is the operand that drops a deadline comparison onto
// wall time; a Deadline call is how a hand-rolled comparison gets the other
// operand. The correct implementation needs neither, so either one is a
// regression rather than a style question.
func clockAndDeadlineReads(rel string, file *ast.File) []string {
	var violations []string
	ast.Inspect(file, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if selector.Sel.Name == "Deadline" {
			violations = append(
				violations,
				rel+": calls Deadline(); the bound must not branch on the caller's deadline, because context.WithTimeout already keeps whichever deadline is earlier",
			)
			return true
		}
		qualifier, ok := selector.X.(*ast.Ident)
		if !ok || !clockPackageNames[qualifier.Name] || !clockReadNames[selector.Sel.Name] {
			return true
		}
		violations = append(
			violations,
			rel+": reads the clock through "+qualifier.Name+"."+selector.Sel.Name+"(); comparing a deadline against a fresh reading drops the comparison onto wall time, where a forward correction hides a live deadline",
		)
		return true
	})
	return violations
}

// callsContextWithTimeout reports whether a function installs its deadline with
// context.WithTimeout.
func callsContextWithTimeout(functionDecl *ast.FuncDecl) bool {
	if functionDecl.Body == nil {
		return false
	}
	found := false
	ast.Inspect(functionDecl.Body, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || selector.Sel.Name != "WithTimeout" {
			return true
		}
		if qualifier, ok := selector.X.(*ast.Ident); ok && qualifier.Name == "context" {
			found = true
			return false
		}
		return true
	})
	return found
}
