package daemon

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// TestRenderLayerDoesNotLaunderJobStatus enforces the status single-source-of-truth
// invariant for the render layer by construction rather than by convention. The
// render files may format a job's resolved status.JobSurface, but they must not
// re-derive a job's state label or error echo from the raw model.Job record. The
// laundering bug that motivated the job resolver (the job list reading raw
// job.State and echoing job.Error.Message while a degraded banner already carried
// the cause) is exactly the pattern this test makes a build failure.
//
// Forbidden in render*.go:
//   - a local job-state-to-word helper (displayJobState), since status.JobStateLabelFor owns that vocabulary;
//   - reading job.Error to compose output, since the resolved JobSurface.ErrorLine decides the echo;
//   - reading the .Retryable flag, the only field that classifies a self-healing failure;
//   - converting a .State selector to a string for display (string(job.State));
//   - the literal " (retryable)" suffix, which the resolver owns.
//
// Branching on job.State for grouping or counting is allowed: that is control
// flow, not a status label, so an equality comparison against a model.JobState
// constant does not trip this guard.
func TestRenderLayerDoesNotLaunderJobStatus(t *testing.T) {
	t.Parallel()

	files, err := filepath.Glob("render*.go")
	if err != nil {
		t.Fatalf("glob render files: %v", err)
	}
	if len(files) == 0 {
		t.Fatal("no render*.go files found; the guard would silently pass")
	}

	var violations []string
	scannedNonTest := false
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		scannedNonTest = true
		violations = append(violations, scanRenderFileForLaundering(t, file)...)
	}
	if !scannedNonTest {
		t.Fatal("no non-test render*.go files scanned; the guard would silently pass")
	}

	if len(violations) > 0 {
		t.Fatalf(
			"render layer launders raw job status instead of using status.ResolveJob:\n  %s",
			strings.Join(violations, "\n  "),
		)
	}
}

// scanRenderFileForLaundering parses one render file and returns one message per
// forbidden raw-job-presentation construct it finds.
func scanRenderFileForLaundering(t *testing.T, file string) []string {
	t.Helper()
	fset := token.NewFileSet()
	parsed, err := parser.ParseFile(fset, file, nil, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", file, err)
	}

	var violations []string
	report := func(node ast.Node, detail string) {
		position := fset.Position(node.Pos())
		violations = append(violations, file+":"+itoa(position.Line)+": "+detail)
	}

	ast.Inspect(parsed, func(node ast.Node) bool {
		switch typed := node.(type) {
		case *ast.Ident:
			if typed.Name == "displayJobState" {
				report(typed, "calls a local job-state word helper; use status.JobStateLabelFor")
			}
		case *ast.SelectorExpr:
			if typed.Sel.Name == "Retryable" {
				report(typed, "reads .Retryable; the retryable fold belongs to status.ResolveJob")
			}
			if typed.Sel.Name == "Error" && isIdent(typed.X, "job") {
				report(typed, "reads job.Error; use the resolved JobSurface.ErrorLine")
			}
		case *ast.CallExpr:
			if isStringConversionOfState(typed) {
				report(typed, "converts a .State selector to a display string; use status.JobStateLabelFor")
			}
		case *ast.BasicLit:
			if typed.Kind == token.STRING && strings.Contains(typed.Value, "(retryable)") {
				report(typed, "hard-codes the \"(retryable)\" suffix; the resolver owns it")
			}
		}
		return true
	})
	return violations
}

// isIdent reports whether expr is a bare identifier with the given name.
func isIdent(expr ast.Expr, name string) bool {
	ident, ok := expr.(*ast.Ident)
	return ok && ident.Name == name
}

// isStringConversionOfState reports whether call is string(x.State), the
// conversion that turns a raw job state into display text.
func isStringConversionOfState(call *ast.CallExpr) bool {
	if !isIdent(call.Fun, "string") || len(call.Args) != 1 {
		return false
	}
	selector, ok := call.Args[0].(*ast.SelectorExpr)
	return ok && selector.Sel.Name == "State"
}

// itoa renders a line number without pulling in strconv at the call site, kept
// local so the guard test has no production dependency surface.
func itoa(value int) string {
	if value == 0 {
		return "0"
	}
	digits := ""
	for value > 0 {
		digits = string(rune('0'+value%10)) + digits
		value /= 10
	}
	return digits
}
