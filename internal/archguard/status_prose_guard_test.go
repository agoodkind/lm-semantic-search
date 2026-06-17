package archguard

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// renderDirPrefix is the render package tree, relative to the module root.
var renderDirPrefix = "internal/render" + string(filepath.Separator)

// statusBodySentinels are phrases unique to the get-index body prose for the
// non-template states (failed, missing, stale, quarantined). After the status
// chokepoint refactor these live only at the daemon boundary in
// internal/daemon/status_narrative.go; the render layer joins the pre-built
// lines and must not synthesize them. A sentinel appearing under internal/render
// means status prose leaked back into the formatter.
var statusBodySentinels = []string{
	"source directory is missing",
	"could not be indexed",
	"is stale",
	"quarantined after a suspicious large disappearance",
}

// TestStatusBodyProseLivesAtBoundary fails if any non-test render file contains a
// non-template status body sentinel. Keep the guard strict with no carve-out: the
// fix for a violation is to move the prose into resolveStatusNarrative at the
// daemon boundary, not to allowlist the render file.
func TestStatusBodyProseLivesAtBoundary(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string
	for _, rel := range productionGoFiles(t, root) {
		if !strings.HasPrefix(rel, renderDirPrefix) {
			continue
		}
		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		for _, literal := range stringLiterals(parsed) {
			for _, sentinel := range statusBodySentinels {
				if strings.Contains(literal, sentinel) {
					violations = append(violations, rel+": string literal contains "+sentinel)
				}
			}
		}
	}
	if len(violations) > 0 {
		t.Fatalf("status body prose found in the render layer; move it to resolveStatusNarrative at the daemon boundary:\n%s", strings.Join(violations, "\n"))
	}
}

// launderedStatusFields are raw model status fields the render layer must never
// read directly. They are resolved into view types at the daemon boundary; a
// render file that reads them is re-deriving status presentation from a raw
// record behind the view wall's back.
var launderedStatusFields = map[string]bool{
	"FilesTotal":        true,
	"FilesProcessed":    true,
	"FilesAdded":        true,
	"FilesModified":     true,
	"FilesRemoved":      true,
	"FilesEmbedded":     true,
	"ChunksGenerated":   true,
	"ChunksReused":      true,
	"ChunksTotal":       true,
	"OverallPercent":    true,
	"StartedAt":         true,
	"CompletedAt":       true,
	"LastSuccessfulRun": true,
	"LastFailedRun":     true,
	"Quarantine":        true,
}

// launderingReceivers are the identifier names a render file would bind a raw
// record to before reading a status field off it.
var launderingReceivers = map[string]bool{
	"progress": true,
	"job":      true,
	"codebase": true,
}

// TestRenderLayerDoesNotLaunderStatusFields fails if a render file reads a raw
// model status field off a progress, job, or codebase identifier. The render
// layer formats resolved view types; raw record fields belong to the daemon
// boundary. This is the deferred laundering guard, landed with the status
// chokepoint as defense in depth behind the internal/render import wall.
func TestRenderLayerDoesNotLaunderStatusFields(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string
	for _, rel := range productionGoFiles(t, root) {
		if !strings.HasPrefix(rel, renderDirPrefix) {
			continue
		}
		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			selector, ok := node.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			receiver, ok := selector.X.(*ast.Ident)
			if !ok || !launderingReceivers[receiver.Name] {
				return true
			}
			if launderedStatusFields[selector.Sel.Name] {
				violations = append(violations, rel+": reads raw "+receiver.Name+"."+selector.Sel.Name)
			}
			return true
		})
	}
	if len(violations) > 0 {
		t.Fatalf("render layer read raw status fields; format resolved view types instead:\n%s", strings.Join(violations, "\n"))
	}
}
