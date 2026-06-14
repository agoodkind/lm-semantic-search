package archguard

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// renderFile is the one sanctioned home for the breakdown presentation
// vocabulary, relative to the module root.
const renderFile = "internal/render/render.go"

// breakdownConstructorFiles may build view.OutcomeBreakdown / view.OutcomeRow:
// the resolver and zero helper in view, and the proto rebuild in pbconv. Every
// other non-test file must go through view.ResolveBreakdown / view.ZeroBreakdown
// / view.NewOutcomeRow.
var breakdownConstructorFiles = map[string]bool{
	"internal/view/view.go":     true,
	"internal/pbconv/pbconv.go": true,
}

// laundering signatures. The tree connectors and the multi-word labels are
// unique to the breakdown renderer; a string literal containing one anywhere
// else is a surface re-deriving status output by hand. Bare kind words like
// "embedded" or "unchanged" are deliberately excluded: they are kind values and
// ordinary status words used in other lines.
var forbiddenVocabulary = []string{
	"├─",
	"└─",
	"pending, not sent yet",
	"skipped, too large",
	"error, unreadable",
}

// moduleRoot walks up from the test's working directory until it finds go.mod.
func moduleRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, statErr := os.Stat(filepath.Join(dir, "go.mod")); statErr == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("go.mod not found above the test directory")
		}
		dir = parent
	}
}

// productionGoFiles returns every non-test .go file under the module root,
// skipping generated and vendored trees, as paths relative to the root.
func productionGoFiles(t *testing.T, root string) []string {
	t.Helper()
	var files []string
	walkErr := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			switch entry.Name() {
			case "gen", "third_party", "vendor", ".git", "dist":
				return filepath.SkipDir
			default:
				return nil
			}
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return relErr
		}
		files = append(files, rel)
		return nil
	})
	if walkErr != nil {
		t.Fatalf("walk: %v", walkErr)
	}
	return files
}

// stringLiterals returns every string-literal value in one parsed file.
func stringLiterals(file *ast.File) []string {
	var literals []string
	ast.Inspect(file, func(node ast.Node) bool {
		lit, ok := node.(*ast.BasicLit)
		if ok && lit.Kind == token.STRING {
			literals = append(literals, lit.Value)
		}
		return true
	})
	return literals
}

// TestBreakdownVocabularyOnlyInRenderer fails if the breakdown tree connectors or
// the unique row labels appear in any production .go file other than the
// renderer, or in a status template. This catches a surface that hand-builds the
// status tree instead of calling render.BreakdownLines.
func TestBreakdownVocabularyOnlyInRenderer(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string

	for _, rel := range productionGoFiles(t, root) {
		if rel == renderFile {
			continue
		}
		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		for _, literal := range stringLiterals(parsed) {
			for _, phrase := range forbiddenVocabulary {
				if strings.Contains(literal, phrase) {
					violations = append(violations, rel+": string literal contains "+phrase)
				}
			}
		}
	}

	// Templates inject the breakdown via {{ .BreakdownBlock }}; none may draw the
	// tree itself, so the connectors must not appear in template source.
	templates, _ := filepath.Glob(filepath.Join(root, "internal/render/templates/status/*.md.tmpl"))
	for _, path := range templates {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read template %s: %v", path, err)
		}
		for _, connector := range []string{"├─", "└─"} {
			if strings.Contains(string(data), connector) {
				rel, _ := filepath.Rel(root, path)
				violations = append(violations, rel+": template draws the tree connector "+connector)
			}
		}
	}

	if len(violations) > 0 {
		t.Fatalf("status breakdown vocabulary found outside %s; route output through render.BreakdownLines:\n%s",
			renderFile, strings.Join(violations, "\n"))
	}
}

// TestBreakdownConstructedOnlyInSanctionedPlaces fails if a view.OutcomeBreakdown
// or view.OutcomeRow composite literal, or a view.NewOutcomeRow call, appears in
// a production file other than the resolver (view) or the proto rebuild
// (pbconv). A surface must obtain a breakdown from view.ResolveBreakdown rather
// than assembling one to feed the renderer.
func TestBreakdownConstructedOnlyInSanctionedPlaces(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string

	for _, rel := range productionGoFiles(t, root) {
		if breakdownConstructorFiles[rel] {
			continue
		}
		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			switch typed := node.(type) {
			case *ast.CompositeLit:
				if name := viewSelector(typed.Type); name == "OutcomeBreakdown" || name == "OutcomeRow" {
					violations = append(violations, rel+": constructs view."+name+" literal")
				}
			case *ast.CallExpr:
				if viewSelector(typed.Fun) == "NewOutcomeRow" {
					violations = append(violations, rel+": calls view.NewOutcomeRow")
				}
			}
			return true
		})
	}

	if len(violations) > 0 {
		t.Fatalf("breakdown built outside view/pbconv; use view.ResolveBreakdown:\n%s", strings.Join(violations, "\n"))
	}
}

// viewSelector returns the selector name when expr is `view.<Name>`, else "".
func viewSelector(expr ast.Expr) string {
	selector, ok := expr.(*ast.SelectorExpr)
	if !ok {
		return ""
	}
	pkg, ok := selector.X.(*ast.Ident)
	if !ok || pkg.Name != "view" {
		return ""
	}
	return selector.Sel.Name
}

// TestRenderExposesNoGlyphAccessor fails if internal/render exports any
// identifier that would let another package pull a per-kind glyph or label out
// of the unexported presentation table, which would reopen the laundering door.
// BreakdownLines must stay the only exported way to render a breakdown.
func TestRenderExposesNoGlyphAccessor(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	renderDir := filepath.Join(root, "internal/render")
	entries, err := os.ReadDir(renderDir)
	if err != nil {
		t.Fatalf("read render dir: %v", err)
	}
	fset := token.NewFileSet()
	var violations []string
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		parsed, parseErr := parser.ParseFile(fset, filepath.Join(renderDir, name), nil, 0)
		if parseErr != nil {
			t.Fatalf("parse %s: %v", name, parseErr)
		}
		for _, decl := range parsed.Decls {
			for _, ident := range exportedTopLevelNames(decl) {
				if strings.Contains(ident, "Glyph") || strings.Contains(ident, "Label") || strings.Contains(ident, "Presentation") {
					violations = append(violations, name+": exports "+ident)
				}
			}
		}
	}
	if len(violations) > 0 {
		t.Fatalf("render exports a glyph/label accessor; keep the presentation table unexported:\n%s", strings.Join(violations, "\n"))
	}
}

// exportedTopLevelNames returns the exported identifiers a top-level declaration
// introduces (funcs, types, vars, consts).
func exportedTopLevelNames(decl ast.Decl) []string {
	var names []string
	switch typed := decl.(type) {
	case *ast.FuncDecl:
		if typed.Name.IsExported() {
			names = append(names, typed.Name.Name)
		}
	case *ast.GenDecl:
		for _, spec := range typed.Specs {
			switch specTyped := spec.(type) {
			case *ast.TypeSpec:
				if specTyped.Name.IsExported() {
					names = append(names, specTyped.Name.Name)
				}
			case *ast.ValueSpec:
				for _, ident := range specTyped.Names {
					if ident.IsExported() {
						names = append(names, ident.Name)
					}
				}
			}
		}
	}
	return names
}
