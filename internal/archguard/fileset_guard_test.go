package archguard

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

var filesetRoutedFiles = []string{
	"internal/merkle/snapshot.go",
	"internal/indexer/indexer.go",
}

var filesetGateFunctions = map[string]bool{
	"EligibleByStat":  true,
	"EligibleContent": true,
	"MaxFileBytes":    true,
}

func TestFilesetEligibilityRoutesThroughSharedPackage(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string

	for _, rel := range filesetRoutedFiles {
		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, parser.ImportsOnly)
		if err != nil {
			t.Fatalf("parse imports %s: %v", rel, err)
		}
		if !importsFileset(parsed) {
			violations = append(violations, rel+": does not import internal/fileset")
			continue
		}

		parsed, err = parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		if !callsFilesetGate(parsed) {
			violations = append(violations, rel+": imports fileset but does not call an eligibility gate")
		}
	}

	if len(violations) > 0 {
		t.Fatalf("file eligibility must route through internal/fileset:\n%s", strings.Join(violations, "\n"))
	}
}

func TestMaxFileBytesConstantOnlyInFileset(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string

	for _, rel := range productionGoFiles(t, root) {
		if strings.HasPrefix(rel, "internal/fileset/") {
			continue
		}
		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		for _, decl := range parsed.Decls {
			genDecl, ok := decl.(*ast.GenDecl)
			if !ok || genDecl.Tok != token.CONST {
				continue
			}
			for _, spec := range genDecl.Specs {
				valueSpec, ok := spec.(*ast.ValueSpec)
				if !ok {
					continue
				}
				if maxFileBytesConstNames(valueSpec) {
					violations = append(violations, rel+": declares a max-file-bytes const outside internal/fileset")
					continue
				}
				if constDeclContainsTwoMiBLiteral(valueSpec) {
					violations = append(violations, rel+": declares the 2 MiB file cap literal outside internal/fileset")
				}
			}
		}
	}

	if len(violations) > 0 {
		t.Fatalf("max file byte cap must have one production home:\n%s", strings.Join(violations, "\n"))
	}
}

func importsFileset(file *ast.File) bool {
	for _, importSpec := range file.Imports {
		path, err := strconv.Unquote(importSpec.Path.Value)
		if err != nil {
			continue
		}
		if path == "goodkind.io/lm-semantic-search/internal/fileset" {
			return true
		}
	}
	return false
}

func callsFilesetGate(file *ast.File) bool {
	callsGate := false
	ast.Inspect(file, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		receiver, ok := selector.X.(*ast.Ident)
		if !ok || receiver.Name != "fileset" {
			return true
		}
		if filesetGateFunctions[selector.Sel.Name] {
			callsGate = true
			return false
		}
		return true
	})
	return callsGate
}

func maxFileBytesConstNames(valueSpec *ast.ValueSpec) bool {
	for _, name := range valueSpec.Names {
		if strings.Contains(name.Name, "MaxFileBytes") || strings.Contains(name.Name, "maxFileBytes") {
			return true
		}
	}
	return false
}

func constDeclContainsTwoMiBLiteral(valueSpec *ast.ValueSpec) bool {
	for _, value := range valueSpec.Values {
		if isTwoMiBConstValue(value) {
			return true
		}
	}
	return false
}

func isTwoMiBConstValue(expr ast.Expr) bool {
	value, ok := constProductValue(expr)
	return ok && value == 2*1024*1024
}

func constProductValue(expr ast.Expr) (int64, bool) {
	switch typed := expr.(type) {
	case *ast.BasicLit:
		if typed.Kind != token.INT {
			return 0, false
		}
		value, err := strconv.ParseInt(typed.Value, 0, 64)
		if err != nil {
			return 0, false
		}
		return value, true
	case *ast.BinaryExpr:
		if typed.Op != token.MUL {
			return 0, false
		}
		left, leftOK := constProductValue(typed.X)
		right, rightOK := constProductValue(typed.Y)
		if !leftOK || !rightOK {
			return 0, false
		}
		return left * right, true
	default:
		return 0, false
	}
}
