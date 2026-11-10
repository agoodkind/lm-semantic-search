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

// filesetDecisionMethods names the resolver entry points that own the file-set
// verdict. The gate helpers (EligibleByStat, EligibleContent, MaxFileBytes) are
// unexported inside internal/indexability, so the routed files must reach the
// size and content gates only through these resolver decisions.
var filesetDecisionMethods = map[string]bool{
	"Decide":        true,
	"DecideContent": true,
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
		if !importsIndexability(parsed) {
			violations = append(violations, rel+": does not import internal/indexability")
			continue
		}

		parsed, err = parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		if !callsResolverDecision(parsed) {
			violations = append(violations, rel+": imports indexability but does not route the file-set verdict through a resolver decision")
		}
	}

	if len(violations) > 0 {
		t.Fatalf("file eligibility must route through internal/indexability:\n%s", strings.Join(violations, "\n"))
	}
}

func TestMaxFileBytesConstantOnlyInIndexability(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string

	for _, rel := range productionGoFiles(t, root) {
		if strings.HasPrefix(rel, "internal/indexability/") {
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
					violations = append(violations, rel+": declares a max-file-bytes const outside internal/indexability")
					continue
				}
				if constDeclContainsTwoMiBLiteral(valueSpec) {
					violations = append(violations, rel+": declares the 2 MiB file cap literal outside internal/indexability")
				}
			}
		}
	}

	if len(violations) > 0 {
		t.Fatalf("max file byte cap must have one production home:\n%s", strings.Join(violations, "\n"))
	}
}

func importsIndexability(file *ast.File) bool {
	for _, importSpec := range file.Imports {
		path, err := strconv.Unquote(importSpec.Path.Value)
		if err != nil {
			continue
		}
		if path == "goodkind.io/lm-semantic-search/internal/indexability" {
			return true
		}
	}
	return false
}

func callsResolverDecision(file *ast.File) bool {
	resolverNames := resolverTypedNames(file)
	callsDecision := false
	ast.Inspect(file, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if !filesetDecisionMethods[selector.Sel.Name] {
			return true
		}
		// Require the receiver be an identifier declared with the resolver type,
		// so a same-named Decide/DecideContent method on some other type cannot
		// satisfy the guard. Type info is unavailable in this AST-only check, so
		// the receiver name set is gathered from declarations in the same file.
		receiver, ok := selector.X.(*ast.Ident)
		if ok && resolverNames[receiver.Name] {
			callsDecision = true
			return false
		}
		return true
	})
	return callsDecision
}

// resolverTypedNames returns the identifier names in file that are declared with
// type indexability.Resolver or *indexability.Resolver, across function
// parameters, struct fields, and var declarations. callsResolverDecision uses
// the set to confirm a Decide/DecideContent call's receiver is a resolver.
func resolverTypedNames(file *ast.File) map[string]bool {
	names := make(map[string]bool)
	ast.Inspect(file, func(node ast.Node) bool {
		field, ok := node.(*ast.Field)
		if !ok || !isIndexabilityResolverType(field.Type) {
			return true
		}
		for _, name := range field.Names {
			names[name.Name] = true
		}
		return true
	})
	for _, decl := range file.Decls {
		genDecl, ok := decl.(*ast.GenDecl)
		if !ok || genDecl.Tok != token.VAR {
			continue
		}
		for _, spec := range genDecl.Specs {
			valueSpec, ok := spec.(*ast.ValueSpec)
			if !ok || !isIndexabilityResolverType(valueSpec.Type) {
				continue
			}
			for _, name := range valueSpec.Names {
				names[name.Name] = true
			}
		}
	}
	return names
}

// isIndexabilityResolverType reports whether expr is indexability.Resolver or
// *indexability.Resolver.
func isIndexabilityResolverType(expr ast.Expr) bool {
	if star, ok := expr.(*ast.StarExpr); ok {
		return isIndexabilityResolverType(star.X)
	}
	selector, ok := expr.(*ast.SelectorExpr)
	if !ok || selector.Sel.Name != "Resolver" {
		return false
	}
	pkg, ok := selector.X.(*ast.Ident)
	return ok && pkg.Name == "indexability"
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
