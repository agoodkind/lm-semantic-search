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

// indexabilityPackageDir is the one production home allowed to define ignore
// rules, the content denylist, and the file-size cap. Every other production
// file must route through it.
const indexabilityPackageDir = "internal/indexability/"

// rawIgnoreEvaluators names the bespoke path-ignore functions the old discovery
// system exposed. A call to any of them outside internal/indexability means a
// caller is matching ignore rules itself instead of asking the resolver.
var rawIgnoreEvaluators = map[string]bool{
	"PathIgnored":             true,
	"pathIgnored":             true,
	"EffectiveIgnorePatterns": true,
	"walkGitignore":           true,
}

// hardcodedIgnoreListThreshold is the count of ignore-pattern-like string
// elements in one composite literal that marks it as a hand-rolled denylist.
// A handful of glob or directory patterns can appear incidentally; a list this
// long is a denylist that belongs in the indexability content rules.
const hardcodedIgnoreListThreshold = 6

// TestNoBespokeIgnoreSystemOutsideIndexability fails when any production file
// outside internal/indexability reintroduces the deleted ignore machinery: a
// file-size-cap constant, a hardcoded ignore-pattern or binary-extension list,
// an ignore-rules value, or a call to a raw path-ignore evaluator. Indexability
// is the single source of truth, so each of these must live only there.
func TestNoBespokeIgnoreSystemOutsideIndexability(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string

	for _, rel := range productionGoFiles(t, root) {
		if strings.HasPrefix(rel, indexabilityPackageDir) {
			continue
		}
		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		violations = append(violations, ignoreSystemViolations(rel, parsed)...)
	}

	if len(violations) > 0 {
		t.Fatalf("ignore rules, the content denylist, and the file-size cap must live only in internal/indexability:\n%s", strings.Join(violations, "\n"))
	}
}

// ignoreSystemViolations reports every bespoke-ignore reintroduction in one
// parsed production file.
func ignoreSystemViolations(rel string, file *ast.File) []string {
	var violations []string
	for _, decl := range file.Decls {
		genDecl, ok := decl.(*ast.GenDecl)
		if !ok {
			continue
		}
		violations = append(violations, declIgnoreViolations(rel, genDecl)...)
	}

	ast.Inspect(file, func(node ast.Node) bool {
		switch typed := node.(type) {
		case *ast.CompositeLit:
			if compositeIsIgnoreRules(typed) {
				violations = append(violations, rel+": constructs an ignore-rules value outside internal/indexability")
			}
		case *ast.CallExpr:
			if rawIgnoreEvaluators[calledFunctionName(typed.Fun)] {
				violations = append(violations, rel+": calls raw path-ignore evaluator "+calledFunctionName(typed.Fun)+" outside internal/indexability")
			}
		}
		return true
	})
	return violations
}

// declIgnoreViolations reports a file-size-cap constant or a hardcoded
// ignore-pattern list declared in one const or var block.
func declIgnoreViolations(rel string, genDecl *ast.GenDecl) []string {
	var violations []string
	for _, spec := range genDecl.Specs {
		valueSpec, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}
		if genDecl.Tok == token.CONST {
			if maxFileBytesConstNames(valueSpec) {
				violations = append(violations, rel+": declares a max-file-bytes const outside internal/indexability")
			}
			if constDeclContainsTwoMiBLiteral(valueSpec) {
				violations = append(violations, rel+": declares the 2 MiB file cap literal outside internal/indexability")
			}
		}
		for _, value := range valueSpec.Values {
			if compositeIsHardcodedIgnoreList(value) {
				violations = append(violations, rel+": declares a hardcoded ignore-pattern or binary-extension list outside internal/indexability")
			}
		}
	}
	return violations
}

// compositeIsIgnoreRules reports whether a composite literal builds a value of a
// type whose name is IgnoreRules, whether referenced bare or through a package
// selector such as discovery.IgnoreRules.
func compositeIsIgnoreRules(lit *ast.CompositeLit) bool {
	switch typed := lit.Type.(type) {
	case *ast.Ident:
		return typed.Name == "IgnoreRules"
	case *ast.SelectorExpr:
		return typed.Sel.Name == "IgnoreRules"
	default:
		return false
	}
}

// compositeIsHardcodedIgnoreList reports whether expr is a string composite
// literal carrying enough ignore-pattern-like elements to be a hand-rolled
// denylist. The element shapes mirror .gitignore: a glob extension (*.ext), a
// directory pattern (name/), or a leading-dot tooling or VCS directory.
func compositeIsHardcodedIgnoreList(expr ast.Expr) bool {
	lit, ok := expr.(*ast.CompositeLit)
	if !ok {
		return false
	}
	patternLike := 0
	for _, element := range lit.Elts {
		basicLit, ok := element.(*ast.BasicLit)
		if !ok || basicLit.Kind != token.STRING {
			continue
		}
		value, err := strconv.Unquote(basicLit.Value)
		if err != nil {
			continue
		}
		if ignorePatternLike(value) {
			patternLike++
		}
	}
	return patternLike >= hardcodedIgnoreListThreshold
}

// ignorePatternLike reports whether one string looks like a .gitignore entry:
// a glob extension, a trailing-slash directory, or a leading-dot tooling or VCS
// directory.
func ignorePatternLike(value string) bool {
	if strings.HasPrefix(value, "*.") {
		return true
	}
	if strings.HasSuffix(value, "/") {
		return true
	}
	if strings.HasPrefix(value, ".") && len(value) > 1 {
		return true
	}
	return false
}
