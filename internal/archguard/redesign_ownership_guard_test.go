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

// ignoreObserverFile is the one production file allowed to call the resolver's
// InvalidateRules. The single ignore-source observer owns invalidation so every
// other component routes its "rules may have changed" signal through the
// observer instead of dropping the resolver cache itself.
const ignoreObserverFile = "internal/daemon/ignore_observer.go"

// TestInvalidateRulesCalledOnlyInObserver fails when any production .go file
// outside internal/daemon/ignore_observer.go calls .InvalidateRules. The
// observer is the sole invalidator, so a call elsewhere reintroduces the
// scattered-invalidation smell the redesign removed.
func TestInvalidateRulesCalledOnlyInObserver(t *testing.T) {
	// Not parallel: this guard parses every production file, so running it
	// alongside the other whole-module scans would multiply peak CI memory.
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string

	for _, rel := range productionGoFiles(t, root) {
		if filepath.ToSlash(rel) == ignoreObserverFile {
			continue
		}
		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			if calledFunctionName(call.Fun) == "InvalidateRules" {
				violations = append(violations, rel+": calls InvalidateRules outside the single ignore-source observer; route through observer.Invalidate instead")
			}
			return true
		})
	}

	if len(violations) > 0 {
		t.Fatalf("resolver cache invalidation must run only through internal/daemon/ignore_observer.go:\n%s", strings.Join(violations, "\n"))
	}
}

// TestScopeCheckedOnlyInIndexability fails when any production .go file outside
// internal/indexability calls gitworktree.PathInsideNestedWorktree. Nested
// same-repo worktree scope is the resolver's decision, reached through Decide
// and Ignored, so a second scope check elsewhere duplicates that logic.
func TestScopeCheckedOnlyInIndexability(t *testing.T) {
	// Not parallel: this guard parses every production file, so running it
	// alongside the other whole-module scans would multiply peak CI memory.
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string

	for _, rel := range productionGoFiles(t, root) {
		if strings.HasPrefix(filepath.ToSlash(rel), indexabilityPackageDir) {
			continue
		}
		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			if calledFunctionName(call.Fun) == "PathInsideNestedWorktree" {
				violations = append(violations, rel+": calls PathInsideNestedWorktree outside internal/indexability; route scope through the resolver")
			}
			return true
		})
	}

	if len(violations) > 0 {
		t.Fatalf("nested-worktree scope must be decided only in internal/indexability:\n%s", strings.Join(violations, "\n"))
	}
}

// TestIgnoreSourcePredicateOnlyInIndexability fails when any production .go file
// outside internal/indexability defines a function that looks like a second
// "is this an ignore source" predicate. The resolver's IgnoreSources and
// IsIgnoreSourcePath are the one definition of which on-disk files decide
// indexability, so a divergent re-derivation elsewhere is the DRY violation the
// redesign removed.
//
// Heuristic: a function declaration or a function literal whose result is a
// single bool and whose body references the string literal ".gitignore" or
// "info/exclude", the two on-disk markers an ignore-source predicate keys off.
// That shape catches both a named predicate and a closure assigned to a var,
// for either marker, while staying narrow; a non-bool helper that merely
// mentions a marker (for example building a path) does not match, and a bool
// function that never names a marker does not match.
func TestIgnoreSourcePredicateOnlyInIndexability(t *testing.T) {
	// Not parallel: this guard parses every production file, so running it
	// alongside the other whole-module scans would multiply peak CI memory.
	root := moduleRoot(t)
	fset := token.NewFileSet()
	var violations []string

	for _, rel := range productionGoFiles(t, root) {
		if strings.HasPrefix(filepath.ToSlash(rel), indexabilityPackageDir) {
			continue
		}
		parsed, err := parser.ParseFile(fset, filepath.Join(root, rel), nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			funcType, body := boolFuncTypeAndBody(node)
			if funcType == nil {
				return true
			}
			if !funcTypeReturnsSingleBool(funcType) || !bodyReferencesIgnoreSourceLiteral(body) {
				return true
			}
			violations = append(violations, rel+`: defines an ignore-source predicate (a bool func or closure naming ".gitignore" or "info/exclude") outside internal/indexability; the resolver owns IgnoreSources and IsIgnoreSourcePath`)
			return true
		})
	}

	if len(violations) > 0 {
		t.Fatalf("the definition of an ignore source must live only in internal/indexability:\n%s", strings.Join(violations, "\n"))
	}
}

// boolFuncTypeAndBody returns the signature and body of a function declaration or
// a function literal, or nil when node is neither, so the predicate guard covers
// both a named func and a closure assigned to a var.
func boolFuncTypeAndBody(node ast.Node) (*ast.FuncType, *ast.BlockStmt) {
	switch typed := node.(type) {
	case *ast.FuncDecl:
		return typed.Type, typed.Body
	case *ast.FuncLit:
		return typed.Type, typed.Body
	default:
		return nil, nil
	}
}

// funcTypeReturnsSingleBool reports whether funcType declares exactly one result
// whose type is the bare bool identifier.
func funcTypeReturnsSingleBool(funcType *ast.FuncType) bool {
	if funcType.Results == nil || len(funcType.Results.List) != 1 {
		return false
	}
	result := funcType.Results.List[0]
	if len(result.Names) > 1 {
		return false
	}
	ident, ok := result.Type.(*ast.Ident)
	return ok && ident.Name == "bool"
}

// bodyReferencesIgnoreSourceLiteral reports whether body contains the string
// literal ".gitignore" or "info/exclude".
func bodyReferencesIgnoreSourceLiteral(body *ast.BlockStmt) bool {
	if body == nil {
		return false
	}
	found := false
	ast.Inspect(body, func(node ast.Node) bool {
		basicLit, ok := node.(*ast.BasicLit)
		if !ok || basicLit.Kind != token.STRING {
			return true
		}
		value, err := strconv.Unquote(basicLit.Value)
		if err != nil {
			return true
		}
		if value == ".gitignore" || value == "info/exclude" {
			found = true
			return false
		}
		return true
	})
	return found
}
