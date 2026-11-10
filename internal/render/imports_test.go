package render

import (
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

// The render package is behind a compile-time wall: it formats view models and
// must never read raw records. The compiler enforces this because the package
// does not import internal/model; this test makes the rule explicit and fails
// with a named violation if anyone re-adds the import.
func TestRenderImportsNoModel(t *testing.T) {
	t.Parallel()
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	fset := token.NewFileSet()
	for _, file := range files {
		parsed, parseErr := parser.ParseFile(fset, file, nil, parser.ImportsOnly)
		if parseErr != nil {
			t.Fatalf("parse %s: %v", file, parseErr)
		}
		for _, imported := range parsed.Imports {
			path := strings.Trim(imported.Path.Value, `"`)
			if strings.HasSuffix(path, "internal/model") || strings.HasSuffix(path, "internal/daemon") {
				t.Fatalf("%s imports %s; the render wall forbids raw record access", file, path)
			}
		}
	}
}
