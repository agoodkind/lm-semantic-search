package cbm

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestEngineIncrementalPrunesRemovedFile proves that a second Index call over
// the same cache directory converges the queryable graph to the current file
// set: a removed file's functions disappear while a surviving file's functions
// remain. This is the behavior the daemon relies on when it re-indexes a
// changed codebase into its per-codebase graph db instead of clearing and
// rebuilding from scratch.
func TestEngineIncrementalPrunesRemovedFile(t *testing.T) {
	repositoryPath := t.TempDir()
	alphaPath := filepath.Join(repositoryPath, "alpha.go")
	betaPath := filepath.Join(repositoryPath, "beta.go")
	writeGoSource(t, alphaPath, `package repo

func AlphaOne() string {
	return AlphaTwo("x")
}

func AlphaTwo(name string) string {
	return "alpha " + name
}
`)
	writeGoSource(t, betaPath, `package repo

func BetaOne() string {
	return "beta"
}
`)

	// Open manages CBM_CACHE_DIR itself from the cacheDir argument, so the test
	// deliberately does not set the env var.
	cacheDir := t.TempDir()

	engine, err := Open("incremental", cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()

	if err = engine.Index(context.Background(), repositoryPath, "fast"); err != nil {
		t.Fatalf("first Index returned error: %v", err)
	}

	firstFunctions := queryFunctionNames(t, engine, "incremental")
	if !slices.Contains(firstFunctions, "BetaOne") {
		t.Fatalf("BetaOne missing from first index; functions: %v", firstFunctions)
	}
	if !slices.Contains(firstFunctions, "AlphaOne") {
		t.Fatalf("AlphaOne missing from first index; functions: %v", firstFunctions)
	}

	if err = os.Remove(betaPath); err != nil {
		t.Fatalf("remove beta.go: %v", err)
	}

	if err = engine.Index(context.Background(), repositoryPath, "fast"); err != nil {
		t.Fatalf("second Index returned error: %v", err)
	}

	secondFunctions := queryFunctionNames(t, engine, "incremental")
	if slices.Contains(secondFunctions, "BetaOne") {
		t.Fatalf("BetaOne still present after removing beta.go; pruning did not fire. functions: %v", secondFunctions)
	}
	if !slices.Contains(secondFunctions, "AlphaOne") {
		t.Fatalf("AlphaOne missing after re-index; functions: %v", secondFunctions)
	}
}

// queryFunctionNames returns the function names from the structured query_graph
// rows, so assertions match graph content rather than envelope formatting.
func queryFunctionNames(t *testing.T, engine *Engine, project string) []string {
	t.Helper()

	arguments, err := json.Marshal(map[string]string{
		"query":   "MATCH (f:Function) RETURN f.name LIMIT 100",
		"project": project,
	})
	if err != nil {
		t.Fatalf("marshal query_graph arguments: %v", err)
	}
	response := callTool(t, engine, "query_graph", string(arguments))
	if response.IsError {
		t.Fatalf("query_graph returned error: %s", response.RawJSON)
	}

	functionNames := make([]string, 0, len(response.StructuredContent.Rows))
	for _, row := range response.StructuredContent.Rows {
		var columns []string
		if err := json.Unmarshal(row, &columns); err != nil {
			t.Fatalf("unmarshal query_graph row %s: %v", row, err)
		}
		if len(columns) == 0 {
			t.Fatalf("query_graph row %s has no columns", row)
		}
		functionNames = append(functionNames, columns[0])
	}
	return functionNames
}

func writeGoSource(t *testing.T, path string, source string) {
	t.Helper()

	if err := os.WriteFile(path, []byte(source), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
