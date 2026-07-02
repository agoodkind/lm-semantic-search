package cbm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

const queryGraphFunctionsJSON = `{"query":"MATCH (f:Function) RETURN f.name LIMIT 25","project":"rt"}`

type toolEnvelope struct {
	StructuredContent queryGraphResult `json:"structuredContent"`
	IsError           bool             `json:"isError"`
}

type queryGraphResult struct {
	Rows []json.RawMessage `json:"rows"`
}

type toolResponse struct {
	RawJSON string
	toolEnvelope
}

func TestEngineRoundTripIndexesAndQueriesGraph(t *testing.T) {
	repositoryPath := t.TempDir()
	writeTestRepository(t, repositoryPath, "roundtrip")

	cacheDir := t.TempDir()
	t.Setenv("CBM_CACHE_DIR", cacheDir)

	engine, errorMessage := Open("rt", cacheDir)
	if errorMessage != nil {
		t.Fatal(errorMessage)
	}
	defer engine.Close()

	if errorMessage = engine.Index(context.Background(), repositoryPath, "fast"); errorMessage != nil {
		t.Fatalf("Index returned error: %v", errorMessage)
	}

	queryResponse := callTool(t, engine, "query_graph", queryGraphFunctionsJSON)
	if queryResponse.IsError {
		t.Fatalf("query_graph returned error: %s", queryResponse.RawJSON)
	}
	if len(queryResponse.StructuredContent.Rows) == 0 {
		t.Fatalf("query_graph returned no rows: %s", queryResponse.RawJSON)
	}

	t.Logf("query_graph JSON: %s", queryResponse.RawJSON)
}

func TestEngineSerializesConcurrentIndexesAndSingleHandleQueries(t *testing.T) {
	t.Setenv("CBM_CACHE_DIR", t.TempDir())

	const engineCount = 4

	engines := make([]*Engine, 0, engineCount)
	repositoryPaths := make([]string, 0, engineCount)
	for i := 0; i < engineCount; i++ {
		projectName := fmt.Sprintf("concurrent_%d", i)
		engine, errorMessage := Open(projectName, t.TempDir())
		if errorMessage != nil {
			t.Fatalf("Open(%q): %v", projectName, errorMessage)
		}
		defer engine.Close()
		engines = append(engines, engine)

		repositoryPath := t.TempDir()
		writeTestRepository(t, repositoryPath, fmt.Sprintf("repo%d", i))
		repositoryPaths = append(repositoryPaths, repositoryPath)
	}

	indexErrors := make(chan error, engineCount)
	var waitGroup sync.WaitGroup
	for i, engine := range engines {
		waitGroup.Add(1)
		go func(i int, engine *Engine) {
			defer waitGroup.Done()

			if errorMessage := engine.Index(context.Background(), repositoryPaths[i], "fast"); errorMessage != nil {
				indexErrors <- fmt.Errorf("engine %d index: %w", i, errorMessage)
			}
		}(i, engine)
	}
	waitGroup.Wait()
	close(indexErrors)

	for errorMessage := range indexErrors {
		t.Error(errorMessage)
	}

	queryEngine := engines[0]
	queryProjectJSON := `{"query":"MATCH (f:Function) RETURN f.name LIMIT 25","project":"concurrent_0"}`

	const queryCount = 8

	queryErrors := make(chan error, queryCount)
	waitGroup = sync.WaitGroup{}
	for i := 0; i < queryCount; i++ {
		waitGroup.Add(1)
		go func(i int) {
			defer waitGroup.Done()

			rawJSON, errorMessage := queryEngine.Tool("query_graph", queryProjectJSON)
			if errorMessage != nil {
				queryErrors <- fmt.Errorf("query %d: %w", i, errorMessage)
				return
			}

			var envelope toolEnvelope
			if errorMessage = json.Unmarshal([]byte(rawJSON), &envelope); errorMessage != nil {
				queryErrors <- fmt.Errorf("query %d invalid JSON: %w", i, errorMessage)
				return
			}
			if envelope.IsError {
				queryErrors <- fmt.Errorf("query %d returned error: %s", i, rawJSON)
				return
			}
			if len(envelope.StructuredContent.Rows) == 0 {
				queryErrors <- fmt.Errorf("query %d returned no rows: %s", i, rawJSON)
			}
		}(i)
	}
	waitGroup.Wait()
	close(queryErrors)

	for errorMessage := range queryErrors {
		t.Error(errorMessage)
	}
}

func TestEngineIndexesIntoItsCacheDirDespiteEnvChange(t *testing.T) {
	repositoryPath := t.TempDir()
	writeTestRepository(t, repositoryPath, "envrace")

	cacheDir := t.TempDir()
	processEnvDir := t.TempDir()
	t.Setenv("CBM_CACHE_DIR", processEnvDir)

	engine, errorMessage := Open("envrace", cacheDir)
	if errorMessage != nil {
		t.Fatal(errorMessage)
	}
	defer engine.Close()

	if errorMessage = engine.Index(context.Background(), repositoryPath, "fast"); errorMessage != nil {
		t.Fatalf("Index returned error: %v", errorMessage)
	}

	cacheEntries := readDirectoryEntries(t, cacheDir)
	if len(cacheEntries) == 0 {
		t.Fatalf("expected engine cache directory %s to be non-empty", cacheDir)
	}

	processEnvEntries := readDirectoryEntries(t, processEnvDir)
	if len(processEnvEntries) != 0 {
		t.Fatalf("expected process-env cache directory %s to stay empty; found %d entries", processEnvDir, len(processEnvEntries))
	}
}

func callTool(t *testing.T, engine *Engine, toolName string, argumentsJSON string) toolResponse {
	t.Helper()

	rawJSON, errorMessage := engine.Tool(toolName, argumentsJSON)
	if errorMessage != nil {
		t.Fatal(errorMessage)
	}

	var envelope toolEnvelope
	if errorMessage = json.Unmarshal([]byte(rawJSON), &envelope); errorMessage != nil {
		t.Fatalf("%s returned invalid JSON: %v\n%s", toolName, errorMessage, rawJSON)
	}

	return toolResponse{
		RawJSON:      rawJSON,
		toolEnvelope: envelope,
	}
}

func readDirectoryEntries(t *testing.T, path string) []os.DirEntry {
	t.Helper()

	entries, errorMessage := os.ReadDir(path)
	if errorMessage != nil {
		t.Fatalf("read directory %s: %v", path, errorMessage)
	}
	return entries
}

func writeTestRepository(t *testing.T, repositoryPath string, packageName string) {
	t.Helper()

	sourcePath := filepath.Join(repositoryPath, "main.go")
	source := fmt.Sprintf(`package %s

func First() string {
	return Second("test")
}

func Second(name string) string {
	return "hello " + name
}
`, packageName)

	if errorMessage := os.WriteFile(sourcePath, []byte(source), 0o644); errorMessage != nil {
		t.Fatalf("write test repository: %v", errorMessage)
	}
}
