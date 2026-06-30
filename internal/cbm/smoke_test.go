//go:build darwin && arm64

package cbm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

type toolEnvelope struct {
	StructuredContent queryGraphResult `json:"structuredContent"`
	IsError           bool             `json:"isError"`
}

type queryGraphResult struct {
	Rows []json.RawMessage `json:"rows"`
}

func TestSmokeIndexRepositoryAndQueryGraph(t *testing.T) {
	repositoryPath := t.TempDir()
	writeSmokeRepository(t, repositoryPath)

	cachePath := t.TempDir()
	t.Setenv("CBM_CACHE_DIR", cachePath)

	initializeAllocator()

	server, errorMessage := newEngineServer("smoke")
	if errorMessage != nil {
		t.Fatal(errorMessage)
	}
	defer server.close()

	indexArguments := fmt.Sprintf(
		`{"repo_path":%q,"mode":"fast","name":"smoke"}`,
		repositoryPath,
	)
	indexResponse := callTool(t, server, "index_repository", indexArguments)
	if indexResponse.IsError {
		t.Fatalf("index_repository returned error: %s", indexResponse.RawJSON)
	}

	queryArguments := `{"query":"MATCH (f:Function) RETURN f.name LIMIT 25","project":"smoke","max_rows":25}`
	queryResponse := callTool(t, server, "query_graph", queryArguments)
	if queryResponse.IsError {
		t.Fatalf("query_graph returned error: %s", queryResponse.RawJSON)
	}
	if len(queryResponse.StructuredContent.Rows) == 0 {
		t.Fatalf("query_graph returned no rows: %s", queryResponse.RawJSON)
	}

	t.Logf("query_graph JSON: %s", queryResponse.RawJSON)
}

type toolResponse struct {
	RawJSON string
	toolEnvelope
}

func callTool(
	t *testing.T,
	server *engineServer,
	toolName string,
	argumentsJSON string,
) toolResponse {
	t.Helper()

	rawJSON, errorMessage := server.callTool(toolName, argumentsJSON)
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

func writeSmokeRepository(t *testing.T, repositoryPath string) {
	t.Helper()

	sourcePath := filepath.Join(repositoryPath, "main.go")
	source := `package main

func main() {
	greet("smoke")
}

func greet(name string) string {
	return "hello " + name
}
`

	if errorMessage := os.WriteFile(sourcePath, []byte(source), 0o644); errorMessage != nil {
		t.Fatalf("write smoke repository: %v", errorMessage)
	}
}
