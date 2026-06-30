//go:build darwin && arm64

package daemon

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

type graphToolEnvelope struct {
	StructuredContent graphToolResult `json:"structuredContent"`
	IsError           bool            `json:"isError"`
}

type graphToolResult struct {
	Rows []json.RawMessage `json:"rows"`
}

func TestManagerGraphToolIndexesAndQueriesCodebase(t *testing.T) {
	manager, cfg, repoPath := newTestManager(t)
	t.Cleanup(func() {
		manager.CloseGraphEngines()
	})
	if err := os.WriteFile(filepath.Join(repoPath, "main.go"), []byte("package main\n\nfunc GraphTarget() string {\n\treturn \"ok\"\n}\n"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexed
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	engine, err := manager.graphEngine(context.Background(), codebase.ID)
	if err != nil {
		t.Fatalf("graphEngine returned error: %v", err)
	}
	if err = engine.Index(context.Background(), repoPath, "fast"); err != nil {
		t.Fatalf("Index returned error: %v", err)
	}

	resultJSON, err := manager.GraphTool(
		context.Background(),
		codebase.ID,
		"query_graph",
		`{"query":"MATCH (f:Function) RETURN f.name LIMIT 25","project":"`+codebase.ID+`","max_rows":200}`,
	)
	if err != nil {
		t.Fatalf("GraphTool returned error: %v", err)
	}

	var envelope graphToolEnvelope
	if err = json.Unmarshal([]byte(resultJSON), &envelope); err != nil {
		t.Fatalf("query_graph returned invalid JSON: %v\n%s", err, resultJSON)
	}
	if envelope.IsError {
		t.Fatalf("query_graph returned error: %s", resultJSON)
	}
	if len(envelope.StructuredContent.Rows) == 0 {
		t.Fatalf("query_graph returned no rows: %s", resultJSON)
	}

	graphPath := filepath.Join(cfg.GraphDir, codebase.ID+".db")
	if _, err = os.Stat(graphPath); err != nil {
		t.Fatalf("Stat(%q) returned error: %v", graphPath, err)
	}

	t.Logf("query_graph JSON: %s", resultJSON)
}
