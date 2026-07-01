package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
)

type graphToolEnvelope struct {
	StructuredContent graphToolResult `json:"structuredContent"`
	IsError           bool            `json:"isError"`
}

type graphToolResult struct {
	Rows []json.RawMessage `json:"rows"`
}

func TestValidateGraphToolNameAllowsReadOnlyToolsOnly(t *testing.T) {
	for _, toolName := range []string{"query_graph", "trace_path", "get_architecture", "manage_adr"} {
		if err := validateGraphToolName(toolName); err != nil {
			t.Fatalf("validateGraphToolName(%q) returned error: %v", toolName, err)
		}
	}

	err := validateGraphToolName("delete_project")
	if err == nil {
		t.Fatal("validateGraphToolName accepted delete_project, want invalid argument")
	}
	var adapterErr *adapterr.AdapterError
	if !errors.As(err, &adapterErr) {
		t.Fatalf("validateGraphToolName returned %T, want AdapterError", err)
	}
	if adapterErr.Class != adapterr.ClassInvalidArgument {
		t.Fatalf("adapter error class = %q, want %q", adapterErr.Class, adapterr.ClassInvalidArgument)
	}
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

func TestRunJobAsyncRunsGraphIndexAfterSemanticLockAndSlotRelease(t *testing.T) {
	manager, _ := newTestManagerWithCap(t, 1)
	manager.semantic = &fakeSemantic{
		count: func(context.Context, string) (int32, error) {
			return 1, nil
		},
	}

	observed := make(chan string, 1)
	manager.graphIndexHook = func() {
		manager.syncLock.mu.Lock()
		refcount := manager.syncLock.refcount
		manager.syncLock.mu.Unlock()
		slotCount := len(manager.indexSlots)
		if refcount != 0 || slotCount != 0 {
			observed <- fmt.Sprintf("syncLock refcount = %d, index slot count = %d", refcount, slotCount)
			return
		}
		observed <- ""
	}

	repoPath := newCapTestRepo(t)
	if _, _, _, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), defaultIndexConfig(), false); err != nil {
		t.Fatalf("StartIndex returned error: %v", err)
	}

	select {
	case message := <-observed:
		if message != "" {
			t.Fatal(message)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("graph index hook was not called")
	}
	waitForCodebaseStatus(t, manager, repoPath, model.CodebaseStatusIndexed)
}

func TestGraphStateRecordsReadyAndReconcilesStaleGraph(t *testing.T) {
	manager, cfg, repoPath := newTestManager(t)
	t.Cleanup(func() {
		manager.CloseGraphEngines()
	})

	codebase := newCodebaseRecord(repoPath)
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = defaultIndexConfig()
	codebase.EffectiveConfig.IgnoreDigest = digestIndexConfig(codebase.EffectiveConfig)
	codebase.MerkleSnapshotPath = manager.merklePath(codebase.ID)

	snapshot, err := merkle.Capture(
		context.Background(),
		manager.indexability,
		codebase.ID,
		repoPath,
		codebase.EffectiveConfig,
	)
	if err != nil {
		t.Fatalf("Capture returned error: %v", err)
	}
	snapshot.ConfigDigest = codebase.EffectiveConfig.IgnoreDigest
	snapshotHash := snapshot.Hash()
	if err = merkle.WriteSnapshot(codebase.MerkleSnapshotPath, snapshot); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}

	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	manager.recordGraphIndexNonFatal(context.Background(), codebase.ID, repoPath, snapshotHash)

	manager.mu.Lock()
	recorded := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if recorded.GraphState != model.GraphStateReady {
		t.Fatalf("GraphState = %q, want %q", recorded.GraphState, model.GraphStateReady)
	}
	if recorded.GraphSnapshotHash == "" {
		t.Fatal("GraphSnapshotHash is empty, want hash after graph build")
	}
	if recorded.GraphSnapshotHash != snapshotHash {
		t.Fatalf("GraphSnapshotHash = %q, want %q", recorded.GraphSnapshotHash, snapshotHash)
	}

	manager.mu.Lock()
	recorded.GraphState = model.GraphStateStale
	manager.codebases[codebase.ID] = recorded
	manager.mu.Unlock()
	manager.closeGraphEngine(codebase.ID)
	if err = manager.removeGraphFiles(context.Background(), codebase.ID); err != nil {
		t.Fatalf("removeGraphFiles returned error: %v", err)
	}
	if !manager.shouldReconcileGraph(codebase.ID, snapshotHash, collectionPresencePresent) {
		t.Fatal("shouldReconcileGraph returned false for stale graph, want true")
	}

	manager.recordGraphIndexNonFatal(context.Background(), codebase.ID, repoPath, snapshotHash)

	manager.mu.Lock()
	reconciled := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if reconciled.GraphState != model.GraphStateReady {
		t.Fatalf("GraphState after reconcile = %q, want %q", reconciled.GraphState, model.GraphStateReady)
	}
	graphPath := filepath.Join(cfg.GraphDir, codebase.ID+".db")
	if _, err = os.Stat(graphPath); err != nil {
		t.Fatalf("Stat(%q) returned error after reconcile: %v", graphPath, err)
	}
}
