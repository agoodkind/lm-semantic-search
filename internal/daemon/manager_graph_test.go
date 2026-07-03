package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/cbm"
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

	engine, release, err := manager.graphEngine(context.Background(), codebase.ID)
	if err != nil {
		t.Fatalf("graphEngine returned error: %v", err)
	}
	defer release()
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
	if _, _, _, _, err := manager.StartIndex(context.Background(), repoPath, testClientInfo(), defaultIndexConfig(), false, emptyAdmissionBudget); err != nil {
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
	if err = manager.clearGraphCache(context.Background(), codebase.ID); err != nil {
		t.Fatalf("clearGraphCache returned error: %v", err)
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

func TestClearGraphCacheRejectsNewOpenAndWaitsForActiveOperation(t *testing.T) {
	manager, cfg, _ := newTestManager(t)
	codebaseID := "graph-lifecycle"
	graphPath := filepath.Join(cfg.GraphDir, codebaseID+".db")
	if err := os.WriteFile(graphPath, []byte("graph"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	release, err := manager.beginGraphOperation(codebaseID)
	if err != nil {
		t.Fatalf("beginGraphOperation returned error: %v", err)
	}

	clearDone := make(chan error, 1)
	go func() {
		clearDone <- manager.clearGraphCache(context.Background(), codebaseID)
	}()
	waitForGraphClearing(t, manager, codebaseID)

	if _, _, err = manager.graphEngine(context.Background(), codebaseID); err == nil {
		t.Fatal("graphEngine returned nil error while graph cache was clearing")
	} else if !strings.Contains(err.Error(), "being cleared") {
		t.Fatalf("graphEngine error = %v, want clear-in-progress error", err)
	}
	if _, statErr := os.Stat(graphPath); statErr != nil {
		t.Fatalf("graph file was removed before active operation released: %v", statErr)
	}

	select {
	case err = <-clearDone:
		t.Fatalf("clearGraphCache completed before active operation released: %v", err)
	case <-time.After(50 * time.Millisecond):
	}

	release()
	select {
	case err = <-clearDone:
		if err != nil {
			t.Fatalf("clearGraphCache returned error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("clearGraphCache did not finish after active operation released")
	}
	if _, statErr := os.Stat(graphPath); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("Stat(%q) error = %v, want os.ErrNotExist", graphPath, statErr)
	}
}

func TestIndexGraphNonFatalReturnsOnCancelAndClearWaitsForWorker(t *testing.T) {
	manager, cfg, repoPath := newTestManager(t)
	codebaseID := "graph-cancel"
	entered := make(chan struct{})
	releaseWorker := make(chan struct{})
	manager.graphIndex = func(ctx context.Context, engine *cbm.Engine, canonicalPath string, mode string) error {
		close(entered)
		<-releaseWorker
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	go func() {
		result <- manager.indexGraphNonFatal(ctx, codebaseID, repoPath, nil)
	}()

	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("fake graph index did not start")
	}
	cancel()
	select {
	case err := <-result:
		if err == nil {
			t.Fatal("indexGraphNonFatal returned nil error after cancellation")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("indexGraphNonFatal did not return promptly after cancellation")
	}

	graphPath := filepath.Join(cfg.GraphDir, codebaseID+".db")
	if err := os.WriteFile(graphPath, []byte("graph"), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	clearDone := make(chan error, 1)
	go func() {
		clearDone <- manager.clearGraphCache(context.Background(), codebaseID)
	}()

	select {
	case err := <-clearDone:
		t.Fatalf("clearGraphCache completed while fake graph index was blocked: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	close(releaseWorker)
	select {
	case err := <-clearDone:
		if err != nil {
			t.Fatalf("clearGraphCache returned error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("clearGraphCache did not finish after fake graph index released")
	}
	if _, statErr := os.Stat(graphPath); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("Stat(%q) error = %v, want os.ErrNotExist", graphPath, statErr)
	}
}

func TestMarshalGraphToolArgumentsAcceptsNull(t *testing.T) {
	resultJSON, err := MarshalGraphToolArguments("null", "project-id")
	if err != nil {
		t.Fatalf("MarshalGraphToolArguments returned error: %v", err)
	}
	var args graphToolArguments
	if err = json.Unmarshal([]byte(resultJSON), &args); err != nil {
		t.Fatalf("Unmarshal returned error: %v", err)
	}
	var project string
	if err = json.Unmarshal(args["project"], &project); err != nil {
		t.Fatalf("project unmarshal returned error: %v", err)
	}
	if project != "project-id" {
		t.Fatalf("project = %q, want project-id", project)
	}
}

func TestUpdateGraphStateClearsSnapshotHashWhenNotReady(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	codebase := newCodebaseRecord(repoPath)
	codebase.GraphState = model.GraphStateReady
	codebase.GraphSnapshotHash = "old-hash"
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	manager.updateGraphState(context.Background(), codebase.ID, model.GraphStateStale, "")

	manager.mu.Lock()
	recorded := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if recorded.GraphSnapshotHash != "" {
		t.Fatalf("GraphSnapshotHash = %q, want empty", recorded.GraphSnapshotHash)
	}
}

func TestUpdateGraphStateRecordsSuccessfulBuildTime(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	codebase := newCodebaseRecord(repoPath)
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	untouched := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if !untouched.GraphUpdatedAt.IsZero() {
		t.Fatalf("GraphUpdatedAt = %v, want zero before graph state update", untouched.GraphUpdatedAt)
	}

	manager.updateGraphState(context.Background(), codebase.ID, model.GraphStateReady, "ready-hash")

	manager.mu.Lock()
	ready := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if ready.GraphUpdatedAt.IsZero() {
		t.Fatal("GraphUpdatedAt is zero after ready graph state update")
	}
	successfulBuildTime := ready.GraphUpdatedAt

	manager.updateGraphState(context.Background(), codebase.ID, model.GraphStateStale, "")

	manager.mu.Lock()
	stale := manager.codebases[codebase.ID]
	manager.mu.Unlock()
	if !stale.GraphUpdatedAt.Equal(successfulBuildTime) {
		t.Fatalf("GraphUpdatedAt = %v, want preserved successful build time %v", stale.GraphUpdatedAt, successfulBuildTime)
	}
	if stale.GraphSnapshotHash != "" {
		t.Fatalf("GraphSnapshotHash = %q, want empty", stale.GraphSnapshotHash)
	}
}

func TestResolveGetIndexViewPopulatesGraphFields(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	now := time.Date(2026, 7, 3, 12, 0, 0, 0, time.UTC)
	useRelativeTimeNowForTest(t, now)

	cases := []struct {
		name               string
		codebase           model.Codebase
		wantUpdatedAt      string
		wantReadyNoTime    bool
		wantNotBuilt       bool
		wantAllFieldsEmpty bool
	}{
		{
			name: "ever built",
			codebase: model.Codebase{
				CanonicalPath:  repoPath,
				Kind:           model.CodebaseKindCode,
				Status:         model.CodebaseStatusIndexed,
				GraphState:     model.GraphStateStale,
				GraphUpdatedAt: now.Add(-6 * time.Minute),
				LastSuccessfulRun: &model.IndexRunSummary{
					CompletedAt: now,
				},
			},
			wantUpdatedAt: "6 minutes ago",
		},
		{
			name: "ready no time",
			codebase: model.Codebase{
				CanonicalPath: repoPath,
				Kind:          model.CodebaseKindCode,
				Status:        model.CodebaseStatusIndexed,
				GraphState:    model.GraphStateReady,
			},
			wantReadyNoTime: true,
		},
		{
			name: "never built",
			codebase: model.Codebase{
				CanonicalPath: repoPath,
				Kind:          model.CodebaseKindCode,
				Status:        model.CodebaseStatusIndexed,
				GraphState:    model.GraphStateAbsent,
			},
			wantNotBuilt: true,
		},
		{
			name: "non code",
			codebase: model.Codebase{
				CanonicalPath: "chat:///thread-alpha",
				Kind:          model.CodebaseKindDocument,
				Status:        model.CodebaseStatusIndexed,
			},
			wantAllFieldsEmpty: true,
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			got := manager.resolveGetIndexView(
				testCase.codebase.CanonicalPath,
				true,
				&testCase.codebase,
				nil,
				dependencyHealth{},
				collectionNotApplicable,
				nil,
				nil,
			)
			if got.Status.GraphUpdatedAt != testCase.wantUpdatedAt {
				t.Fatalf("GraphUpdatedAt = %q, want %q", got.Status.GraphUpdatedAt, testCase.wantUpdatedAt)
			}
			if got.Status.GraphReadyNoTime != testCase.wantReadyNoTime {
				t.Fatalf("GraphReadyNoTime = %t, want %t", got.Status.GraphReadyNoTime, testCase.wantReadyNoTime)
			}
			if got.Status.GraphNotBuilt != testCase.wantNotBuilt {
				t.Fatalf("GraphNotBuilt = %t, want %t", got.Status.GraphNotBuilt, testCase.wantNotBuilt)
			}
			if testCase.wantAllFieldsEmpty && (got.Status.GraphUpdatedAt != "" || got.Status.GraphReadyNoTime || got.Status.GraphNotBuilt) {
				t.Fatalf("non-code graph fields = %+v, want all zero", got.Status)
			}
		})
	}
}

func TestGraphDiagnosticReportsSnapshotMatchOnlyWhenReady(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
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
	snapshotHash := snapshotHashForGraph(snapshot, codebase.EffectiveConfig.IgnoreDigest)
	if err = merkle.WriteSnapshot(codebase.MerkleSnapshotPath, snapshot); err != nil {
		t.Fatalf("WriteSnapshot returned error: %v", err)
	}
	codebase.GraphState = model.GraphStateStale
	codebase.GraphSnapshotHash = snapshotHash

	diagnostic := manager.graphDiagnostic(codebase)
	if strings.Contains(diagnostic, "but matches the semantic snapshot") {
		t.Fatalf("graphDiagnostic = %q, want no ready-only match label", diagnostic)
	}
}

func TestGraphStatusMissingSnapshotDoesNotLogError(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	codebase := newCodebaseRecord(repoPath)
	codebase.MerkleSnapshotPath = filepath.Join(t.TempDir(), "missing.json")

	handler := &recordingSlogHandler{}
	previousLogger := slog.Default()
	slog.SetDefault(slog.New(handler))
	t.Cleanup(func() {
		slog.SetDefault(previousLogger)
	})

	_ = manager.graphDiagnostic(codebase)

	if handler.errorCount() != 0 {
		t.Fatalf("logged %d ERROR entries for missing snapshot, want 0", handler.errorCount())
	}
}

func waitForGraphClearing(t *testing.T, manager *Manager, codebaseID string) {
	t.Helper()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		manager.graphMutex.Lock()
		state := manager.graphLifecycle[codebaseID]
		clearing := state != nil && state.clearing
		manager.graphMutex.Unlock()
		if clearing {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("graph cache did not enter clearing state")
}

type recordingSlogHandler struct {
	mutex  sync.Mutex
	errors int
}

func (handler *recordingSlogHandler) Enabled(context.Context, slog.Level) bool {
	return true
}

func (handler *recordingSlogHandler) Handle(ctx context.Context, record slog.Record) error {
	_ = ctx
	if record.Level >= slog.LevelError {
		handler.mutex.Lock()
		handler.errors++
		handler.mutex.Unlock()
	}
	return nil
}

func (handler *recordingSlogHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	_ = attrs
	return handler
}

func (handler *recordingSlogHandler) WithGroup(name string) slog.Handler {
	_ = name
	return handler
}

func (handler *recordingSlogHandler) errorCount() int {
	handler.mutex.Lock()
	defer handler.mutex.Unlock()
	return handler.errors
}
