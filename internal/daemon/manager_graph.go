package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/cbm"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
)

type (
	graphToolArguments map[string]json.RawMessage
	graphIndexTask     struct {
		codebaseID    string
		canonicalPath string
		snapshotHash  string
		complete      func(context.Context)
	}
)

var allowedGraphTools = map[string]struct{}{
	"query_graph":      {},
	"trace_path":       {},
	"get_architecture": {},
	"manage_adr":       {},
}

// GraphTool calls a cbm graph engine tool for a tracked codebase.
func (manager *Manager) GraphTool(ctx context.Context, codebaseID string, toolName string, argsJSON string) (string, error) {
	if err := validateGraphToolName(toolName); err != nil {
		return "", err
	}

	engine, err := manager.graphEngine(ctx, codebaseID)
	if err != nil {
		slog.ErrorContext(ctx, "open graph engine failed", "codebase_id", codebaseID, "err", err)
		return "", fmt.Errorf("open graph engine for %s: %w", codebaseID, err)
	}

	enrichedArgs, err := MarshalGraphToolArguments(argsJSON, codebaseID)
	if err != nil {
		return "", err
	}
	resultJSON, err := engine.Tool(toolName, enrichedArgs)
	if err != nil {
		slog.ErrorContext(ctx, "graph tool call failed", "codebase_id", codebaseID, "tool_name", toolName, "err", err)
		return "", fmt.Errorf("call graph tool %s for %s: %w", toolName, codebaseID, err)
	}
	return resultJSON, nil
}

func validateGraphToolName(toolName string) error {
	if _, found := allowedGraphTools[toolName]; found {
		return nil
	}
	return adapterr.NewInvalidArgument(fmt.Sprintf("unsupported graph tool_name %q", toolName))
}

func newGraphIndexTask(codebaseID string, canonicalPath string, snapshotHash string, complete func(context.Context)) *graphIndexTask {
	return &graphIndexTask{
		codebaseID:    codebaseID,
		canonicalPath: canonicalPath,
		snapshotHash:  snapshotHash,
		complete:      complete,
	}
}

func (manager *Manager) runGraphIndexTask(ctx context.Context, task *graphIndexTask) {
	if task == nil {
		return
	}
	manager.recordGraphIndexNonFatal(ctx, task.codebaseID, task.canonicalPath, task.snapshotHash)
	if task.complete != nil {
		task.complete(ctx)
	}
}

func (manager *Manager) graphEngine(ctx context.Context, codebaseID string) (*cbm.Engine, error) {
	manager.graphMutex.Lock()
	defer manager.graphMutex.Unlock()

	if engine, found := manager.graphEngines[codebaseID]; found {
		return engine, nil
	}

	engine, err := cbm.Open(codebaseID, manager.config.GraphDir)
	if err != nil {
		slog.ErrorContext(ctx, "open cbm engine failed", "codebase_id", codebaseID, "err", err)
		return nil, fmt.Errorf("open cbm engine: %w", err)
	}
	manager.graphEngines[codebaseID] = engine
	return engine, nil
}

// CloseGraphEngines closes every cached graph engine. It is safe to call more
// than once.
func (manager *Manager) CloseGraphEngines() {
	manager.graphMutex.Lock()
	defer manager.graphMutex.Unlock()

	for codebaseID, engine := range manager.graphEngines {
		engine.Close()
		delete(manager.graphEngines, codebaseID)
	}
}

func (manager *Manager) closeGraphEngine(codebaseID string) {
	manager.graphMutex.Lock()
	defer manager.graphMutex.Unlock()

	engine, found := manager.graphEngines[codebaseID]
	if !found {
		return
	}
	engine.Close()
	delete(manager.graphEngines, codebaseID)
}

func (manager *Manager) removeGraphFiles(ctx context.Context, codebaseID string) error {
	for _, path := range manager.graphPaths(codebaseID) {
		if err := store.RemoveFile(path); err != nil {
			slog.ErrorContext(ctx, "remove graph file failed", "codebase_id", codebaseID, "path", path, "err", err)
			return fmt.Errorf("remove graph file %s: %w", path, err)
		}
	}
	return nil
}

func (manager *Manager) indexGraphNonFatal(ctx context.Context, codebaseID string, canonicalPath string) error {
	engine, err := manager.graphEngine(ctx, codebaseID)
	if err != nil {
		slog.WarnContext(ctx, "open graph engine failed; continuing without graph index", "codebase_id", codebaseID, "err", err)
		return fmt.Errorf("open graph engine: %w", err)
	}
	if err = engine.Index(ctx, canonicalPath, "fast"); err != nil {
		slog.WarnContext(ctx, "graph indexing failed; continuing with semantic index", "codebase_id", codebaseID, "path", canonicalPath, "err", err)
		return fmt.Errorf("index graph: %w", err)
	}
	return nil
}

func (manager *Manager) recordGraphIndexNonFatal(ctx context.Context, codebaseID string, canonicalPath string, snapshotHash string) {
	if manager.graphIndexHook != nil {
		manager.graphIndexHook()
	}
	err := manager.indexGraphNonFatal(ctx, codebaseID, canonicalPath)
	if err == nil {
		manager.updateGraphState(ctx, codebaseID, model.GraphStateReady, snapshotHash)
		return
	}
	manager.updateGraphState(ctx, codebaseID, model.GraphStateStale, "")
}

func (manager *Manager) updateGraphState(ctx context.Context, codebaseID string, graphState model.GraphState, snapshotHash string) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	codebase, found := manager.codebases[codebaseID]
	if !found {
		return
	}
	codebase.GraphState = graphState
	if snapshotHash != "" {
		codebase.GraphSnapshotHash = snapshotHash
	}
	codebase.UpdatedAt = clock.Now()
	manager.codebases[codebaseID] = codebase
	if err := manager.saveLocked(); err != nil {
		slog.ErrorContext(ctx, "write registry after graph state update failed", "codebase_id", codebaseID, "err", err)
	}
}

func (manager *Manager) shouldReconcileGraph(codebaseID string, currentSnapshotHash string, presence collectionPresence) bool {
	if presence != collectionPresencePresent {
		return false
	}

	manager.mu.Lock()
	defer manager.mu.Unlock()

	codebase, found := manager.codebases[codebaseID]
	if !found {
		return false
	}
	if codebase.Kind != model.CodebaseKindCode {
		return false
	}
	graphState := codebase.GraphState
	if graphState == "" {
		graphState = model.GraphStateAbsent
	}
	return graphState != model.GraphStateReady || codebase.GraphSnapshotHash != currentSnapshotHash
}

func snapshotHashForGraph(snapshot merkle.Snapshot, configDigest string) string {
	snapshot.ConfigDigest = configDigest
	return snapshot.Hash()
}

func (manager *Manager) graphStatusLine(codebase model.Codebase) string {
	if codebase.Kind != model.CodebaseKindCode {
		return ""
	}
	graphState := codebase.GraphState
	if graphState == "" {
		graphState = model.GraphStateAbsent
	}

	matchLabel := "semantic snapshot unknown"
	snapshot, err := merkle.ReadSnapshot(manager.snapshotPathForCodebase(codebase))
	if err == nil {
		currentHash := snapshotHashForGraph(snapshot, codebase.EffectiveConfig.IgnoreDigest)
		if codebase.GraphSnapshotHash == currentHash {
			matchLabel = "matches semantic snapshot"
		} else {
			matchLabel = "does not match semantic snapshot"
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		matchLabel = "semantic snapshot unreadable"
	}
	return fmt.Sprintf("🕸️ Graph: %s, %s", graphState, matchLabel)
}

func (manager *Manager) graphDiagnostic(codebase model.Codebase) string {
	if codebase.Kind != model.CodebaseKindCode {
		return ""
	}
	graphState := codebase.GraphState
	if graphState == "" {
		graphState = model.GraphStateAbsent
	}

	snapshot, err := merkle.ReadSnapshot(manager.snapshotPathForCodebase(codebase))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			if graphState == model.GraphStateReady {
				return codebase.CanonicalPath + ": graph ready but semantic snapshot is missing"
			}
			return fmt.Sprintf("%s: graph %s and semantic snapshot is missing", codebase.CanonicalPath, graphState)
		}
		return fmt.Sprintf("%s: graph %s and semantic snapshot is unreadable", codebase.CanonicalPath, graphState)
	}
	currentHash := snapshotHashForGraph(snapshot, codebase.EffectiveConfig.IgnoreDigest)
	if graphState == model.GraphStateReady && codebase.GraphSnapshotHash == currentHash {
		return ""
	}
	if codebase.GraphSnapshotHash == currentHash {
		return fmt.Sprintf("%s: graph %s but matches the semantic snapshot", codebase.CanonicalPath, graphState)
	}
	return fmt.Sprintf("%s: graph %s and does not match the semantic snapshot", codebase.CanonicalPath, graphState)
}

func (manager *Manager) graphPaths(codebaseID string) []string {
	basePath := filepath.Join(manager.config.GraphDir, codebaseID+".db")
	return []string{
		basePath,
		basePath + "-wal",
		basePath + "-shm",
	}
}

// MarshalGraphToolArguments returns graph tool arguments with the daemon-owned project id.
func MarshalGraphToolArguments(argsJSON string, codebaseID string) (string, error) {
	var args graphToolArguments
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		slog.Error("unmarshal graph tool arguments failed", "codebase_id", codebaseID, "err", err)
		return "", fmt.Errorf("unmarshal graph tool arguments: %w", err)
	}

	projectJSON, err := json.Marshal(codebaseID)
	if err != nil {
		slog.Error("marshal graph project argument failed", "codebase_id", codebaseID, "err", err)
		return "", fmt.Errorf("marshal graph project argument: %w", err)
	}
	args["project"] = projectJSON

	enrichedArgs, err := json.Marshal(args)
	if err != nil {
		slog.Error("marshal graph tool arguments failed", "codebase_id", codebaseID, "err", err)
		return "", fmt.Errorf("marshal graph tool arguments: %w", err)
	}
	return string(enrichedArgs), nil
}
