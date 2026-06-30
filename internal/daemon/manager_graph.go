package daemon

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"path/filepath"

	"goodkind.io/lm-semantic-search/internal/cbm"
	"goodkind.io/lm-semantic-search/internal/store"
)

type graphToolArguments map[string]json.RawMessage

// GraphTool calls a cbm graph engine tool for a tracked codebase.
func (manager *Manager) GraphTool(ctx context.Context, codebaseID string, toolName string, argsJSON string) (string, error) {
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

func (manager *Manager) graphEngine(ctx context.Context, codebaseID string) (*cbm.Engine, error) {
	manager.graphMutex.Lock()
	defer manager.graphMutex.Unlock()

	if engine, found := manager.graphEngines[codebaseID]; found {
		return engine, nil
	}

	engine, err := cbm.Open(codebaseID)
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
