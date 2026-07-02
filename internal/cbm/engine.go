// Package cbm wraps the codebase-memory-mcp C engine, linked as a localized
// static archive, behind a small Go API with the locking the engine's
// process-global pipeline state requires.
package cbm

/*
#cgo pkg-config: cbm
#include <stdlib.h>
#include "cbm.h"
#include "mcp/mcp.h"
*/
import "C"

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"unsafe"
)

var (
	globalEngineMutex sync.Mutex
	allocatorOnce     sync.Once
)

// Engine wraps one cbm MCP server handle.
type Engine struct {
	pointer  *C.cbm_mcp_server_t
	project  string
	cacheDir string
	mutex    sync.Mutex
}

type indexArguments struct {
	RepositoryPath string `json:"repo_path"`
	Mode           string `json:"mode"`
	Name           string `json:"name"`
}

type mcpEnvelope struct {
	Content []mcpContent `json:"content"`
	IsError bool         `json:"isError"`
}

type mcpContent struct {
	Text string `json:"text"`
}

// Open initializes and returns an Engine for project. Open and each engine C
// call set CBM_CACHE_DIR to cacheDir under the process-wide lock, so callers
// should pass cacheDir here instead of setting CBM_CACHE_DIR themselves.
func Open(project string, cacheDir string) (*Engine, error) {
	globalEngineMutex.Lock()
	defer globalEngineMutex.Unlock()

	restoreCacheDir, err := setCacheDirLocked(cacheDir)
	if err != nil {
		slog.Error("set cbm cache directory failed", "project", project, "cache_dir", cacheDir, "err", err)
		return nil, fmt.Errorf("set cbm cache directory: %w", err)
	}
	defer restoreCacheDir()

	allocatorOnce.Do(func() {
		C.cbm_alloc_init()
	})

	cProject := C.CString(project)
	defer C.free(unsafe.Pointer(cProject))

	pointer := C.cbm_mcp_server_new(cProject)
	if pointer == nil {
		return nil, fmt.Errorf("cbm_mcp_server_new returned nil")
	}

	C.cbm_mcp_server_set_project(pointer, cProject)

	return &Engine{
		pointer:  pointer,
		project:  project,
		cacheDir: cacheDir,
		mutex:    sync.Mutex{},
	}, nil
}

// Index indexes repositoryPath into the engine project using mode. The global
// engine lock serializes C calls across the process because the engine reads
// process-global cache directory state on each operation. The per-handle lock
// serializes calls on this one engine.
func (engine *Engine) Index(ctx context.Context, repositoryPath string, mode string) error {
	globalEngineMutex.Lock()
	defer globalEngineMutex.Unlock()

	restoreCacheDir, err := setCacheDirLocked(engine.cacheDir)
	if err != nil {
		return fmt.Errorf("set cbm cache directory: %w", err)
	}
	defer restoreCacheDir()

	engine.mutex.Lock()
	defer engine.mutex.Unlock()

	arguments := indexArguments{
		RepositoryPath: repositoryPath,
		Mode:           mode,
		Name:           engine.project,
	}
	argumentsJSON, errorMessage := json.Marshal(arguments)
	if errorMessage != nil {
		slog.ErrorContext(ctx, "marshal cbm index_repository arguments failed", "project", engine.project, "err", errorMessage)
		return fmt.Errorf("marshal index_repository arguments: %w", errorMessage)
	}

	rawJSON, errorMessage := engine.callToolLocked(
		"index_repository",
		string(argumentsJSON),
	)
	if errorMessage != nil {
		slog.ErrorContext(ctx, "cbm index_repository call failed", "project", engine.project, "err", errorMessage)
		return errorMessage
	}

	var envelope mcpEnvelope
	if errorMessage = json.Unmarshal([]byte(rawJSON), &envelope); errorMessage != nil {
		slog.ErrorContext(ctx, "cbm index_repository returned invalid JSON", "project", engine.project, "err", errorMessage)
		return fmt.Errorf("index_repository returned invalid JSON: %w", errorMessage)
	}
	if envelope.IsError {
		indexError := fmt.Errorf("index_repository returned error: %s", envelopeText(envelope))
		slog.ErrorContext(ctx, "cbm index_repository reported error", "project", engine.project, "err", indexError)
		return indexError
	}

	return nil
}

// Tool calls toolName with argumentsJSON and returns the raw MCP envelope JSON.
func (engine *Engine) Tool(toolName string, argumentsJSON string) (string, error) {
	if toolName == "index_repository" {
		return "", fmt.Errorf("index_repository must be called through Index")
	}

	globalEngineMutex.Lock()
	defer globalEngineMutex.Unlock()

	restoreCacheDir, err := setCacheDirLocked(engine.cacheDir)
	if err != nil {
		slog.Error("set cbm cache directory failed", "project", engine.project, "cache_dir", engine.cacheDir, "err", err)
		return "", fmt.Errorf("set cbm cache directory: %w", err)
	}
	defer restoreCacheDir()

	engine.mutex.Lock()
	defer engine.mutex.Unlock()

	return engine.callToolLocked(toolName, argumentsJSON)
}

// Close frees the underlying cbm server handle. Close is safe to call twice.
func (engine *Engine) Close() {
	if engine == nil {
		return
	}

	globalEngineMutex.Lock()
	defer globalEngineMutex.Unlock()

	engine.mutex.Lock()
	defer engine.mutex.Unlock()

	if engine.pointer == nil {
		return
	}

	C.cbm_mcp_server_free(engine.pointer)
	engine.pointer = nil
}

func (engine *Engine) callToolLocked(toolName string, argumentsJSON string) (string, error) {
	if engine.pointer == nil {
		return "", fmt.Errorf("cbm engine is closed")
	}

	cToolName := C.CString(toolName)
	defer C.free(unsafe.Pointer(cToolName))

	cArgumentsJSON := C.CString(argumentsJSON)
	defer C.free(unsafe.Pointer(cArgumentsJSON))

	rawResponse := C.cbm_mcp_handle_tool(engine.pointer, cToolName, cArgumentsJSON)
	if rawResponse == nil {
		return "", fmt.Errorf("%s returned nil", toolName)
	}
	defer C.free(unsafe.Pointer(rawResponse))

	return C.GoString(rawResponse), nil
}

func envelopeText(envelope mcpEnvelope) string {
	textParts := make([]string, 0, len(envelope.Content))
	for _, content := range envelope.Content {
		if content.Text == "" {
			continue
		}
		textParts = append(textParts, content.Text)
	}
	if len(textParts) == 0 {
		return "MCP envelope isError=true"
	}

	return strings.Join(textParts, "\n")
}

func setCacheDirLocked(cacheDir string) (func(), error) {
	if cacheDir == "" {
		return func() {}, nil
	}

	previousValue, hadPreviousValue := os.LookupEnv("CBM_CACHE_DIR")
	if err := os.Setenv("CBM_CACHE_DIR", cacheDir); err != nil {
		return nil, fmt.Errorf("set CBM_CACHE_DIR: %w", err)
	}

	return func() {
		if hadPreviousValue {
			if err := os.Setenv("CBM_CACHE_DIR", previousValue); err != nil {
				slog.Error("restore CBM_CACHE_DIR failed", "err", err)
			}
			return
		}
		if err := os.Unsetenv("CBM_CACHE_DIR"); err != nil {
			slog.Error("unset CBM_CACHE_DIR failed", "err", err)
		}
	}, nil
}
