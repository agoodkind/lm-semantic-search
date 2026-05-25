// Package tssnapshot reads the upstream TS MCP codebase snapshot so the Go
// daemon can answer queries about codebases the upstream adapter indexed.
//
// The Go daemon never imports, copies, or rewrites the TS snapshot file. The
// snapshot is a read-only oracle. Each entry is synthesized into an in-memory
// model.Codebase only when a request comes in for a path the Go registry
// does not own.
package tssnapshot

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"goodkind.io/claude-context-go/internal/model"
	"goodkind.io/claude-context-go/internal/tshash"
)

// Status is one of the TS adapter's snapshot status values.
type Status string

const (
	// StatusIndexed marks a codebase fully indexed by the TS adapter.
	StatusIndexed Status = "indexed"
	// StatusIndexFailed marks a codebase whose TS index attempt failed.
	StatusIndexFailed Status = "indexfailed"
	// StatusIndexing marks a codebase whose TS index attempt was interrupted.
	StatusIndexing Status = "indexing"
)

// Entry is one entry in the TS snapshot.
type Entry struct {
	Status                  string   `json:"status"`
	IndexedFiles            int32    `json:"indexedFiles"`
	TotalChunks             int32    `json:"totalChunks"`
	IndexStatus             string   `json:"indexStatus"`
	IndexingPercentage      float64  `json:"indexingPercentage"`
	ErrorMessage            string   `json:"errorMessage"`
	LastAttemptedPercentage float64  `json:"lastAttemptedPercentage"`
	RequestSplitter         string   `json:"requestSplitter"`
	RequestCustomExtensions []string `json:"requestCustomExtensions"`
	RequestIgnorePatterns   []string `json:"requestIgnorePatterns"`
	LastUpdated             string   `json:"lastUpdated"`
}

type fileFormat struct {
	FormatVersion string           `json:"formatVersion"`
	Codebases     map[string]Entry `json:"codebases"`
	LastUpdated   string           `json:"lastUpdated"`
}

// Path returns the TS snapshot path rooted at the supplied context directory.
// Passing an empty contextRoot falls back to "$HOME/.context".
func Path(contextRoot string) (string, error) {
	if contextRoot != "" {
		return filepath.Join(contextRoot, "mcp-codebase-snapshot.json"), nil
	}
	homeDir, err := os.UserHomeDir()
	if err != nil {
		slog.Error("resolve user home directory failed", "err", err)
		return "", fmt.Errorf("resolve user home directory: %w", err)
	}
	return filepath.Join(homeDir, ".context", "mcp-codebase-snapshot.json"), nil
}

// Load parses the upstream TS snapshot at path. A missing file returns
// (nil, nil) so callers can treat absence as "no TS data" without filtering.
func Load(path string) (map[string]Entry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		slog.Error("read TS snapshot failed", "path", path, "err", err)
		return nil, fmt.Errorf("read TS snapshot %s: %w", path, err)
	}

	var snapshot fileFormat
	if err := json.Unmarshal(data, &snapshot); err != nil {
		slog.Error("unmarshal TS snapshot failed", "path", path, "err", err)
		return nil, fmt.Errorf("unmarshal TS snapshot %s: %w", path, err)
	}
	return snapshot.Codebases, nil
}

// Synthesize converts one TS snapshot entry into an in-memory model.Codebase
// that mirrors what the Go daemon would have produced if it had done the
// indexing itself. The returned record is never persisted; callers use it to
// answer GetIndex and SearchCode for paths the Go registry does not own.
//
// canonicalPath must already be resolved to its absolute form.
// hybridMode is the daemon's current HYBRID_MODE setting; both TS and Go use
// the same Milvus collection name when this flag matches the original index.
func Synthesize(canonicalPath string, entry Entry, hybridMode bool) model.Codebase {
	updatedAt := parseTimestamp(entry.LastUpdated)
	codebase := model.Codebase{
		ID:                tsCodebaseID(canonicalPath),
		CanonicalPath:     canonicalPath,
		Aliases:           []string{canonicalPath},
		Status:            model.CodebaseStatusNotIndexed,
		ActiveJobID:       "",
		LastSuccessfulRun: nil,
		LastFailedRun:     nil,
		UpdatedAt:         updatedAt,
		EffectiveConfig: model.IndexConfig{
			SplitterType:       defaultIfEmpty(entry.RequestSplitter, "ast"),
			SplitterChunkSize:  2500,
			SplitterOverlap:    300,
			Extensions:         append([]string{}, entry.RequestCustomExtensions...),
			IgnorePatterns:     append([]string{}, entry.RequestIgnorePatterns...),
			IgnoreDigest:       "",
			EmbeddingProvider:  "",
			EmbeddingModel:     "",
			EmbeddingDimension: 0,
			VectorBackend:      "milvus",
			Hybrid:             hybridMode,
		},
		CollectionName:        tsCollectionName(canonicalPath, hybridMode),
		LegacyCollectionNames: nil,
		MerkleSnapshotPath:    "",
	}

	switch Status(entry.Status) {
	case StatusIndexed:
		codebase.Status = model.CodebaseStatusIndexed
		codebase.LastSuccessfulRun = &model.IndexRunSummary{
			IndexedFiles: entry.IndexedFiles,
			TotalChunks:  entry.TotalChunks,
			Status:       defaultIfEmpty(entry.IndexStatus, "completed"),
			CompletedAt:  updatedAt,
		}
	case StatusIndexFailed:
		codebase.Status = model.CodebaseStatusFailed
		codebase.LastFailedRun = &model.IndexRunFailure{
			Message:                 defaultIfEmpty(entry.ErrorMessage, "TS adapter reported indexfailed"),
			LastAttemptedPercentage: int32(entry.LastAttemptedPercentage),
			FailedAt:                updatedAt,
		}
	case StatusIndexing:
		codebase.Status = model.CodebaseStatusFailed
		codebase.LastFailedRun = &model.IndexRunFailure{
			Message:                 "Indexing was interrupted (TS snapshot reported in-progress)",
			LastAttemptedPercentage: int32(entry.IndexingPercentage),
			FailedAt:                updatedAt,
		}
	default:
		codebase.Status = model.CodebaseStatusStale
	}
	return codebase
}

func tsCollectionName(path string, hybrid bool) string {
	prefix := "code_chunks"
	if hybrid {
		prefix = "hybrid_code_chunks"
	}
	return prefix + "_" + tshash.PathPrefix(path)
}

func tsCodebaseID(path string) string {
	return "cb_ts_" + tshash.PathPrefix(path)
}

func parseTimestamp(value string) time.Time {
	if value == "" {
		return time.Time{}
	}
	parsedTime, err := time.Parse(time.RFC3339, value)
	if err == nil {
		return parsedTime
	}
	return time.Time{}
}

func defaultIfEmpty(value string, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
