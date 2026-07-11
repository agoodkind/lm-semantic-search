package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"os"
	"path/filepath"

	"goodkind.io/lm-semantic-search/internal/store"
)

type conversationDerivedMarkerFile struct {
	Versions map[string]string `json:"versions"`
}

func conversationDerivedMarkerPath(snapshotPath string) string {
	return snapshotPath + ".derived"
}

// loadConversationDerivedMarkers treats missing, unreadable, and legacy files
// as empty so every delivered conversation is safely reexamined once.
func loadConversationDerivedMarkers(path string) map[string]string {
	data, err := os.ReadFile(path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			slog.Warn("load conversation derived markers failed; treating as empty", "path", path, "err", err)
		}
		return map[string]string{}
	}

	var markerFile conversationDerivedMarkerFile
	if err := json.Unmarshal(data, &markerFile); err != nil {
		slog.Warn("unmarshal conversation derived markers failed; treating as empty", "path", path, "err", err)
		return map[string]string{}
	}
	if markerFile.Versions == nil {
		return map[string]string{}
	}
	return markerFile.Versions
}

// writeConversationDerivedMarkers persists the marker map atomically, mirroring
// merkle.WriteSnapshot: it logs each failure before returning the wrapped error
// so the sibling atomic writers share one diagnostic contract.
func writeConversationDerivedMarkers(path string, versions map[string]string) error {
	markerFile := conversationDerivedMarkerFile{Versions: maps.Clone(versions)}
	data, err := json.MarshalIndent(markerFile, "", "  ")
	if err != nil {
		slog.Error("marshal conversation derived markers failed", "path", path, "err", err)
		return fmt.Errorf("marshal conversation derived markers %s: %w", path, err)
	}
	if err := store.EnsureDir(filepath.Dir(path)); err != nil {
		slog.Error("ensure conversation derived marker directory failed", "path", path, "err", err)
		return fmt.Errorf("ensure conversation derived marker directory for %s: %w", path, err)
	}

	tempFile, err := os.CreateTemp(filepath.Dir(path), ".tmp-*")
	if err != nil {
		slog.Error("create temp conversation derived marker file failed", "path", path, "err", err)
		return fmt.Errorf("create temp conversation derived marker file for %s: %w", path, err)
	}
	tempPath := tempFile.Name()
	if _, err := tempFile.Write(data); err != nil {
		_ = tempFile.Close()
		_ = os.Remove(tempPath)
		slog.Error("write temp conversation derived marker file failed", "path", tempPath, "err", err)
		return fmt.Errorf("write temp conversation derived marker file %s: %w", tempPath, err)
	}
	if err := tempFile.Close(); err != nil {
		_ = os.Remove(tempPath)
		slog.Error("close temp conversation derived marker file failed", "path", tempPath, "err", err)
		return fmt.Errorf("close temp conversation derived marker file %s: %w", tempPath, err)
	}
	if err := os.Rename(tempPath, path); err != nil {
		_ = os.Remove(tempPath)
		slog.Error("rename temp conversation derived marker file failed", "from", tempPath, "to", path, "err", err)
		return fmt.Errorf("rename temp conversation derived marker file %s to %s: %w", tempPath, path, err)
	}
	return nil
}

// removeConversationDerivedMarkers drops the derived-marker sidecar beside a
// snapshot and logs a best-effort warning on failure. It keeps the marker
// sidecar's lifecycle in step with the Merkle checkpoint it accompanies when a
// staging or bootstrap checkpoint is discarded.
func removeConversationDerivedMarkers(ctx context.Context, snapshotPath string, message string) {
	markerPath := conversationDerivedMarkerPath(snapshotPath)
	if removeErr := store.RemoveFile(markerPath); removeErr != nil {
		slog.WarnContext(ctx, message, "path", markerPath, "err", removeErr)
	}
}

// bootstrapDerivedVersions returns the derived markers a resuming bootstrap
// should carry: the persisted staging markers when a checkpoint seeds the
// resume, or an empty map for a from-scratch build that re-embeds every item.
func bootstrapDerivedVersions(snapshotPath string, seededFiles int) map[string]string {
	if seededFiles == 0 {
		return map[string]string{}
	}
	return loadConversationDerivedMarkers(conversationDerivedMarkerPath(snapshotPath))
}

// promoteConversationDerivedMarkers moves the staging derived-marker sidecar
// onto the live path alongside the promoted Merkle snapshot. It runs only for a
// conversation source, and a missing staging sidecar clears any stale live
// sidecar so a promoted empty build keeps no markers from an earlier run.
func promoteConversationDerivedMarkers(ctx context.Context, source itemSource, snapshotPath string, livePath string) {
	if _, ok := source.(conversationItemSource); !ok {
		return
	}
	stagingMarkerPath := conversationDerivedMarkerPath(snapshotPath)
	liveMarkerPath := conversationDerivedMarkerPath(livePath)
	if err := os.Rename(stagingMarkerPath, liveMarkerPath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			if removeErr := store.RemoveFile(liveMarkerPath); removeErr != nil {
				slog.WarnContext(ctx, "remove stale live derived markers failed", "path", liveMarkerPath, "err", removeErr)
			}
		} else {
			slog.WarnContext(ctx, "promote staging derived markers failed", "from", stagingMarkerPath, "to", liveMarkerPath, "err", err)
		}
	}
}
