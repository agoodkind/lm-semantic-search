package daemon

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"goodkind.io/lm-semantic-search/internal/cbm"
	"goodkind.io/lm-semantic-search/internal/store"
)

const (
	graphDatabaseSuffix       = ".db"
	graphDatabaseBackupSuffix = ".corrupt"
	graphDatabaseWALSuffix    = "-wal"
	graphDecisionsSuffix      = ".adr"
	graphDecisionsTempSuffix  = ".tmp"
	retainedDecisionsDir      = "retained-decisions"
	retainedDecisionsCopyType = "retained graph decisions"
	graphDecisionsCopyType    = "graph decisions"
)

// graphDecisionsPath returns the path of the engine's decision file for a graph.
func (manager *Manager) graphDecisionsPath(codebaseID string) string {
	return filepath.Join(manager.config.GraphDir, codebaseID+graphDatabaseSuffix+graphDecisionsSuffix)
}

// retainedDecisionsPath returns a decision file path based on the codebase's
// canonical path. The daemon assigns a new codebase ID when it indexes that
// path again.
func (manager *Manager) retainedDecisionsPath(canonicalPath string) string {
	digest := sha256.Sum256([]byte(canonicalPath))
	return filepath.Join(manager.config.GraphDir, retainedDecisionsDir, hex.EncodeToString(digest[:])+graphDecisionsSuffix)
}

// readGraphDecisions reads the decision record from the graph database or, if
// the database does not provide a readable record, the engine's decision file.
// If the engine reports ErrDecisionsUnreadable and the file does not contain a
// record, it links any remaining database to a backup name and renames any WAL
// file before clearGraphCache removes the original database path.
func (manager *Manager) readGraphDecisions(ctx context.Context, codebaseID string, cachedEngine *cbm.Engine) (string, error) {
	content, err := manager.readDatabaseDecisions(ctx, codebaseID, cachedEngine)
	databaseUnreadable := errors.Is(err, cbm.ErrDecisionsUnreadable)
	if err != nil && !databaseUnreadable {
		return "", err
	}
	if content != "" {
		return content, nil
	}

	decisionsPath := manager.graphDecisionsPath(codebaseID)
	data, readErr := os.ReadFile(decisionsPath)
	if readErr != nil && !errors.Is(readErr, os.ErrNotExist) {
		slog.ErrorContext(ctx, "The daemon failed to read the graph decisions file", "codebase_id", codebaseID, "path", decisionsPath, "err", readErr)
		return "", fmt.Errorf("the daemon could not read graph decisions file %s: %w", decisionsPath, readErr)
	}
	if len(data) == 0 && databaseUnreadable {
		if backupErr := manager.backupUnreadableDatabase(ctx, codebaseID); backupErr != nil {
			return "", backupErr
		}
	}
	return string(data), nil
}

// closeCachedGraphEngine closes the cached engine and removes it from
// graphEngines before the daemon backs up the database and its WAL file.
// SQLite can write to the WAL file while a connection remains open.
func (manager *Manager) closeCachedGraphEngine(codebaseID string) {
	manager.graphMutex.Lock()
	defer manager.graphMutex.Unlock()
	if engine, found := manager.graphEngines[codebaseID]; found {
		engine.Close()
		delete(manager.graphEngines, codebaseID)
	}
}

// backupUnreadableDatabase closes the cached engine, links the graph database
// to an unused backup name, and renames any WAL file to match. It tries
// <database path>.corrupt first, then .corrupt.1, .corrupt.2, and later numbers.
// [os.Link] does not replace an existing database backup. If the database path
// does not exist, backupUnreadableDatabase returns without creating a backup.
func (manager *Manager) backupUnreadableDatabase(ctx context.Context, codebaseID string) error {
	manager.closeCachedGraphEngine(codebaseID)
	databasePath := filepath.Join(manager.config.GraphDir, codebaseID+graphDatabaseSuffix)
	backupPath := databasePath + graphDatabaseBackupSuffix
	for attempt := 1; ; attempt++ {
		linkErr := os.Link(databasePath, backupPath)
		if linkErr == nil {
			break
		}
		if errors.Is(linkErr, os.ErrNotExist) {
			return nil
		}
		if !errors.Is(linkErr, os.ErrExist) {
			slog.ErrorContext(ctx, "The daemon failed to link the unreadable graph database to its backup name", "codebase_id", codebaseID, "path", backupPath, "err", linkErr)
			return fmt.Errorf("the daemon could not link the unreadable graph database to %s: %w", backupPath, linkErr)
		}
		backupPath = fmt.Sprintf("%s%s.%d", databasePath, graphDatabaseBackupSuffix, attempt)
	}
	walErr := os.Rename(databasePath+graphDatabaseWALSuffix, backupPath+graphDatabaseWALSuffix)
	if walErr != nil && !errors.Is(walErr, os.ErrNotExist) {
		slog.ErrorContext(ctx, "The daemon failed to rename the unreadable graph database WAL file", "codebase_id", codebaseID, "path", backupPath, "err", walErr)
		if removeErr := os.Remove(backupPath); removeErr != nil {
			slog.ErrorContext(ctx, "The daemon failed to remove the graph database backup name", "codebase_id", codebaseID, "path", backupPath, "err", removeErr)
			return fmt.Errorf("the daemon could not rename the unreadable graph database WAL file for %s: %w", backupPath, errors.Join(walErr, removeErr))
		}
		return fmt.Errorf("the daemon could not rename the unreadable graph database WAL file for %s: %w", backupPath, walErr)
	}
	slog.WarnContext(ctx, "The daemon linked the unreadable graph database to a backup name", "codebase_id", codebaseID, "path", backupPath)
	return nil
}

// readDatabaseDecisions reads the decision row when the graph database exists.
func (manager *Manager) readDatabaseDecisions(ctx context.Context, codebaseID string, cachedEngine *cbm.Engine) (string, error) {
	databasePath := filepath.Join(manager.config.GraphDir, codebaseID+graphDatabaseSuffix)
	if _, statErr := os.Stat(databasePath); statErr != nil {
		if errors.Is(statErr, os.ErrNotExist) {
			return "", nil
		}
		slog.ErrorContext(ctx, "The daemon failed to inspect the graph database", "codebase_id", codebaseID, "path", databasePath, "err", statErr)
		return "", fmt.Errorf("the daemon could not inspect graph database %s: %w", databasePath, statErr)
	}
	engine := cachedEngine
	if engine == nil {
		openedEngine, openErr := cbm.Open(codebaseID, manager.config.GraphDir)
		if openErr != nil {
			slog.ErrorContext(ctx, "The daemon failed to open the graph engine", "codebase_id", codebaseID, "err", openErr)
			return "", fmt.Errorf("the daemon could not open the graph engine: %w", openErr)
		}
		defer openedEngine.Close()
		engine = openedEngine
	}
	content, readErr := engine.ReadDecisions()
	if readErr != nil {
		slog.ErrorContext(ctx, "The daemon failed to read decisions from the graph database", "codebase_id", codebaseID, "err", readErr)
		return "", fmt.Errorf("the daemon could not read the decision record from the graph database: %w", readErr)
	}
	return content, nil
}

// retainGraphDecisions copies a graph's decision record outside the graph files.
// If the file for the canonical path contains a different record, it copies
// that older record to a second file and writes the new record to the file
// that the next index restores.
func (manager *Manager) retainGraphDecisions(ctx context.Context, codebaseID string, canonicalPath string, cachedEngine *cbm.Engine) error {
	content, err := manager.readGraphDecisions(ctx, codebaseID, cachedEngine)
	if err != nil {
		return err
	}
	if content == "" {
		return nil
	}

	data := []byte(content)
	retainedPath := manager.retainedDecisionsPath(canonicalPath)
	existing, readErr := os.ReadFile(retainedPath)
	if readErr != nil && !errors.Is(readErr, os.ErrNotExist) {
		return fmt.Errorf("the daemon could not read retained graph decisions file %s: %w", retainedPath, readErr)
	}
	if readErr == nil {
		if bytes.Equal(existing, data) {
			return nil
		}
		olderDigest := sha256.Sum256(existing)
		olderPath := retainedPath + "." + hex.EncodeToString(olderDigest[:])
		slog.WarnContext(ctx, "The retained graph decisions differ; the daemon will copy the older record to a second file", "codebase_id", codebaseID, "path", olderPath)
		if err = store.WriteFileAtomically(olderPath, existing, retainedDecisionsCopyType); err != nil {
			return fmt.Errorf("the daemon could not copy the older retained graph decision record: %w", err)
		}
	}
	if err = store.WriteFileAtomically(retainedPath, data, retainedDecisionsCopyType); err != nil {
		return fmt.Errorf("the daemon could not write the retained graph decision record: %w", err)
	}
	return nil
}

// restoreRetainedDecisions copies the retained record to the engine's decision
// file if that file does not exist. It removes the retained file after a
// successful copy or if both files match. It keeps both files if they differ.
func (manager *Manager) restoreRetainedDecisions(ctx context.Context, codebaseID string, canonicalPath string) error {
	retainedPath := manager.retainedDecisionsPath(canonicalPath)
	retained, err := os.ReadFile(retainedPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("the daemon could not read retained graph decisions file %s: %w", retainedPath, err)
	}

	decisionsPath := manager.graphDecisionsPath(codebaseID)
	current, err := os.ReadFile(decisionsPath)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("the daemon could not read graph decisions file %s: %w", decisionsPath, err)
	}
	if err == nil && !bytes.Equal(current, retained) {
		slog.WarnContext(ctx, "The graph decisions differ from the retained decisions; the daemon will keep both files", "codebase_id", codebaseID, "path", retainedPath)
		return nil
	}
	if err != nil {
		if writeErr := store.WriteFileAtomically(decisionsPath, retained, graphDecisionsCopyType); writeErr != nil {
			return fmt.Errorf("the daemon could not write the graph decisions file: %w", writeErr)
		}
	}
	if removeErr := store.RemoveFile(retainedPath); removeErr != nil {
		return fmt.Errorf("the daemon could not remove the retained graph decisions file: %w", removeErr)
	}
	return nil
}
