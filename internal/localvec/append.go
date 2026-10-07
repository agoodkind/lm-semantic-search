package localvec

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"goodkind.io/lm-semantic-search/internal/vectorindex"
)

// appendLocked writes the row file before replacing the index file.
// An interruption between those writes can produce mismatched rows and labels.
var errIndexRowsMismatch = errors.New("local vector index does not match the row file")

// Lock stored.mutex before calling appendLocked.
func (stored *collection) appendLocked(added []row) error {
	labels := make(map[uint64]string, len(stored.rows)+len(added))
	for _, existing := range stored.rows {
		labels[existing.Label] = existing.ID
	}
	appended := cloneRows(added)
	for index := range appended {
		appended[index].Label = labelForRowID(appended[index].ID, labels)
		labels[appended[index].Label] = appended[index].ID
	}
	for _, candidate := range appended {
		if err := stored.index.Add(candidate.Label, candidate.Vector); err != nil {
			stored.discardLoadedLocked()
			slog.Error("add local vector row to vector index failed", "collection", stored.name, "row_id", candidate.ID, "err", err)
			return fmt.Errorf("add local vector row %s to vector index: %w", candidate.ID, err)
		}
	}
	if err := appendRows(filepath.Join(stored.path, metadataFileName), appended); err != nil {
		stored.discardLoadedLocked()
		return err
	}
	if err := saveIndexFile(stored.path, stored); err != nil {
		stored.discardLoadedLocked()
		return err
	}
	for _, candidate := range appended {
		stored.reuseRows[candidate.ContentVectorKey] = len(stored.rows)
		stored.rows = append(stored.rows, candidate)
	}
	return nil
}

// After a failed append, the loaded index can include vectors that the
// collection files do not include. The next access reads the files again.
func (stored *collection) discardLoadedLocked() {
	if stored.index != nil {
		stored.index.Close()
		stored.index = nil
	}
	stored.rows = nil
	stored.reuseRows = nil
	stored.loaded = false
}

func (stored *collection) rebuildIndexLocked(rows []row, dimensions int) (*vectorindex.Index, int, error) {
	vectorIndex, err := buildVectorIndex(rows, dimensions)
	if err != nil {
		return nil, 0, err
	}
	stored.index = vectorIndex
	if err := saveIndexFile(stored.path, stored); err != nil {
		stored.index = nil
		vectorIndex.Close()
		return nil, 0, err
	}
	stored.index = nil
	return vectorIndex, dimensions, nil
}

func appendRows(path string, rows []row) error {
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		slog.Error("open local vector row file for append failed", "path", path, "err", err)
		return fmt.Errorf("open local vector row file %s for append: %w", path, err)
	}
	encoder := json.NewEncoder(file)
	for _, stored := range rows {
		if err := encoder.Encode(stored); err != nil {
			file.Close()
			slog.Error("append local vector row failed", "path", path, "err", err)
			return fmt.Errorf("append local vector row to %s: %w", path, err)
		}
	}
	if err := file.Sync(); err != nil {
		file.Close()
		slog.Error("sync local vector row file failed", "path", path, "err", err)
		return fmt.Errorf("sync local vector row file %s: %w", path, err)
	}
	if err := file.Close(); err != nil {
		slog.Error("close local vector row file failed", "path", path, "err", err)
		return fmt.Errorf("close local vector row file %s: %w", path, err)
	}
	return nil
}

func saveIndexFile(directory string, stored *collection) error {
	tempFile, err := os.CreateTemp(directory, "."+indexFileName+".tmp-*")
	if err != nil {
		slog.Error("create local vector index file failed", "collection", stored.name, "err", err)
		return fmt.Errorf("create local vector index file for %s: %w", stored.name, err)
	}
	tempPath := tempFile.Name()
	if err := tempFile.Close(); err != nil {
		_ = os.Remove(tempPath)
		slog.Error("close local vector index file failed", "path", tempPath, "err", err)
		return fmt.Errorf("close local vector index file %s: %w", tempPath, err)
	}
	if err := stored.index.Save(tempPath); err != nil {
		_ = os.Remove(tempPath)
		slog.Error("save local vector index file failed", "collection", stored.name, "err", err)
		return fmt.Errorf("save local vector index file for %s: %w", stored.name, err)
	}
	if err := os.Chmod(tempPath, 0o600); err != nil {
		_ = os.Remove(tempPath)
		slog.Error("set local vector index permissions failed", "path", tempPath, "err", err)
		return fmt.Errorf("set local vector index permissions: %w", err)
	}
	if err := os.Rename(tempPath, filepath.Join(directory, indexFileName)); err != nil {
		_ = os.Remove(tempPath)
		slog.Error("replace local vector index file failed", "collection", stored.name, "err", err)
		return fmt.Errorf("replace local vector index file for %s: %w", stored.name, err)
	}
	return nil
}
