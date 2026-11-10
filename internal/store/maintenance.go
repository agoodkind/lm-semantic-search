package store

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"goodkind.io/lm-semantic-search/internal/model"
)

const maintenanceFilename = "maintenance.json"

// MaintenancePath returns the maintenance-mode file beside the registry.
func MaintenancePath(registryPath string) string {
	return filepath.Join(filepath.Dir(registryPath), maintenanceFilename)
}

// ReadMaintenance reads the persisted maintenance mode. A missing file is the
// mode never having been set, which reads as off.
func ReadMaintenance(path string) (model.MaintenanceState, error) {
	var off model.MaintenanceState
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return off, nil
	}
	if err != nil {
		slog.Error("read maintenance file failed", "path", path, "err", err)
		return off, fmt.Errorf("read maintenance file %s: %w", path, err)
	}
	var state model.MaintenanceState
	if err := json.Unmarshal(data, &state); err != nil {
		slog.Error("unmarshal maintenance file failed", "path", path, "err", err)
		return off, fmt.Errorf("unmarshal maintenance file %s: %w", path, err)
	}
	return state, nil
}

// WriteMaintenance atomically replaces the persisted maintenance mode.
func WriteMaintenance(path string, state model.MaintenanceState) error {
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		slog.Error("marshal maintenance file failed", "path", path, "err", err)
		return fmt.Errorf("marshal maintenance file %s: %w", path, err)
	}
	return replaceFileAtomically(path, "maintenance file", func(file *os.File) error {
		if _, writeErr := file.Write(data); writeErr != nil {
			return fmt.Errorf("write maintenance bytes: %w", writeErr)
		}
		return nil
	})
}
