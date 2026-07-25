package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

const (
	profileJSONField                           = "profile"
	offlineEmbeddingModelJSONField             = "offlineEmbeddingModel"
	persistedConfigDirectoryMode   os.FileMode = 0o755
	persistedConfigFileMode        os.FileMode = 0o600
)

// SetProfile atomically updates the profile in the daemon JSON config while
// preserving every other known or future field.
func SetProfile(path string, profile string) error {
	if profile != ProfileStandard && profile != ProfileOffline {
		return fmt.Errorf(
			"profile must be %q or %q",
			ProfileStandard,
			ProfileOffline,
		)
	}

	document := make(map[string]json.RawMessage)
	data, err := os.ReadFile(path)
	if err == nil {
		if err := json.Unmarshal(data, &document); err != nil {
			slog.Error("unmarshal daemon config failed", "path", path, "err", err)
			return fmt.Errorf("unmarshal daemon config %s: %w", path, err)
		}
		if document == nil {
			document = make(map[string]json.RawMessage)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		slog.Error("read daemon config failed", "path", path, "err", err)
		return fmt.Errorf("read daemon config %s: %w", path, err)
	}

	profileData, err := json.Marshal(profile)
	if err != nil {
		slog.Error("marshal daemon profile failed", "profile", profile, "err", err)
		return fmt.Errorf("marshal profile %q: %w", profile, err)
	}
	document[profileJSONField] = profileData

	output, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		slog.Error("marshal daemon config failed", "path", path, "err", err)
		return fmt.Errorf("marshal daemon config %s: %w", path, err)
	}
	output = append(output, '\n')
	if err := writePersistedConfig(path, output); err != nil {
		return err
	}
	slog.Info("set daemon profile", "path", path, "profile", profile)
	return nil
}

// SetOfflineModel atomically updates the offline embedding model in the daemon
// JSON config while preserving every other known or future field.
func SetOfflineModel(path string, model string) error {
	if model == "" {
		return nil
	}
	preset, err := offlinemodel.Resolve(model)
	if err != nil {
		return fmt.Errorf("resolve offline embedding model: %w", err)
	}

	document := make(map[string]json.RawMessage)
	data, err := os.ReadFile(path)
	if err == nil {
		if err := json.Unmarshal(data, &document); err != nil {
			slog.Error("unmarshal daemon config failed", "path", path, "err", err)
			return fmt.Errorf("unmarshal daemon config %s: %w", path, err)
		}
		if document == nil {
			document = make(map[string]json.RawMessage)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		slog.Error("read daemon config failed", "path", path, "err", err)
		return fmt.Errorf("read daemon config %s: %w", path, err)
	}

	modelData, err := json.Marshal(preset.Name)
	if err != nil {
		slog.Error("marshal offline embedding model failed", "model", preset.Name, "err", err)
		return fmt.Errorf("marshal offline embedding model %q: %w", preset.Name, err)
	}
	document[offlineEmbeddingModelJSONField] = modelData

	output, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		slog.Error("marshal daemon config failed", "path", path, "err", err)
		return fmt.Errorf("marshal daemon config %s: %w", path, err)
	}
	output = append(output, '\n')
	if err := writePersistedConfig(path, output); err != nil {
		return err
	}
	slog.Info("set offline embedding model", "path", path, "model", preset.Name)
	return nil
}

func writePersistedConfig(path string, data []byte) error {
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, persistedConfigDirectoryMode); err != nil {
		slog.Error("create daemon config directory failed", "path", directory, "err", err)
		return fmt.Errorf("create daemon config directory %s: %w", directory, err)
	}

	tempFile, err := os.CreateTemp(directory, ".config-*.tmp")
	if err != nil {
		slog.Error("create temporary daemon config failed", "path", directory, "err", err)
		return fmt.Errorf("create temporary daemon config in %s: %w", directory, err)
	}
	tempPath := tempFile.Name()
	removeTemp := true
	defer func() {
		if removeTemp {
			_ = os.Remove(tempPath)
		}
	}()

	if err := tempFile.Chmod(persistedConfigFileMode); err != nil {
		_ = tempFile.Close()
		slog.Error("set temporary daemon config permissions failed", "path", tempPath, "err", err)
		return fmt.Errorf("set temporary daemon config permissions %s: %w", tempPath, err)
	}
	if _, err := tempFile.Write(data); err != nil {
		_ = tempFile.Close()
		slog.Error("write temporary daemon config failed", "path", tempPath, "err", err)
		return fmt.Errorf("write temporary daemon config %s: %w", tempPath, err)
	}
	if err := tempFile.Sync(); err != nil {
		_ = tempFile.Close()
		slog.Error("sync temporary daemon config failed", "path", tempPath, "err", err)
		return fmt.Errorf("sync temporary daemon config %s: %w", tempPath, err)
	}
	if err := tempFile.Close(); err != nil {
		slog.Error("close temporary daemon config failed", "path", tempPath, "err", err)
		return fmt.Errorf("close temporary daemon config %s: %w", tempPath, err)
	}
	if err := os.Rename(tempPath, path); err != nil {
		slog.Error("replace daemon config failed", "path", path, "err", err)
		return fmt.Errorf("replace daemon config %s: %w", path, err)
	}
	removeTemp = false
	if err := syncPersistedConfigDirectory(directory); err != nil {
		return err
	}
	slog.Debug("wrote daemon config", "path", path, "bytes", len(data))
	return nil
}

func syncPersistedConfigDirectory(directory string) error {
	directoryFile, err := os.Open(directory)
	if err != nil {
		slog.Error("open daemon config directory failed", "path", directory, "err", err)
		return fmt.Errorf("open daemon config directory %s: %w", directory, err)
	}
	syncErr := directoryFile.Sync()
	closeErr := directoryFile.Close()
	if syncErr != nil {
		slog.Error("sync daemon config directory failed", "path", directory, "err", syncErr)
		return fmt.Errorf("sync daemon config directory %s: %w", directory, syncErr)
	}
	if closeErr != nil {
		slog.Error("close daemon config directory failed", "path", directory, "err", closeErr)
		return fmt.Errorf("close daemon config directory %s: %w", directory, closeErr)
	}
	return nil
}
