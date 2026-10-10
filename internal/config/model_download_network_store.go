package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
)

const modelDownloadNetworkOverrideJSONField = "modelDownloadNetworkOverride"

// SetModelDownloadNetworkOverride changes only the override in the daemon's
// JSON config file.
func SetModelDownloadNetworkOverride(path string, override bool) error {
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

	overrideData, err := json.Marshal(override)
	if err != nil {
		slog.Error("config.model_download_network_override.marshal_failed", "override", override, "err", err)
		return fmt.Errorf("marshal model download network override %t: %w", override, err)
	}
	document[modelDownloadNetworkOverrideJSONField] = overrideData

	output, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		slog.Error("marshal daemon config failed", "path", path, "err", err)
		return fmt.Errorf("marshal daemon config %s: %w", path, err)
	}
	output = append(output, '\n')
	if err := writePersistedConfig(path, output); err != nil {
		return err
	}
	slog.Info("config.model_download_network_override.set", "path", path, "override", override)
	return nil
}
