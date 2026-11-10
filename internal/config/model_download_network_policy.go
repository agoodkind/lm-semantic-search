package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"

	"goodkind.io/lm-semantic-search/internal/networkcost"
)

const modelDownloadNetworkPolicyEnvVar = "CLAUDE_CONTEXT_MODEL_DOWNLOAD_NETWORK_POLICY"

const modelDownloadNetworkOverrideEnvVar = "CLAUDE_CONTEXT_MODEL_DOWNLOAD_NETWORK_OVERRIDE"

// ModelDownloadNetworkSettings includes an override that permits downloads
// on every network.
type ModelDownloadNetworkSettings struct {
	Policy   networkcost.Preference
	Override bool
}

// ReadModelDownloadNetworkSettings rereads the config file and environment on every call.
// A missing config file supplies no persisted values; environment variables and defaults determine the settings.
// The function returns an error for unknown policies, unreadable existing files, or invalid JSON.
func ReadModelDownloadNetworkSettings(configPath string) (ModelDownloadNetworkSettings, error) {
	fileConfig, err := readModelDownloadNetworkConfig(configPath)
	if err != nil {
		return ModelDownloadNetworkSettings{}, err
	}
	return resolveModelDownloadNetworkSettings(fileConfig)
}

// readModelDownloadNetworkConfig returns an empty config without an error
// only when the config file does not exist.
func readModelDownloadNetworkConfig(path string) (persistedConfig, error) {
	var cfg persistedConfig
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return cfg, nil
	}
	if err != nil {
		slog.Error("read model download network settings config failed", "path", path, "err", err)
		return cfg, fmt.Errorf("read model download network settings from %s: %w", path, err)
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		slog.Error("invalid JSON in model download network settings config", "path", path, "err", err)
		var emptyConfig persistedConfig
		return emptyConfig, fmt.Errorf("parse model download network settings JSON from %s: %w", path, err)
	}
	return cfg, nil
}

func resolveModelDownloadNetworkSettings(
	fileConfig persistedConfig,
) (ModelDownloadNetworkSettings, error) {
	policy, err := resolveModelDownloadNetworkPolicy(fileConfig.ModelDownloadNetworkPolicy)
	if err != nil {
		return ModelDownloadNetworkSettings{}, err
	}
	return ModelDownloadNetworkSettings{
		Policy:   policy,
		Override: resolveModelDownloadNetworkOverride(fileConfig.ModelDownloadNetworkOverride),
	}, nil
}

func resolveModelDownloadNetworkOverride(fileValue *bool) bool {
	return envBoolOrDefault(
		modelDownloadNetworkOverrideEnvVar,
		boolOrDefault(fileValue, false),
	)
}

func resolveModelDownloadNetworkPolicy(fileValue string) (networkcost.Preference, error) {
	configuredValue := envOrDefault(
		modelDownloadNetworkPolicyEnvVar,
		stringOrDefault(fileValue, string(networkcost.DefaultPreference)),
	)
	preference, err := networkcost.ParsePreference(configuredValue)
	if err != nil {
		slog.Error(
			"invalid model download network policy",
			"value", configuredValue,
			"config_field", modelDownloadNetworkPolicyJSONField,
			"env_var", modelDownloadNetworkPolicyEnvVar,
			"err", err,
		)
		return "", fmt.Errorf(
			"resolve model download network policy from %s or %s: %w",
			modelDownloadNetworkPolicyJSONField,
			modelDownloadNetworkPolicyEnvVar,
			err,
		)
	}
	return preference, nil
}
