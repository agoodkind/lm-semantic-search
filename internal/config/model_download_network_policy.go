package config

import (
	"fmt"
	"log/slog"

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

// ReadModelDownloadNetworkSettings rereads the config file and environment
// on every call. The function rejects unknown policies.
func ReadModelDownloadNetworkSettings(configPath string) (ModelDownloadNetworkSettings, error) {
	return resolveModelDownloadNetworkSettings(readPersistedConfig(configPath))
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
