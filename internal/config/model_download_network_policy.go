package config

import (
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/networkcost"
)

const modelDownloadNetworkPolicyEnvVar = "CLAUDE_CONTEXT_MODEL_DOWNLOAD_NETWORK_POLICY"

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
