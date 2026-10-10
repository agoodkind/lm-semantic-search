package config_test

import (
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/networkcost"
)

const (
	networkPolicyEnvironmentVariable             = "CLAUDE_CONTEXT_MODEL_DOWNLOAD_NETWORK_POLICY"
	networkPolicyConfigFileMode      os.FileMode = 0o600
)

func isolatedConfigPath(t *testing.T, environmentPolicy string) string {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("CLAUDE_CONTEXTD_STATE_ROOT", t.TempDir())
	t.Setenv("CLAUDE_CONTEXT_PROFILE", "")
	t.Setenv("EMBEDDING_MODEL", "")
	t.Setenv(networkPolicyEnvironmentVariable, environmentPolicy)
	configRoot := t.TempDir()
	t.Setenv("CLAUDE_CONTEXTD_CONFIG_ROOT", configRoot)
	return filepath.Join(configRoot, "config.json")
}

func writeConfigFile(t *testing.T, configPath string, contents string) {
	t.Helper()
	err := os.WriteFile(configPath, []byte(contents), networkPolicyConfigFileMode)
	if err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
}

func TestDefaultResolvesModelDownloadNetworkPolicy(t *testing.T) {
	testCases := []struct {
		name        string
		environment string
		fileData    string
		want        networkcost.Preference
	}{
		{
			name:     "omitted keeps the default",
			fileData: `{}`,
			want:     networkcost.PreferenceWarn,
		},
		{
			name:     "config.json allow",
			fileData: `{"modelDownloadNetworkPolicy":"allow"}`,
			want:     networkcost.PreferenceAllow,
		},
		{
			name:     "config.json warn",
			fileData: `{"modelDownloadNetworkPolicy":"warn"}`,
			want:     networkcost.PreferenceWarn,
		},
		{
			name:     "config.json defer",
			fileData: `{"modelDownloadNetworkPolicy":"defer"}`,
			want:     networkcost.PreferenceDefer,
		},
		{
			name:        "environment allow",
			environment: "allow",
			fileData:    `{}`,
			want:        networkcost.PreferenceAllow,
		},
		{
			name:        "environment warn",
			environment: "warn",
			fileData:    `{"modelDownloadNetworkPolicy":"defer"}`,
			want:        networkcost.PreferenceWarn,
		},
		{
			name:        "environment defer",
			environment: "defer",
			fileData:    `{}`,
			want:        networkcost.PreferenceDefer,
		},
		{
			name:        "environment overrides config.json",
			environment: "defer",
			fileData:    `{"modelDownloadNetworkPolicy":"allow"}`,
			want:        networkcost.PreferenceDefer,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			configPath := isolatedConfigPath(t, testCase.environment)
			writeConfigFile(t, configPath, testCase.fileData)

			cfg, err := config.Default()
			if err != nil {
				t.Fatalf("Default returned error: %v", err)
			}
			if cfg.ModelDownloadNetworkPolicy != testCase.want {
				t.Errorf(
					"ModelDownloadNetworkPolicy = %q want %q",
					cfg.ModelDownloadNetworkPolicy,
					testCase.want,
				)
			}
		})
	}
}

func TestDefaultRejectsUnknownModelDownloadNetworkPolicy(t *testing.T) {
	testCases := []struct {
		name        string
		environment string
		fileData    string
	}{
		{name: "config.json value", fileData: `{"modelDownloadNetworkPolicy":"block"}`},
		{
			name:        "environment value",
			environment: "block",
			fileData:    `{"modelDownloadNetworkPolicy":"allow"}`,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			configPath := isolatedConfigPath(t, testCase.environment)
			writeConfigFile(t, configPath, testCase.fileData)

			cfg, err := config.Default()
			if err == nil {
				t.Fatalf(
					"Default returned no error, ModelDownloadNetworkPolicy = %q",
					cfg.ModelDownloadNetworkPolicy,
				)
			}
		})
	}
}

func TestSetModelDownloadNetworkPolicyPersistsValueDefaultReads(t *testing.T) {
	configPath := isolatedConfigPath(t, "")
	writeConfigFile(t, configPath, `{"futureField":{"enabled":true}}`)

	if err := config.SetModelDownloadNetworkPolicy(configPath, "defer"); err != nil {
		t.Fatalf("SetModelDownloadNetworkPolicy returned error: %v", err)
	}

	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("Default returned error: %v", err)
	}
	if cfg.ModelDownloadNetworkPolicy != networkcost.PreferenceDefer {
		t.Errorf(
			"ModelDownloadNetworkPolicy = %q want %q",
			cfg.ModelDownloadNetworkPolicy,
			networkcost.PreferenceDefer,
		)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	wantData := "{\n  \"futureField\": {\n    \"enabled\": true\n  },\n" +
		"  \"modelDownloadNetworkPolicy\": \"defer\"\n}\n"
	if string(data) != wantData {
		t.Errorf("config = %q want %q", data, wantData)
	}
}

func TestDefaultResolvesModelDownloadNetworkOverride(t *testing.T) {
	testCases := []struct {
		name        string
		environment string
		fileData    string
		want        bool
	}{
		{name: "omitted is off", fileData: `{}`, want: false},
		{name: "config.json on", fileData: `{"modelDownloadNetworkOverride":true}`, want: true},
		{name: "environment on", environment: "true", fileData: `{}`, want: true},
		{
			name:        "environment overrides config.json",
			environment: "false",
			fileData:    `{"modelDownloadNetworkOverride":true}`,
			want:        false,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			configPath := isolatedConfigPath(t, "")
			t.Setenv("CLAUDE_CONTEXT_MODEL_DOWNLOAD_NETWORK_OVERRIDE", testCase.environment)
			writeConfigFile(t, configPath, testCase.fileData)

			cfg, err := config.Default()
			if err != nil {
				t.Fatalf("Default returned error: %v", err)
			}
			if cfg.ModelDownloadNetworkOverride != testCase.want {
				t.Errorf(
					"ModelDownloadNetworkOverride = %t want %t",
					cfg.ModelDownloadNetworkOverride,
					testCase.want,
				)
			}
		})
	}
}

func TestSetModelDownloadNetworkOverridePersistsValueSettingsRead(t *testing.T) {
	configPath := isolatedConfigPath(t, "")
	t.Setenv("CLAUDE_CONTEXT_MODEL_DOWNLOAD_NETWORK_OVERRIDE", "")
	writeConfigFile(t, configPath, `{"modelDownloadNetworkPolicy":"defer"}`)

	if err := config.SetModelDownloadNetworkOverride(configPath, true); err != nil {
		t.Fatalf("SetModelDownloadNetworkOverride returned error: %v", err)
	}

	settings, err := config.ReadModelDownloadNetworkSettings(configPath)
	if err != nil {
		t.Fatalf("ReadModelDownloadNetworkSettings returned error: %v", err)
	}
	if !settings.Override || settings.Policy != networkcost.PreferenceDefer {
		t.Errorf("settings = %+v want override with policy %q", settings, networkcost.PreferenceDefer)
	}
}

func TestReadModelDownloadNetworkSettingsDefaultsWithoutConfigFile(t *testing.T) {
	configPath := isolatedConfigPath(t, "")
	t.Setenv("CLAUDE_CONTEXT_MODEL_DOWNLOAD_NETWORK_OVERRIDE", "")

	settings, err := config.ReadModelDownloadNetworkSettings(configPath)
	if err != nil {
		t.Fatalf("ReadModelDownloadNetworkSettings returned error: %v", err)
	}
	if settings.Override || settings.Policy != networkcost.DefaultPreference {
		t.Errorf("ReadModelDownloadNetworkSettings returned %+v instead of override off with the default policy %q when no config file exists.", settings, networkcost.DefaultPreference)
	}
}

func TestReadModelDownloadNetworkSettingsRejectsInvalidJSON(t *testing.T) {
	configPath := isolatedConfigPath(t, "")
	t.Setenv("CLAUDE_CONTEXT_MODEL_DOWNLOAD_NETWORK_OVERRIDE", "")
	writeConfigFile(t, configPath, `{"modelDownloadNetworkPolicy":`)

	settings, err := config.ReadModelDownloadNetworkSettings(configPath)
	if err == nil {
		t.Fatalf("ReadModelDownloadNetworkSettings returned %+v without an error for a config file with invalid JSON.", settings)
	}
}

func TestSetModelDownloadNetworkPolicyRejectsUnknownWithoutWriting(t *testing.T) {
	configPath := isolatedConfigPath(t, "")
	initialData := "{\"modelDownloadNetworkPolicy\":\"allow\"}\n"
	writeConfigFile(t, configPath, initialData)

	if err := config.SetModelDownloadNetworkPolicy(configPath, "block"); err == nil {
		t.Fatal("SetModelDownloadNetworkPolicy returned no error")
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	if string(data) != initialData {
		t.Fatalf("config changed after invalid policy: %q", data)
	}
}
