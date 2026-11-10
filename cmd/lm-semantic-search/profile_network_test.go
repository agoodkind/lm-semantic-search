package main_test

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/networkcost"
)

const profileCLIPackage = "goodkind.io/lm-semantic-search/cmd/lm-semantic-search"

func buildProfileCLI(t *testing.T) string {
	t.Helper()
	binaryPath := filepath.Join(t.TempDir(), "lm-semantic-search")
	build := exec.Command("go", "build", "-o", binaryPath, profileCLIPackage)
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build CLI: %v\n%s", err, output)
	}
	return binaryPath
}

func isolatedProfileConfigPath(t *testing.T) string {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("CLAUDE_CONTEXTD_STATE_ROOT", t.TempDir())
	t.Setenv("CLAUDE_CONTEXT_PROFILE", "")
	t.Setenv("CLAUDE_CONTEXT_MODEL_DOWNLOAD_NETWORK_POLICY", "")
	t.Setenv("CLAUDE_CONTEXT_MODEL_DOWNLOAD_NETWORK_OVERRIDE", "")
	configRoot := t.TempDir()
	t.Setenv("CLAUDE_CONTEXTD_CONFIG_ROOT", configRoot)
	return filepath.Join(configRoot, "config.json")
}

func runProfileCLI(binaryPath string, arguments ...string) ([]byte, error) {
	command := exec.Command(binaryPath, append([]string{"profile"}, arguments...)...)
	command.Env = os.Environ()
	return command.CombinedOutput()
}

func TestProfilePersistsModelDownloadNetworkPolicyAndOverride(t *testing.T) {
	binaryPath := buildProfileCLI(t)
	configPath := isolatedProfileConfigPath(t)

	output, err := runProfileCLI(
		binaryPath,
		"offline",
		"--model-download-network-policy", "defer",
		"--model-download-network-override",
	)
	if err != nil {
		t.Fatalf("profile returned error: %v\n%s", err, output)
	}
	settings, err := config.ReadModelDownloadNetworkSettings(configPath)
	if err != nil {
		t.Fatalf("ReadModelDownloadNetworkSettings returned error: %v", err)
	}
	if settings.Policy != networkcost.PreferenceDefer {
		t.Errorf("persisted policy = %q, want %q", settings.Policy, networkcost.PreferenceDefer)
	}
	if !settings.Override {
		t.Error("persisted override = false, want true")
	}

	output, err = runProfileCLI(binaryPath, "offline", "--model-download-network-override=false")
	if err != nil {
		t.Fatalf("profile returned error: %v\n%s", err, output)
	}
	settings, err = config.ReadModelDownloadNetworkSettings(configPath)
	if err != nil {
		t.Fatalf("ReadModelDownloadNetworkSettings returned error: %v", err)
	}
	if settings.Override {
		t.Error("persisted override = true after --model-download-network-override=false")
	}
	if settings.Policy != networkcost.PreferenceDefer {
		t.Errorf("policy = %q after a command without the policy flag, want %q", settings.Policy, networkcost.PreferenceDefer)
	}
}

func TestProfileRejectsUnknownModelDownloadNetworkPolicy(t *testing.T) {
	binaryPath := buildProfileCLI(t)
	configPath := isolatedProfileConfigPath(t)

	output, err := runProfileCLI(binaryPath, "offline", "--model-download-network-policy", "block")
	var exitError *exec.ExitError
	if !errors.As(err, &exitError) {
		t.Fatalf("profile with the policy value block returned %v, want a failed exit\n%s", err, output)
	}
	if _, statErr := os.Stat(configPath); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("config file exists after the rejected policy: stat error = %v", statErr)
	}
}
