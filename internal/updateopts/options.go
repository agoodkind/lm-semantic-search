// Package updateopts adapts lm-semantic-search release state to selfupdate.
package updateopts

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"goodkind.io/gklog/version"
	"goodkind.io/go-makefile/selfupdate"
	"goodkind.io/lm-semantic-search/internal/config"
)

const (
	repository   = "agoodkind/lm-semantic-search"
	daemonBinary = "lm-semantic-search-daemon"
	cliBinary    = "lm-semantic-search"
	mcpBinary    = "lm-semantic-search-mcp"

	updateStateFileName = "update-state.json"
	updateCacheDirName  = "update-cache"
	updateAPIBaseURLEnv = "LM_SEMANTIC_SEARCH_UPDATE_API_BASE_URL"
)

var (
	binaryApplyOrder = []string{cliBinary, mcpBinary, daemonBinary}
	applyBinary      = selfupdate.Apply
)

// Overrides carries operation-specific update settings.
type Overrides struct {
	Client     *http.Client
	InstallDir string
	StateRoot  string
	CacheDir   string
	DryRun     bool
	Log        *slog.Logger
}

// BinaryApplyResult records the result for one binary in an apply-all run.
type BinaryApplyResult struct {
	Binary string
	Result selfupdate.ApplyResult
}

// ApplyAllResult records a multi-binary self-update run.
type ApplyAllResult struct {
	Results         []BinaryApplyResult
	UpdateAvailable bool
	Applied         bool
	DryRun          bool
}

// Options builds selfupdate options for every release binary in apply order.
func Options(overrides Overrides) ([]selfupdate.Options, error) {
	installDir := strings.TrimSpace(overrides.InstallDir)
	if installDir == "" {
		executablePath, err := os.Executable()
		if err != nil {
			slog.Warn("resolve executable path failed", "err", err)
			return nil, fmt.Errorf("resolve executable path: %w", err)
		}
		installDir = filepath.Dir(executablePath)
	}
	return OptionsForInstallDir(installDir, overrides)
}

// OptionsForInstallDir builds options for every release binary under installDir.
func OptionsForInstallDir(installDir string, overrides Overrides) ([]selfupdate.Options, error) {
	installDir = strings.TrimSpace(installDir)
	if installDir == "" {
		return nil, fmt.Errorf("install dir is required")
	}
	stateRoot, err := resolveStateRoot(overrides)
	if err != nil {
		return nil, err
	}
	statePath := filepath.Join(stateRoot, updateStateFileName)
	cacheDir := overrides.CacheDir
	if strings.TrimSpace(cacheDir) == "" {
		cacheDir = filepath.Join(stateRoot, updateCacheDirName)
	}

	options := make([]selfupdate.Options, 0, len(binaryApplyOrder))
	for _, binary := range binaryApplyOrder {
		options = append(options, selfupdate.Options{
			Config:      configForBinary(binary),
			Client:      overrides.Client,
			InstallPath: filepath.Join(installDir, binary),
			CacheDir:    cacheDir,
			StatePath:   statePath,
			DryRun:      overrides.DryRun,
			Log:         overrides.Log,
		})
	}
	return options, nil
}

// CheckOptions returns the daemon-binary options used for check/status surfaces.
func CheckOptions(overrides Overrides) (selfupdate.Options, error) {
	options, err := Options(overrides)
	if err != nil {
		return selfupdate.Options{}, err
	}
	for _, option := range options {
		if option.Config.Binary == daemonBinary {
			return option, nil
		}
	}
	return selfupdate.Options{}, fmt.Errorf("daemon update options unavailable")
}

// StatePath returns the shared update state path.
func StatePath(overrides Overrides) (string, error) {
	stateRoot, err := resolveStateRoot(overrides)
	if err != nil {
		return "", err
	}
	return filepath.Join(stateRoot, updateStateFileName), nil
}

// ApplyAll applies the CLI and MCP binaries before applying the daemon binary.
func ApplyAll(ctx context.Context, overrides Overrides) (ApplyAllResult, error) {
	options, err := Options(overrides)
	if err != nil {
		return ApplyAllResult{}, err
	}
	result := ApplyAllResult{
		Results:         make([]BinaryApplyResult, 0, len(options)),
		UpdateAvailable: false,
		Applied:         false,
		DryRun:          overrides.DryRun,
	}
	for _, option := range options {
		applyResult, applyErr := applyBinary(ctx, option)
		result.Results = append(result.Results, BinaryApplyResult{
			Binary: option.Config.Binary,
			Result: applyResult,
		})
		result.UpdateAvailable = result.UpdateAvailable || applyResult.UpdateAvailable
		result.Applied = result.Applied || applyResult.Applied
		result.DryRun = result.DryRun || applyResult.DryRun
		if applyErr != nil {
			log := option.Log
			if log == nil {
				log = slog.Default()
			}
			log.WarnContext(ctx, "apply binary update failed", "binary", option.Config.Binary, "err", applyErr)
			return result, fmt.Errorf("apply %s: %w", option.Config.Binary, applyErr)
		}
	}
	return result, nil
}

func configForBinary(binary string) selfupdate.Config {
	return selfupdate.Config{
		Repo:             repository,
		Binary:           binary,
		CurrentVersion:   version.Version,
		CurrentCommit:    version.Commit,
		CurrentBuildHash: version.BuildHash(),
		CurrentDirty:     version.Dirty == "true",
		AllowPrerelease:  nil,
		Interval:         24 * time.Hour,
		APIBaseURLEnv:    updateAPIBaseURLEnv,
		ValidateArgs:     []string{"version"},
		ValidateMatch:    "version:",
	}
}

func resolveStateRoot(overrides Overrides) (string, error) {
	if strings.TrimSpace(overrides.StateRoot) != "" {
		return overrides.StateRoot, nil
	}
	cfg, err := config.Default()
	if err != nil {
		slog.Warn("load default config for update state failed", "err", err)
		return "", fmt.Errorf("load default config: %w", err)
	}
	return cfg.StateRoot, nil
}
