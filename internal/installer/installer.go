// Package installer installs the lm-semantic-search release binaries, the ONNX
// Runtime shared library, the lms alias, and the daemon user service.
package installer

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"goodkind.io/go-makefile/selfupdate"
	"goodkind.io/lm-semantic-search/internal/onnxruntimedist"
)

const (
	repository          = "agoodkind/lm-semantic-search"
	repositoryURL       = "https://github.com/" + repository
	daemonBinary        = "lm-semantic-search-daemon"
	cliBinary           = "lm-semantic-search"
	mcpBinary           = "lm-semantic-search-mcp"
	updateAPIBaseURLEnv = "LM_SEMANTIC_SEARCH_UPDATE_API_BASE_URL"

	// CLIAlias is the short name linked to the CLI binary in the bin dir.
	CLIAlias = "lms"

	launchdLabel       = "io.goodkind.lm-semantic-search-daemon"
	launchdLogName     = "lm-semantic-search-daemon.log"
	systemdUnit        = "lm-semantic-search-daemon.service"
	systemdDescription = "lm-semantic-search daemon"
	systemdRestart     = "always"
	systemdRestartSec  = "2"
)

// Options configures one install run.
type Options struct {
	BinDir         string
	Version        string
	InstallService bool
	Stdout         io.Writer
}

// Run installs the daemon, MCP, and CLI binaries from one release into
// BinDir, stages ONNX Runtime beside the daemon, links the lms alias, and
// optionally installs the daemon user service.
func Run(ctx context.Context, options Options) error {
	binDir := strings.TrimSpace(options.BinDir)
	if binDir == "" {
		return errors.New("install bin dir is required")
	}
	stdout := options.Stdout
	if stdout == nil {
		stdout = io.Discard
	}
	// Refuse before downloading anything, so an unrelated lms program leaves
	// the bin dir untouched instead of half installed.
	if err := checkAliasPath(binDir); err != nil {
		return err
	}

	daemonResult, err := installReleaseBinary(ctx, daemonBinary, options.Version, binDir)
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "installed: %s (%s)\n", daemonResult.InstallPath, daemonResult.Tag)
	for _, binary := range []string{mcpBinary, cliBinary} {
		result, installErr := installReleaseBinary(ctx, binary, daemonResult.Tag, binDir)
		if installErr != nil {
			return installErr
		}
		_, _ = fmt.Fprintf(stdout, "installed: %s (%s)\n", result.InstallPath, result.Tag)
	}

	if err := installONNXRuntime(ctx, binDir); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "installed: ONNX Runtime %s in %s\n", onnxruntimedist.Version, binDir)

	if err := LinkCLIAlias(binDir); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(stdout, "linked: %s -> %s\n", filepath.Join(binDir, CLIAlias), cliBinary)

	if !options.InstallService {
		_, _ = fmt.Fprintln(stdout, "service setup skipped")
		return nil
	}
	return installService(daemonResult.InstallPath, stdout)
}

// LinkCLIAlias points binDir/lms at the CLI binary with a relative symlink. It
// replaces an existing symlink and refuses to replace any other file.
func LinkCLIAlias(binDir string) error {
	if err := checkAliasPath(binDir); err != nil {
		return err
	}
	aliasPath := filepath.Join(binDir, CLIAlias)
	temporaryAliasPath := aliasPath + ".tmp"
	if err := os.Remove(temporaryAliasPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		slog.Error("remove stale alias temp failed", "path", temporaryAliasPath, "err", err)
		return fmt.Errorf("remove stale alias temp: %w", err)
	}
	if err := os.Symlink(cliBinary, temporaryAliasPath); err != nil {
		slog.Error("create alias symlink failed", "path", temporaryAliasPath, "err", err)
		return fmt.Errorf("create %s symlink: %w", CLIAlias, err)
	}
	if err := os.Rename(temporaryAliasPath, aliasPath); err != nil {
		_ = os.Remove(temporaryAliasPath)
		slog.Error("replace alias symlink failed", "path", aliasPath, "err", err)
		return fmt.Errorf("replace %s symlink: %w", CLIAlias, err)
	}
	return nil
}

func checkAliasPath(binDir string) error {
	aliasPath := filepath.Join(binDir, CLIAlias)
	info, err := os.Lstat(aliasPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		slog.Error("inspect alias path failed", "path", aliasPath, "err", err)
		return fmt.Errorf("inspect %s: %w", aliasPath, err)
	}
	if info.Mode()&os.ModeSymlink == 0 {
		return fmt.Errorf(
			"%s exists and is not a symlink; remove it to install the %s alias",
			aliasPath,
			CLIAlias,
		)
	}
	return nil
}

func installReleaseBinary(
	ctx context.Context,
	binary string,
	version string,
	binDir string,
) (selfupdate.InstallReleaseBinaryResult, error) {
	result, err := selfupdate.InstallReleaseBinary(ctx, selfupdate.InstallReleaseBinaryOptions{
		Options: selfupdate.Options{
			Config: selfupdate.Config{
				Repo:          repository,
				Binary:        binary,
				APIBaseURLEnv: updateAPIBaseURLEnv,
				AuthToken:     githubToken(),
			},
		},
		Version: version,
		Channel: selfupdate.ReleaseChannelRolling,
		BinDir:  binDir,
	})
	if err != nil {
		slog.ErrorContext(ctx, "install release binary failed", "binary", binary, "err", err)
		return selfupdate.InstallReleaseBinaryResult{}, fmt.Errorf("install %s: %w", binary, err)
	}
	return result, nil
}

func githubToken() string {
	token := strings.TrimSpace(os.Getenv("GITHUB_TOKEN"))
	if token != "" {
		return token
	}
	return strings.TrimSpace(os.Getenv("GH_TOKEN"))
}

func installONNXRuntime(ctx context.Context, binDir string) error {
	archive, err := onnxruntimedist.ArchiveFor(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		slog.ErrorContext(ctx, "resolve ONNX Runtime archive failed", "err", err)
		return fmt.Errorf("resolve ONNX Runtime archive: %w", err)
	}
	names, err := onnxruntimedist.LibraryNamesFor(runtime.GOOS)
	if err != nil {
		slog.ErrorContext(ctx, "resolve ONNX Runtime library names failed", "err", err)
		return fmt.Errorf("resolve ONNX Runtime library names: %w", err)
	}
	if err := onnxruntimedist.InstallSharedLibrary(
		ctx,
		http.DefaultClient,
		archive,
		names,
		binDir,
	); err != nil {
		slog.ErrorContext(ctx, "install ONNX Runtime failed", "err", err)
		return fmt.Errorf("install ONNX Runtime: %w", err)
	}
	return nil
}

func installService(daemonPath string, stdout io.Writer) error {
	home, err := os.UserHomeDir()
	if err != nil {
		slog.Error("resolve home dir failed", "err", err)
		return fmt.Errorf("resolve home dir: %w", err)
	}
	environment := []selfupdate.EnvironmentPair{{Name: "HOME", Value: home}}
	switch runtime.GOOS {
	case "darwin":
		err = selfupdate.InstallLaunchdService(selfupdate.LaunchdServiceOptions{
			Label:       launchdLabel,
			ProgramPath: daemonPath,
			PlistPath:   filepath.Join(home, "Library", "LaunchAgents", launchdLabel+".plist"),
			LogPath:     filepath.Join(home, "Library", "Logs", launchdLogName),
			Environment: environment,
			RunAtLoad:   true,
			KeepAlive:   true,
			Stdout:      stdout,
		})
	case "linux":
		err = selfupdate.InstallSystemdUserService(selfupdate.SystemdUserServiceOptions{
			Unit:          systemdUnit,
			ProgramPath:   daemonPath,
			UnitPath:      filepath.Join(home, ".config", "systemd", "user", systemdUnit),
			Description:   systemdDescription,
			Documentation: repositoryURL,
			Restart:       systemdRestart,
			RestartSec:    systemdRestartSec,
			Environment:   environment,
			Stdout:        stdout,
		})
	default:
		err = fmt.Errorf("unsupported OS %s", runtime.GOOS)
	}
	if err != nil {
		slog.Error("install daemon service failed", "err", err)
		return fmt.Errorf("install daemon service: %w", err)
	}
	return nil
}
