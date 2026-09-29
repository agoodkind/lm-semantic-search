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
	"goodkind.io/lm-semantic-search/internal/updateopts"
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

type operatingSystem string

const (
	operatingSystemDarwin operatingSystem = "darwin"
	operatingSystemLinux  operatingSystem = "linux"
)

// Options configures one install run.
type Options struct {
	BinDir         string
	Version        string
	InstallService bool
	Stdout         io.Writer
}

// Run installs the daemon, MCP, and CLI binaries from one release into BinDir
// as one set, links the lms alias, and optionally installs the daemon user
// service.
//
// Run writes the pinned versioned ONNX Runtime library file into BinDir first.
// An installed daemon loads the library named by the SONAME symlink, and a new
// versioned file leaves that symlink unchanged. Each candidate validates with
// a library search path that finds the new library file. The SONAME and
// unversioned symlinks change in the same commit as the binaries. A failed
// download, verification, or validation leaves every binary and symlink
// unchanged.
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

	results, err := installReleaseSet(ctx, options.Version, binDir)
	if err != nil {
		return err
	}
	daemonPath := filepath.Join(binDir, daemonBinary)
	for _, result := range results {
		_, _ = fmt.Fprintf(stdout, "installed: %s (%s)\n", result.InstallPath, result.Tag)
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
	return installService(daemonPath, stdout)
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

// installReleaseSet writes the versioned ONNX Runtime library into binDir and
// installs the CLI, MCP, and daemon binaries of one release with the SONAME
// and unversioned library symlinks in one selfupdate.InstallReleaseBinaries
// commit.
func installReleaseSet(
	ctx context.Context,
	version string,
	binDir string,
) ([]selfupdate.InstallReleaseBinaryResult, error) {
	archive, err := onnxruntimedist.ArchiveFor(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		slog.ErrorContext(ctx, "resolve ONNX Runtime archive failed", "err", err)
		return nil, fmt.Errorf("resolve ONNX Runtime archive: %w", err)
	}
	names, err := onnxruntimedist.LibraryNamesFor(runtime.GOOS)
	if err != nil {
		slog.ErrorContext(ctx, "resolve ONNX Runtime library names failed", "err", err)
		return nil, fmt.Errorf("resolve ONNX Runtime library names: %w", err)
	}
	if err := onnxruntimedist.StageSharedLibrary(ctx, http.DefaultClient, archive, names, binDir); err != nil {
		slog.ErrorContext(ctx, "stage ONNX Runtime failed", "err", err)
		return nil, fmt.Errorf("stage ONNX Runtime: %w", err)
	}
	validationDir, err := os.MkdirTemp("", "lms-install-validation.")
	if err != nil {
		slog.ErrorContext(ctx, "create validation library dir failed", "err", err)
		return nil, fmt.Errorf("create validation library dir: %w", err)
	}
	defer func() { _ = os.RemoveAll(validationDir) }()
	stagedLibraryPath, err := filepath.Abs(filepath.Join(binDir, names.Versioned))
	if err != nil {
		slog.ErrorContext(ctx, "resolve staged library path failed", "err", err)
		return nil, fmt.Errorf("resolve staged library path: %w", err)
	}
	if err := os.Symlink(stagedLibraryPath, filepath.Join(validationDir, names.SONAME)); err != nil {
		slog.ErrorContext(ctx, "create validation library link failed", "err", err)
		return nil, fmt.Errorf("create validation library link: %w", err)
	}

	options, err := updateopts.NetworkOptionsForInstallDir(ctx, binDir, updateopts.Overrides{
		Client:     nil,
		InstallDir: binDir,
		StateRoot:  "",
		CacheDir:   "",
		DryRun:     false,
		Log:        nil,
	})
	if err != nil {
		slog.ErrorContext(ctx, "build install options failed", "err", err)
		return nil, fmt.Errorf("build install options: %w", err)
	}
	validateEnv := []string{libraryPathEnv() + "=" + validationDir}
	for index := range options {
		options[index].Config.ValidateEnv = validateEnv
	}
	results, err := selfupdate.InstallReleaseBinaries(ctx, selfupdate.InstallReleaseBinariesOptions{
		Options: options,
		Version: version,
		Channel: selfupdate.ReleaseChannelRolling,
		BinDir:  binDir,
		Symlinks: []selfupdate.InstallSymlink{
			{Name: names.SONAME, Target: names.Versioned},
			{Name: names.Unversioned, Target: names.Versioned},
		},
	})
	if err != nil {
		slog.ErrorContext(ctx, "install release binaries failed", "err", err)
		return nil, fmt.Errorf("install release binaries: %w", err)
	}
	return results, nil
}

// libraryPathEnv returns the dynamic loader variable that the loader searches
// before the runpath beside the binary: DYLD_LIBRARY_PATH on darwin and
// LD_LIBRARY_PATH on linux, where the release binaries carry DT_RUNPATH.
func libraryPathEnv() string {
	if operatingSystem(runtime.GOOS) == operatingSystemDarwin {
		return "DYLD_LIBRARY_PATH"
	}
	return "LD_LIBRARY_PATH"
}

func installService(daemonPath string, stdout io.Writer) error {
	home, err := os.UserHomeDir()
	if err != nil {
		slog.Error("resolve home dir failed", "err", err)
		return fmt.Errorf("resolve home dir: %w", err)
	}
	environment := []selfupdate.EnvironmentPair{{Name: "HOME", Value: home}}
	switch operatingSystem(runtime.GOOS) {
	case operatingSystemDarwin:
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
	case operatingSystemLinux:
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
