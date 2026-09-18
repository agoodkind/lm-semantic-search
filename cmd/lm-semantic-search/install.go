package main

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"goodkind.io/lm-semantic-search/internal/installer"
)

func newInstallCmd() *cobra.Command {
	var binDir string
	var releaseVersion string
	var noService bool
	install := &cobra.Command{
		Use:   "install",
		Short: "Install the daemon, MCP adapter, ONNX Runtime, lms alias, and daemon service",
		Long: "Install the lm-semantic-search daemon, MCP adapter, and CLI from one GitHub release, " +
			"stage ONNX Runtime beside the daemon, link lms to the CLI, and install the " +
			"daemon as a launchd or systemd user service.",
		Args: requireNoArgs("install"),
		RunE: func(cmd *cobra.Command, args []string) error {
			if binDir == "" {
				executablePath, err := os.Executable()
				if err != nil {
					slog.Error("resolve executable path failed", "err", err)
					return fmt.Errorf("resolve executable path: %w", err)
				}
				binDir = filepath.Dir(executablePath)
			}
			if err := installer.Run(commandContext(cmd), installer.Options{
				BinDir:         binDir,
				Version:        releaseVersion,
				InstallService: !noService,
				Stdout:         cmd.OutOrStdout(),
			}); err != nil {
				slog.Error("install failed", "err", err)
				return fmt.Errorf("install: %w", err)
			}
			return nil
		},
	}
	install.Flags().StringVar(&binDir, "bin-dir", "", "install dir (default: the directory of this executable)")
	install.Flags().BoolVar(&noService, "no-service", false, "skip launchd/systemd user service setup")
	install.Flags().BoolVar(&noService, "bin-only", false, "alias for --no-service")
	install.Flags().StringVar(&releaseVersion, "version", "", "exact release tag to install (default: latest release)")
	return install
}
