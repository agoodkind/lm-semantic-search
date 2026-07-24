package main

import (
	"fmt"
	"log/slog"
	"strings"

	"github.com/spf13/cobra"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

const profileArgumentChoices = "standard|offline"

func newProfileCmd() *cobra.Command {
	var offlineModel string
	validModelNames := strings.Join(offlinemodel.Names(), ", ")
	command := &cobra.Command{
		Use:   "profile PROFILE",
		Short: "Select the daemon capability profile",
		Long: strings.Join([]string{
			"Select the daemon capability profile.",
			"",
			"Arguments:",
			"  PROFILE    The profile to persist: standard or offline",
			"",
			"The --model flag selects the offline embedding model.",
			"Valid offline model names: " + validModelNames + ".",
			"",
			"The running daemon reads the new profile after it restarts.",
		}, "\n"),
		Args:      validateProfileArgs,
		ValidArgs: []string{config.ProfileStandard, config.ProfileOffline},
		Example: strings.Join([]string{
			"  lm-semantic-search profile offline",
			"  lm-semantic-search profile offline --model bge-small",
			"  lm-semantic-search profile standard",
		}, "\n"),
		RunE: func(cmd *cobra.Command, args []string) error {
			profile := args[0]
			if err := validateOfflineModel(profile, offlineModel); err != nil {
				return err
			}
			daemonConfig, err := config.Default()
			if err != nil {
				slog.Error("load daemon config failed", "err", err)
				return fmt.Errorf("load daemon config: %w", err)
			}
			if err := config.SetProfile(daemonConfig.ConfigPath, profile); err != nil {
				slog.Error(
					"set daemon profile failed",
					"path",
					daemonConfig.ConfigPath,
					"profile",
					profile,
					"err",
					err,
				)
				return fmt.Errorf("set daemon profile: %w", err)
			}
			if err := config.SetOfflineModel(daemonConfig.ConfigPath, offlineModel); err != nil {
				slog.Error(
					"set offline embedding model failed",
					"path",
					daemonConfig.ConfigPath,
					"model",
					offlineModel,
					"err",
					err,
				)
				return fmt.Errorf("set offline embedding model: %w", err)
			}
			_, err = fmt.Fprintf(
				cmd.OutOrStdout(),
				"Profile set to %s in %s. Restart the daemon to apply it.\n",
				profile,
				daemonConfig.ConfigPath,
			)
			if err != nil {
				slog.Error("write profile result failed", "err", err)
				return fmt.Errorf("write profile result: %w", err)
			}
			return nil
		},
	}
	command.Flags().StringVar(
		&offlineModel,
		"model",
		"",
		"offline embedding model ("+validModelNames+")",
	)
	return command
}

func validateProfileArgs(_ *cobra.Command, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf(
			"profile requires a PROFILE argument (%s)",
			profileArgumentChoices,
		)
	}
	if len(args) > 1 {
		extraArguments := strings.Join(args[1:], " ")
		return fmt.Errorf("profile received too many arguments: %s", extraArguments)
	}
	return nil
}

func validateOfflineModel(profile string, model string) error {
	if model == "" {
		return nil
	}
	if profile != config.ProfileOffline {
		return fmt.Errorf("--model only applies to the offline profile")
	}
	if _, err := offlinemodel.Resolve(model); err != nil {
		slog.Error("validate offline model failed", "model", model, "err", err)
		return fmt.Errorf("validate offline model: %w", err)
	}
	return nil
}
