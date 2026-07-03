package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/spf13/cobra"

	"goodkind.io/gklog/version"
	"goodkind.io/go-makefile/selfupdate"
	daemonclient "goodkind.io/lm-semantic-search/client"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/updateopts"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var dialDaemon = daemonclient.DialDaemon

func newUpdateCmd(options *rootOptions) *cobra.Command {
	update := &cobra.Command{
		Use:   "update",
		Short: "Check and apply release updates",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return errors.New("update requires a subcommand")
		},
	}
	update.AddCommand(newUpdateCheckCmd())
	update.AddCommand(newUpdateApplyCmd(options))
	update.AddCommand(newUpdateStatusCmd())
	return update
}

func defaultUpdateOverrides(dryRun bool) updateopts.Overrides {
	return updateopts.Overrides{
		Client:     nil,
		InstallDir: "",
		StateRoot:  "",
		CacheDir:   "",
		DryRun:     dryRun,
		Log:        nil,
	}
}

func newUpdateCheckCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "check",
		Short: "Check the latest release",
		Args:  requireNoArgs("update check"),
		RunE: func(cmd *cobra.Command, args []string) error {
			option, err := updateopts.CheckOptions(defaultUpdateOverrides(false))
			if err != nil {
				slog.Error("build update check options failed", "err", err)
				return fmt.Errorf("build update check options: %w", err)
			}
			result, err := selfupdate.Check(commandContext(cmd), option)
			if err != nil {
				slog.Error("check update failed", "err", err)
				return fmt.Errorf("check update: %w", err)
			}
			printCheckResult(cmd, result)
			return nil
		},
	}
}

func newUpdateApplyCmd(options *rootOptions) *cobra.Command {
	var dryRun bool
	apply := &cobra.Command{
		Use:   "apply",
		Short: "Apply the latest release",
		Args:  requireNoArgs("update apply"),
		RunE: func(cmd *cobra.Command, args []string) error {
			result, err := updateopts.ApplyAll(commandContext(cmd), defaultUpdateOverrides(dryRun))
			if err != nil {
				slog.Error("apply update failed", "err", err)
				return fmt.Errorf("apply update: %w", err)
			}
			if !result.UpdateAvailable {
				_, _ = fmt.Fprintln(cmd.OutOrStdout(), "lm-semantic-search: already current")
				return nil
			}
			if dryRun || result.DryRun {
				_, _ = fmt.Fprintln(cmd.OutOrStdout(), "lm-semantic-search: update apply dry run ok")
				return nil
			}
			restarted, err := requestDaemonShutdown(commandContext(cmd), options.socketPath)
			if err != nil {
				slog.Error("restart daemon after update failed", "err", err)
				return fmt.Errorf("restart daemon after update: %w", err)
			}
			if restarted {
				_, _ = fmt.Fprintln(cmd.OutOrStdout(), "lm-semantic-search: update applied and daemon restart requested")
				return nil
			}
			_, _ = fmt.Fprintln(cmd.OutOrStdout(), "lm-semantic-search: update applied; daemon not running")
			return nil
		},
	}
	apply.Flags().BoolVar(&dryRun, "dry-run", false, "download and verify without installing")
	return apply
}

func newUpdateStatusCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "status",
		Short: "Show update state",
		Args:  requireNoArgs("update status"),
		RunE: func(cmd *cobra.Command, args []string) error {
			statePath, err := updateopts.StatePath(defaultUpdateOverrides(false))
			if err != nil {
				slog.Error("resolve update state path failed", "err", err)
				return fmt.Errorf("resolve update state path: %w", err)
			}
			state, err := selfupdate.LoadState(statePath)
			if err != nil {
				if os.IsNotExist(err) {
					printUpdateStatus(cmd, selfupdate.State{})
					return nil
				}
				slog.Error("load update state failed", "err", err)
				return fmt.Errorf("load update state: %w", err)
			}
			printUpdateStatus(cmd, state)
			return nil
		},
	}
}

func commandContext(cmd *cobra.Command) context.Context {
	ctx := cmd.Context()
	if ctx == nil {
		return context.Background()
	}
	return ctx
}

func printCheckResult(cmd *cobra.Command, result selfupdate.CheckResult) {
	_, _ = fmt.Fprintln(cmd.OutOrStdout(), "current version: "+result.CurrentVersion)
	_, _ = fmt.Fprintln(cmd.OutOrStdout(), "latest tag:      "+result.LatestTag)
	_, _ = fmt.Fprintln(cmd.OutOrStdout(), "asset:           "+result.AssetName)
	if result.UpdateAvailable {
		_, _ = fmt.Fprintln(cmd.OutOrStdout(), "update available: yes")
		return
	}
	_, _ = fmt.Fprintln(cmd.OutOrStdout(), "update available: no")
}

func printUpdateStatus(cmd *cobra.Command, state selfupdate.State) {
	_, _ = fmt.Fprintln(cmd.OutOrStdout(), "current version:   "+version.Version)
	_, _ = fmt.Fprintln(cmd.OutOrStdout(), "current commit:    "+version.Commit)
	_, _ = fmt.Fprintln(cmd.OutOrStdout(), "current buildHash: "+version.BinHash)
	if !state.LastCheckAt.IsZero() {
		_, _ = fmt.Fprintln(cmd.OutOrStdout(), "last check:        "+state.LastCheckAt.Format(time.RFC3339))
	}
	if !state.NextCheckAt.IsZero() {
		_, _ = fmt.Fprintln(cmd.OutOrStdout(), "next check:        "+state.NextCheckAt.Format(time.RFC3339))
	}
	if state.LatestTag != "" {
		_, _ = fmt.Fprintln(cmd.OutOrStdout(), "latest tag:        "+state.LatestTag)
	}
	if state.AppliedTag != "" {
		_, _ = fmt.Fprintln(cmd.OutOrStdout(), "applied tag:       "+state.AppliedTag)
	}
	if state.LastResult != "" {
		_, _ = fmt.Fprintln(cmd.OutOrStdout(), "last result:       "+state.LastResult)
	}
	if state.LastError != "" {
		_, _ = fmt.Fprintln(cmd.OutOrStdout(), "last error:        "+state.LastError)
	}
}

func requestDaemonShutdown(ctx context.Context, socketPath string) (bool, error) {
	connection, client, err := dialDaemon(ctx, socketPath)
	if err != nil {
		return false, fmt.Errorf("dial daemon: %w", err)
	}
	if connection != nil {
		defer func() { _ = connection.Close() }()
	}
	_, err = client.Shutdown(ctx, &pb.ShutdownRequest{})
	if err != nil {
		if status.Code(err) == codes.Unavailable {
			return false, nil
		}
		slog.ErrorContext(ctx, "shutdown daemon after update failed", "err", err)
		return false, fmt.Errorf("shutdown daemon: %w", err)
	}
	return true, nil
}
