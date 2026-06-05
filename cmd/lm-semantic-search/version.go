package main

import (
	"fmt"
	"log/slog"

	"github.com/spf13/cobra"

	"goodkind.io/gklog/version"
)

func newVersionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print build information",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			_, err := fmt.Fprintf(cmd.OutOrStdout(), "version=%s commit=%s build_time=%s\n", version.String(), version.Commit, version.BuildTime)
			if err != nil {
				slog.Error("write version output failed", "err", err)
				return fmt.Errorf("write version output: %w", err)
			}
			return nil
		},
	}
}
