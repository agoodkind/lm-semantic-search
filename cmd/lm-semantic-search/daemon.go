package main

import (
	"context"
	"errors"

	"github.com/spf13/cobra"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

func newDaemonCmd(options *rootOptions) *cobra.Command {
	daemon := &cobra.Command{
		Use:   "daemon",
		Short: "Inspect and control the local daemon",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return errors.New("daemon requires a subcommand")
		},
	}
	daemon.AddCommand(newDaemonStatusCmd(options))
	daemon.AddCommand(newDaemonStopCmd(options))
	daemon.AddCommand(newDaemonDoctorCmd(options))
	return daemon
}

func newDaemonStatusCmd(options *rootOptions) *cobra.Command {
	return &cobra.Command{
		Use:     "status",
		Short:   "Show daemon build and runtime status",
		Args:    requireNoArgs("daemon status"),
		Example: "  lm-semantic-search daemon status\n",
		RunE: func(cmd *cobra.Command, args []string) error {
			return callAndPrint(options.cliOptions(), func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				return client.Version(ctx, &pb.VersionRequest{})
			})
		},
	}
}

func newDaemonStopCmd(options *rootOptions) *cobra.Command {
	return &cobra.Command{
		Use:     "stop",
		Short:   "Request daemon shutdown",
		Args:    requireNoArgs("daemon stop"),
		Example: "  lm-semantic-search daemon stop\n",
		RunE: func(cmd *cobra.Command, args []string) error {
			return callAndPrint(options.cliOptions(), func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				return client.Shutdown(ctx, &pb.ShutdownRequest{})
			})
		},
	}
}

func newDaemonDoctorCmd(options *rootOptions) *cobra.Command {
	return &cobra.Command{
		Use:     "doctor",
		Short:   "Show daemon-local diagnostics",
		Args:    requireNoArgs("daemon doctor"),
		Example: "  lm-semantic-search daemon doctor\n",
		RunE: func(cmd *cobra.Command, args []string) error {
			return callAndPrint(options.cliOptions(), func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				return client.Doctor(ctx, &pb.DoctorRequest{})
			})
		},
	}
}
