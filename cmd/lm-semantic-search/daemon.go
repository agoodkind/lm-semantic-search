package main

import (
	"context"
	"errors"
	"strings"

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
	daemon.AddCommand(newDaemonMaintenanceCmd(options))
	return daemon
}

// newDaemonMaintenanceCmd groups the maintenance mode switch. on stops every
// daemon-driven store interaction so an operator can back up or restore the
// vector store; off lets the next sweep resume it.
func newDaemonMaintenanceCmd(options *rootOptions) *cobra.Command {
	maintenance := &cobra.Command{
		Use:   "maintenance",
		Short: "Pause or resume the daemon's store work for a backup or restore",
		Long: strings.Join([]string{
			"Pause or resume the daemon's store work for a backup or restore.",
			"",
			"While maintenance mode is on the daemon starts no background sync, repair",
			"pass, automatic rebuild, or collection load, refuses index and conversation",
			"writes, and fails searches fast with a maintenance status. Jobs already",
			"running keep going; the reply counts them so you can wait or cancel them.",
			"The mode survives a daemon restart until it is turned off.",
		}, "\n"),
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return errors.New("daemon maintenance requires on or off")
		},
	}
	maintenance.AddCommand(newDaemonMaintenanceOnCmd(options))
	maintenance.AddCommand(newDaemonMaintenanceOffCmd(options))
	return maintenance
}

func newDaemonMaintenanceOnCmd(options *rootOptions) *cobra.Command {
	var reason string
	cmd := &cobra.Command{
		Use:     "on",
		Short:   "Turn maintenance mode on",
		Args:    requireNoArgs("daemon maintenance on"),
		Example: "  lm-semantic-search daemon maintenance on --reason \"milvus restore\"\n",
		RunE: func(cmd *cobra.Command, args []string) error {
			return callAndPrint(options.cliOptions(), func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				clientInfo, err := resolveClientInfo()
				if err != nil {
					return nil, err
				}
				return client.SetMaintenanceMode(ctx, &pb.SetMaintenanceModeRequest{
					Enabled: true,
					Reason:  reason,
					Client:  clientInfo,
				})
			})
		},
	}
	cmd.Flags().StringVar(&reason, "reason", "", "note shown on every status surface while the mode is on")
	return cmd
}

func newDaemonMaintenanceOffCmd(options *rootOptions) *cobra.Command {
	return &cobra.Command{
		Use:     "off",
		Short:   "Turn maintenance mode off",
		Args:    requireNoArgs("daemon maintenance off"),
		Example: "  lm-semantic-search daemon maintenance off\n",
		RunE: func(cmd *cobra.Command, args []string) error {
			return callAndPrint(options.cliOptions(), func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				clientInfo, err := resolveClientInfo()
				if err != nil {
					return nil, err
				}
				return client.SetMaintenanceMode(ctx, &pb.SetMaintenanceModeRequest{
					Enabled: false,
					Reason:  "",
					Client:  clientInfo,
				})
			})
		},
	}
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
