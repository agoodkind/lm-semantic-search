package main

import (
	"context"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/response"
)

const (
	// defaultStatusInterval is how often the live screen re-reads the daemon.
	defaultStatusInterval = 2 * time.Second
	// minimumStatusInterval floors the cadence so several open screens cannot
	// become a busy loop against the daemon socket.
	minimumStatusInterval = 500 * time.Millisecond
)

func newStatusCmd(options *rootOptions) *cobra.Command {
	var once bool
	var interval time.Duration

	cmd := &cobra.Command{
		Use:   "status",
		Short: "Show what the daemon is doing and what its counters read",
		Long: strings.Join([]string{
			"Show what the daemon is doing and what its counters read.",
			"",
			"On a terminal this refreshes in place and shows the change since the",
			"previous read. Piped or under --json it prints one snapshot and exits.",
			"",
			"Counter names are the daemon's own, so a name here greps the daemon log.",
		}, "\n"),
		Args: requireNoArgs("status"),
		Example: strings.Join([]string{
			"  lm-semantic-search status",
			"  lm-semantic-search status --once",
			"  lm-semantic-search --json status",
			"  lm-semantic-search status | grep embed_vectors_total",
		}, "\n"),
		RunE: func(cmd *cobra.Command, args []string) error {
			cliOpts := options.cliOptions()
			if interval < minimumStatusInterval {
				interval = minimumStatusInterval
			}
			live := cliOpts.outputMode == response.ModeHuman &&
				!once &&
				term.IsTerminal(int(os.Stdout.Fd()))
			if live {
				return runStatusTUI(cliOpts, interval)
			}
			return callAndPrint(cliOpts, func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				return client.GetStatus(ctx, &pb.GetStatusRequest{})
			})
		},
	}
	cmd.Flags().BoolVar(&once, "once", false, "print one snapshot even on a terminal")
	cmd.Flags().DurationVar(&interval, "interval", defaultStatusInterval, "refresh cadence for the live screen")
	return cmd
}
