package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/response"
	"goodkind.io/lm-semantic-search/internal/statushistory"
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
	var since time.Duration

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
			"  lm-semantic-search --json status --since 1h",
			"  lm-semantic-search status | grep embed_vectors_total",
		}, "\n"),
		RunE: func(cmd *cobra.Command, args []string) error {
			cliOpts := options.cliOptions()
			if cmd.Flags().Changed("since") {
				if since <= 0 {
					return fmt.Errorf("--since must be positive")
				}
				return runHistoricalStatus(cliOpts, since)
			}
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
	cmd.Flags().DurationVar(&since, "since", 0, "report daemon history for a duration")
	return cmd
}

func runHistoricalStatus(options cliOptions, since time.Duration) error {
	result, err := callDaemon(options, func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
		return client.GetStatus(ctx, &pb.GetStatusRequest{})
	})
	if err != nil {
		return err
	}
	status, ok := result.(*pb.GetStatusResponse)
	if !ok {
		return fmt.Errorf("unexpected response type from GetStatus")
	}
	configValue, err := config.Default()
	if err != nil {
		return fmt.Errorf("resolve daemon state root: %w", err)
	}
	if err := validateHistoricalSocket(options, configValue.SocketPath); err != nil {
		return err
	}
	report, err := statushistory.Build(statushistory.Input{
		StateRoot: configValue.StateRoot,
		Since:     since,
		Now:       time.Time{},
		Status:    status,
	})
	if err != nil {
		slog.Error("build historical status failed", "err", err)
		return fmt.Errorf("build historical status: %w", err)
	}
	output, err := MarshalHistoricalStatus(options.outputMode, report)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(os.Stdout, output)
	if err != nil {
		return fmt.Errorf("write historical status: %w", err)
	}
	return nil
}

// MarshalHistoricalStatus formats one historical report for the selected CLI output mode.
func MarshalHistoricalStatus(mode response.Mode, report statushistory.Report) (string, error) {
	if mode == response.ModeJSON {
		var output strings.Builder
		encoder := json.NewEncoder(&output)
		encoder.SetEscapeHTML(false)
		if err := encoder.Encode(report); err != nil {
			slog.Error("encode historical status failed", "err", err)
			return "", fmt.Errorf("write historical JSON: %w", err)
		}
		return strings.TrimSuffix(output.String(), "\n"), nil
	}
	human := statushistory.Human(report)
	if mode == response.ModeSingleLine {
		for line := range strings.SplitSeq(human, "\n") {
			if trimmed := strings.TrimSpace(line); trimmed != "" {
				return trimmed, nil
			}
		}
		return "", nil
	}
	return human, nil
}

func validateHistoricalSocket(options cliOptions, configuredSocket string) error {
	if options.socketPath != configuredSocket {
		return fmt.Errorf("--socket does not match the configured daemon socket")
	}
	return nil
}
