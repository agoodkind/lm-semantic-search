package main

import (
	"context"
	"strings"

	"github.com/spf13/cobra"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

func newJobCmd(options *rootOptions) *cobra.Command {
	job := &cobra.Command{
		Use:   "job",
		Short: "Inspect and manage daemon jobs",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
	job.AddCommand(newJobListCmd(options))
	job.AddCommand(newJobGetCmd(options))
	job.AddCommand(newJobCancelCmd(options))
	return job
}

func newJobListCmd(options *rootOptions) *cobra.Command {
	var codebaseID string

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List tracked jobs",
		Args:  requireNoArgs("job list"),
		Example: strings.Join([]string{
			"  lm-semantic-search job list",
			"  lm-semantic-search job list --codebase-id cb_123",
		}, "\n"),
		RunE: func(cmd *cobra.Command, args []string) error {
			return callAndPrint(options.cliOptions(), func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				return client.ListJobs(ctx, &pb.ListJobsRequest{CodebaseId: codebaseID})
			})
		},
	}
	cmd.Flags().StringVar(&codebaseID, "codebase-id", "", "filter jobs by codebase id")
	return cmd
}

func newJobGetCmd(options *rootOptions) *cobra.Command {
	return &cobra.Command{
		Use:   "get JOB_ID",
		Short: "Get one tracked job",
		Long: strings.Join([]string{
			"Get one tracked job.",
			"",
			"Arguments:",
			"  JOB_ID    Daemon job identifier",
		}, "\n"),
		Args: requireExactArgs("job get requires JOB_ID", 1),
		Example: strings.Join([]string{
			"  lm-semantic-search job get job_123",
		}, "\n"),
		RunE: func(cmd *cobra.Command, args []string) error {
			return callAndPrint(options.cliOptions(), func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				return client.GetJob(ctx, &pb.GetJobRequest{JobId: args[0]})
			})
		},
	}
}

func newJobCancelCmd(options *rootOptions) *cobra.Command {
	return &cobra.Command{
		Use:   "cancel JOB_ID",
		Short: "Cancel one tracked job",
		Long: strings.Join([]string{
			"Cancel one tracked job.",
			"",
			"Arguments:",
			"  JOB_ID    Daemon job identifier",
		}, "\n"),
		Args: requireExactArgs("job cancel requires JOB_ID", 1),
		Example: strings.Join([]string{
			"  lm-semantic-search job cancel job_123",
		}, "\n"),
		RunE: func(cmd *cobra.Command, args []string) error {
			clientInfo, err := currentClientInfo()
			if err != nil {
				return err
			}
			return callAndPrint(options.cliOptions(), func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				return client.CancelJob(ctx, &pb.CancelJobRequest{JobId: args[0], Client: clientInfo})
			})
		},
	}
}
