package main

import (
	"context"
	"os"
	"strings"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/response"
)

func newCodebaseCmd(options *rootOptions) *cobra.Command {
	codebase := &cobra.Command{
		Use:   "codebase",
		Short: "Inspect and mutate tracked codebases",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
	codebase.AddCommand(newCodebaseListCmd(options))
	codebase.AddCommand(newCodebaseStatusCmd(options))
	codebase.AddCommand(newCodebaseIndexCmd(options))
	codebase.AddCommand(newCodebaseSyncCmd(options))
	codebase.AddCommand(newCodebaseSearchCmd(options))
	codebase.AddCommand(newCodebaseClearCmd(options))
	return codebase
}

func newCodebaseListCmd(options *rootOptions) *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List tracked codebases",
		Args:  requireNoArgs("codebase list"),
		Example: strings.Join([]string{
			"  lm-semantic-search codebase list",
			"  lm-semantic-search --json codebase list",
		}, "\n"),
		RunE: func(cmd *cobra.Command, args []string) error {
			cliOpts := options.cliOptions()
			if cliOpts.outputMode == response.ModeHuman && term.IsTerminal(int(os.Stdout.Fd())) {
				return runCodebaseListTUI(cliOpts)
			}
			return callAndPrint(cliOpts, func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				return client.ListIndexes(ctx, &pb.ListIndexesRequest{})
			})
		},
	}
}

func newCodebaseStatusCmd(options *rootOptions) *cobra.Command {
	return &cobra.Command{
		Use:   "status PATH|ID",
		Short: "Show indexing status for one codebase path",
		Long: strings.Join([]string{
			"Show indexing status for one codebase path.",
			"",
			"Arguments:",
			"  PATH|ID    A codebase path, a symlink to it, or its codebase id",
		}, "\n"),
		Args: requireExactArgs("codebase status requires PATH", 1),
		Example: strings.Join([]string{
			"  lm-semantic-search codebase status /abs/path/to/repo",
		}, "\n"),
		RunE: func(cmd *cobra.Command, args []string) error {
			return callAndPrint(options.cliOptions(), func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				return client.GetIndex(ctx, &pb.GetIndexRequest{Path: args[0]})
			})
		},
	}
}

func newCodebaseIndexCmd(options *rootOptions) *cobra.Command {
	var force bool
	var splitterType string
	var customExtensions []string
	var ignorePatterns []string

	cmd := &cobra.Command{
		Use:   "index PATH|ID",
		Short: "Start background indexing for one codebase",
		Long: strings.Join([]string{
			"Start background indexing for one codebase.",
			"",
			"Arguments:",
			"  PATH|ID    A codebase path, a symlink to it, or its codebase id",
		}, "\n"),
		Args: requireExactArgs("codebase index requires PATH", 1),
		Example: strings.Join([]string{
			"  lm-semantic-search codebase index /abs/path/to/repo",
			"  lm-semantic-search codebase index /abs/path/to/repo --force",
			"  lm-semantic-search codebase index /abs/path/to/repo --splitter ast",
		}, "\n"),
		RunE: func(cmd *cobra.Command, args []string) error {
			clientInfo, err := currentClientInfo()
			if err != nil {
				return err
			}
			request := &pb.StartIndexRequest{
				Path:             args[0],
				Force:            force,
				CustomExtensions: customExtensions,
				IgnorePatterns:   ignorePatterns,
				Client:           clientInfo,
			}
			if splitterType != "" {
				request.Splitter = &pb.SplitterConfig{Type: splitterType}
			}
			return callAndPrint(options.cliOptions(), func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				return client.StartIndex(ctx, request)
			})
		},
	}
	cmd.Flags().BoolVar(&force, "force", false, "force reindex even if already indexed")
	cmd.Flags().StringVar(&splitterType, "splitter", "", "splitter type: ast|langchain")
	cmd.Flags().StringArrayVar(&customExtensions, "extension", nil, "custom file extension to include")
	cmd.Flags().StringArrayVar(&ignorePatterns, "ignore", nil, "ignore pattern to exclude")
	return cmd
}

func newCodebaseSyncCmd(options *rootOptions) *cobra.Command {
	return &cobra.Command{
		Use:   "sync PATH|ID",
		Short: "Start an incremental sync for one tracked codebase",
		Long: strings.Join([]string{
			"Start an incremental sync for one tracked codebase.",
			"",
			"Arguments:",
			"  PATH|ID    A codebase path, a symlink to it, or its codebase id",
		}, "\n"),
		Args: requireExactArgs("codebase sync requires PATH", 1),
		Example: strings.Join([]string{
			"  lm-semantic-search codebase sync /abs/path/to/repo",
		}, "\n"),
		RunE: func(cmd *cobra.Command, args []string) error {
			clientInfo, err := currentClientInfo()
			if err != nil {
				return err
			}
			return callAndPrint(options.cliOptions(), func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				return client.SyncIndex(ctx, &pb.SyncIndexRequest{Path: args[0], Client: clientInfo})
			})
		},
	}
}

func newCodebaseSearchCmd(options *rootOptions) *cobra.Command {
	var limit int
	var extensions []string

	cmd := &cobra.Command{
		Use:   "search PATH|ID QUERY",
		Short: "Search one indexed codebase",
		Long: strings.Join([]string{
			"Search one indexed codebase.",
			"",
			"Arguments:",
			"  PATH|ID    A codebase path, a symlink to it, or its codebase id",
			"  QUERY      Natural-language search query",
		}, "\n"),
		Args: requireExactArgs("codebase search requires PATH and QUERY", 2),
		Example: strings.Join([]string{
			"  lm-semantic-search codebase search /abs/path/to/repo \"indexing flow\"",
			"  lm-semantic-search codebase search /abs/path/to/repo splitter --limit 5",
		}, "\n"),
		RunE: func(cmd *cobra.Command, args []string) error {
			searchLimit, err := safeSearchLimit(limit)
			if err != nil {
				return err
			}
			return callAndPrint(options.cliOptions(), func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				return client.SearchCode(ctx, &pb.SearchCodeRequest{
					Path:            args[0],
					Query:           args[1],
					Limit:           searchLimit,
					ExtensionFilter: extensions,
				})
			})
		},
	}
	cmd.Flags().IntVar(&limit, "limit", 10, "maximum number of results")
	cmd.Flags().StringArrayVar(&extensions, "extension", nil, "file extension filter")
	return cmd
}

func newCodebaseClearCmd(options *rootOptions) *cobra.Command {
	return &cobra.Command{
		Use:   "clear PATH|ID",
		Short: "Clear one tracked codebase",
		Long: strings.Join([]string{
			"Clear one tracked codebase.",
			"",
			"Arguments:",
			"  PATH|ID    A codebase path, a symlink to it, or its codebase id",
		}, "\n"),
		Args: requireExactArgs("codebase clear requires PATH", 1),
		Example: strings.Join([]string{
			"  lm-semantic-search codebase clear /abs/path/to/repo",
		}, "\n"),
		RunE: func(cmd *cobra.Command, args []string) error {
			clientInfo, err := currentClientInfo()
			if err != nil {
				return err
			}
			return callAndPrint(options.cliOptions(), func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
				return client.ClearIndex(ctx, &pb.ClearIndexRequest{Path: args[0], Client: clientInfo})
			})
		},
	}
}
