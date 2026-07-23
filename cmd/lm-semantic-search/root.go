package main

import (
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/spf13/cobra"

	"goodkind.io/gklog/version"
	daemonclient "goodkind.io/lm-semantic-search/client"
	"goodkind.io/lm-semantic-search/internal/response"
)

type cliOptions struct {
	socketPath string
	outputMode response.Mode
}

type rootOptions struct {
	socketPath  string
	jsonOutput  bool
	outputValue string
}

func (options *rootOptions) resolvedMode() response.Mode {
	mode := response.ParseMode(options.outputValue)
	if options.jsonOutput {
		mode = response.ModeJSON
	}
	return mode
}

func (options *rootOptions) cliOptions() cliOptions {
	return cliOptions{
		socketPath: options.socketPath,
		outputMode: options.resolvedMode(),
	}
}

func executeRoot(args []string) error {
	defaultSocketPath := daemonclient.ResolveSocketPath()
	if defaultSocketPath == "" {
		err := errors.New("daemon socket path is empty")
		slog.Error("load config failed", "err", err)
		return fmt.Errorf("load config: %w", err)
	}

	root := newRoot(defaultSocketPath)
	root.SetArgs(args)
	if err := root.Execute(); err != nil {
		return fmt.Errorf("execute root command: %w", err)
	}
	return nil
}

func newRoot(defaultSocketPath string) *cobra.Command {
	options := &rootOptions{
		socketPath:  defaultSocketPath,
		jsonOutput:  false,
		outputValue: string(response.ModeHuman),
	}

	root := &cobra.Command{
		Use:     "lm-semantic-search",
		Short:   "Inspect and operate the local semantic indexing daemon",
		Version: version.String(),
		Args:    cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmd.Help()
		},
	}
	root.CompletionOptions.DisableDefaultCmd = true
	root.SilenceErrors = true
	root.SilenceUsage = false
	root.PersistentPreRun = func(cmd *cobra.Command, _ []string) {
		cmd.SilenceUsage = true
	}
	root.SetHelpFunc(func(cmd *cobra.Command, _ []string) {
		_, _ = fmt.Fprint(cmd.OutOrStdout(), helpText(cmd))
	})

	root.PersistentFlags().StringVar(&options.socketPath, "socket", options.socketPath, "unix socket path")
	root.PersistentFlags().BoolVar(&options.jsonOutput, "json", options.jsonOutput, "print compact JSON instead of human text")
	root.PersistentFlags().StringVar(&options.outputValue, "output", options.outputValue, "output mode: human, json, or single-line")

	root.AddCommand(newVersionCmd())
	root.AddCommand(newCodebaseCmd(options))
	root.AddCommand(newJobCmd(options))
	root.AddCommand(newDaemonCmd(options))
	root.AddCommand(newUpdateCmd(options))
	root.AddCommand(newProfileCmd())
	return root
}

func helpText(cmd *cobra.Command) string {
	description := strings.TrimSpace(cmd.Long)
	if description == "" {
		description = strings.TrimSpace(cmd.Short)
	}
	usage := cmd.UsageString()
	if description == "" {
		return usage
	}
	return description + "\n\n" + usage
}

func requireNoArgs(name string) cobra.PositionalArgs {
	return func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 {
			return nil
		}
		return fmt.Errorf("%s accepts no arguments", name)
	}
}

func requireExactArgs(message string, count int) cobra.PositionalArgs {
	return func(cmd *cobra.Command, args []string) error {
		if len(args) != count {
			return errors.New(message)
		}
		for _, arg := range args {
			if strings.TrimSpace(arg) == "" {
				return errors.New(message)
			}
		}
		return nil
	}
}
