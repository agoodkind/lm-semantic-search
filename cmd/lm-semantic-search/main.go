// Command lm-semantic-search is the operator CLI for the local daemon.
package main

import (
	"fmt"
	"log/slog"
	"os"
	"strings"

	"goodkind.io/gklog/correlation"
)

func main() {
	slog.Debug("cli.main.entry", "component", "cli")
	os.Exit(run())
}

func run() int {
	if err := executeRoot(os.Args[1:]); err != nil {
		_, _ = fmt.Fprint(os.Stderr, formatCLIError(err))
		return 1
	}
	return 0
}

func formatCLIError(err error) string {
	message := err.Error()
	if strings.HasPrefix(message, correlation.HeaderMarker) {
		return message + "\n"
	}
	return fmt.Sprintf("Error: %s\n", message)
}
