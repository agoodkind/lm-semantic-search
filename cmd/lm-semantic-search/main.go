// Command lm-semantic-search is the operator CLI for the local daemon.
package main

import (
	"fmt"
	"log/slog"
	"os"
)

func main() {
	slog.Debug("cli.main.entry", "component", "cli")
	os.Exit(run())
}

func run() int {
	if err := executeRoot(os.Args[1:]); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "Error: %s\n", err.Error())
		return 1
	}
	return 0
}
