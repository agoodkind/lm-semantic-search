// Command tokenizers-dep stages the static Hugging Face tokenizer library.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"goodkind.io/lm-semantic-search/internal/tokenizersdist"
)

const toolLogPrefix = "setup-cgo-tokenizers"

func main() {
	slog.Debug("tokenizers dependency process entry")
	os.Exit(runMain())
}

func runMain() int {
	slog.Debug("tokenizers dependency command entry")
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := tokenizersdist.Stage(ctx); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "%s: %v\n", toolLogPrefix, err)
		return 1
	}

	return 0
}
