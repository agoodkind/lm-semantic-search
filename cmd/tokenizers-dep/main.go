// Command tokenizers-dep stages the static Hugging Face tokenizer library.
package main

import (
	"log/slog"
	"os"

	"goodkind.io/lm-semantic-search/internal/tokenizersdist"
)

func main() {
	slog.Debug("tokenizers dependency process entry")
	os.Exit(tokenizersdist.Run())
}
