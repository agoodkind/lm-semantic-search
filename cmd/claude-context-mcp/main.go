// Command claude-context-mcp will host the MCP adapter process.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/zilliztech/claude-context-go/internal/mcpserver"
)

func main() {
	if err := mcpserver.Run(context.Background()); err != nil {
		slog.Error("claude-context-mcp failed", "err", err)
		os.Exit(1)
	}
}
