// Package client exposes helpers for external modules to talk to the daemon.
package client

import (
	"context"
	"fmt"
	"log/slog"

	lmsemanticsearchv1 "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"google.golang.org/grpc"
)

// DialDaemon creates a gRPC client connection to the local daemon socket.
func DialDaemon(ctx context.Context, socketPath string) (*grpc.ClientConn, lmsemanticsearchv1.SemanticSearchDaemonServiceClient, error) {
	connection, daemonClient, err := grpcutil.DialDaemon(ctx, socketPath)
	if err != nil {
		slog.ErrorContext(ctx, "dial daemon client failed", "socket_path", socketPath, "err", err)
		return nil, nil, fmt.Errorf("dial daemon client: %w", err)
	}
	return connection, daemonClient, nil
}

// ResolveSocketPath returns the same daemon socket path used by the daemon and CLI.
func ResolveSocketPath() string {
	cfg, err := config.Default()
	if err != nil {
		slog.Error("resolve daemon socket path failed", "err", err)
		return ""
	}
	return cfg.SocketPath
}
