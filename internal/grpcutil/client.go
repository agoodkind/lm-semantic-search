// Package grpcutil contains client helpers for talking to the daemon.
package grpcutil

import (
	"context"
	"fmt"
	"log/slog"
	"net"

	"goodkind.io/gklog/correlation"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// MaxMessageBytes is the gRPC message ceiling for the daemon socket on both
// the client send and server receive sides. A conversation upsert carries one
// whole conversation's documents in a single message because the engine
// replaces a conversation atomically, and a long transcript exceeds gRPC's
// 4 MiB default, so the local-socket ceiling is raised instead of splitting
// a conversation across messages.
const MaxMessageBytes = 64 << 20

// DialDaemon creates a gRPC client connection to the local daemon
// socket. Callers should wrap their context with [WithCorrelation]
// (or call [correlation.NewOutgoingContext] directly) so the daemon
// receives the trace, span, and request identifiers.
func DialDaemon(ctx context.Context, socketPath string) (*grpc.ClientConn, pb.SemanticSearchDaemonServiceClient, error) {
	connection, err := grpc.NewClient(
		"passthrough:///unix",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallSendMsgSize(MaxMessageBytes),
			grpc.MaxCallRecvMsgSize(MaxMessageBytes),
		),
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			var dialer net.Dialer
			return dialer.DialContext(ctx, "unix", socketPath)
		}),
	)
	if err != nil {
		slog.ErrorContext(ctx, "create gRPC client failed", "socket_path", socketPath, "err", err)
		return nil, nil, fmt.Errorf("create gRPC client for %s: %w", socketPath, err)
	}
	connection.Connect()
	return connection, pb.NewSemanticSearchDaemonServiceClient(connection), nil
}

// WithCorrelation builds an outgoing-metadata context that carries
// the [correlation.Context] from ctx, building a fresh one when ctx
// has none. Daemon callers wrap each gRPC client invocation with this
// helper instead of using a global interceptor.
func WithCorrelation(ctx context.Context) context.Context {
	ctx, _ = correlation.Ensure(ctx, "")
	return correlation.NewOutgoingContext(ctx)
}
