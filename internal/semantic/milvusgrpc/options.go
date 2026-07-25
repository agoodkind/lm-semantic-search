// Package milvusgrpc defines the gRPC transport policy for Milvus clients.
package milvusgrpc

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"
)

const defaultMilvusCallTimeout = 60 * time.Second

type deadlineReportable struct {
	timeout time.Duration
	logger  *slog.Logger
}

func (reportable deadlineReportable) ClientReporter(
	ctx context.Context,
	callMeta interceptors.CallMeta,
) (interceptors.Reporter, context.Context) {
	callContext := ctx
	cancel := func() {}
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		callContext, cancel = context.WithTimeout(ctx, reportable.timeout)
	}
	reporter := &deadlineReporter{
		NoopReporter: interceptors.NoopReporter{},
		cancel:       cancel,
		method:       callMeta.FullMethod(),
		timeout:      reportable.timeout,
		logger:       reportable.logger,
	}
	return reporter, callContext
}

type deadlineReporter struct {
	interceptors.NoopReporter

	cancel  context.CancelFunc
	method  string
	timeout time.Duration
	logger  *slog.Logger
}

func (reporter *deadlineReporter) PostCall(err error, duration time.Duration) {
	defer reporter.cancel()

	if status.Code(err) != codes.DeadlineExceeded && !errors.Is(err, context.DeadlineExceeded) {
		return
	}
	reporter.logger.Error(
		"milvus call deadline exceeded",
		"method", reporter.method,
		"timeout_ms", reporter.timeout.Milliseconds(),
		"duration_ms", duration.Milliseconds(),
		"err", err,
	)
}

// DialOptions returns the keepalive and unary deadline policy for a Milvus
// client connection.
func DialOptions(logger *slog.Logger) []grpc.DialOption {
	return []grpc.DialOption{
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                30 * time.Second,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}),
		grpc.WithChainUnaryInterceptor(milvusDeadlineInterceptor(logger)),
	}
}

func milvusDeadlineInterceptor(logger *slog.Logger) grpc.UnaryClientInterceptor {
	reportable := deadlineReportable{
		timeout: defaultMilvusCallTimeout,
		logger:  logger,
	}
	return interceptors.UnaryClientInterceptor(reportable)
}
