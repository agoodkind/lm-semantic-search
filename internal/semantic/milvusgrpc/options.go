// Package milvusgrpc defines the gRPC transport policy for Milvus clients.
package milvusgrpc

import (
	"context"
	"errors"
	"log/slog"
	"slices"
	"time"

	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors"
	"goodkind.io/lm-semantic-search/internal/clock"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"
)

const (
	// defaultMetadataCallTimeout bounds a Milvus call whose server-side cost does
	// not grow with the number of rows it touches: schema reads, load-state
	// polls, searches, and queries. A call still outstanding after this long is
	// wedged rather than slow, and failing it here is what stops an ingest job
	// from reporting itself as running while nothing moves.
	defaultMetadataCallTimeout = 60 * time.Second

	// defaultMutationCallTimeout bounds a Milvus call that writes or deletes
	// rows. One filter-based Delete covers every row of a conversation, and
	// Milvus evaluates the non-primary relativePath predicate across the loaded
	// collection before it answers, so the matching row count is unbounded and a
	// valid bulk delete on a large collection can run well past the metadata
	// bound. A separate, larger bound keeps the call bounded without turning a
	// valid delete into a milvus_unavailable ingest failure.
	defaultMutationCallTimeout = 5 * time.Minute

	// keepaliveTime is how long the connection stays idle before the client
	// sends an HTTP/2 keepalive ping. It matches grpc-go's default server
	// keepalive.EnforcementPolicy MinTime (5 minutes, internal/transport
	// defaultKeepalivePolicyMinTime). A client that pings sooner collects ping
	// strikes and the server closes the connection with ENHANCE_YOUR_CALM and
	// "too_many_pings", which is the outage this policy exists to avoid.
	keepaliveTime = 5 * time.Minute

	// keepaliveTimeout is how long the client waits for a ping ack before it
	// treats the connection as dead.
	keepaliveTimeout = 20 * time.Second
)

// mutationMethodNames lists the Milvus unary methods that write or delete
// collection rows and therefore carry the mutation deadline. The names match
// the milvuspb MilvusService method names exactly, so the unrelated
// DeleteCredential and DeleteUserTags calls keep the metadata bound.
var mutationMethodNames = []string{"Delete", "Insert", "Upsert", "Flush"}

// CallTimeouts is the per-call deadline policy a Milvus client applies to every
// unary call. Two classes are enough because Milvus calls split cleanly into
// calls whose cost is independent of the stored row count and calls that write
// or delete an unbounded number of rows.
type CallTimeouts struct {
	// Metadata bounds schema, load-state, search, and query calls.
	Metadata time.Duration
	// Mutation bounds row-writing and row-deleting calls.
	Mutation time.Duration
}

// DefaultCallTimeouts returns the deadline policy a Milvus client uses unless
// its caller tunes the bounds.
func DefaultCallTimeouts() CallTimeouts {
	return CallTimeouts{
		Metadata: defaultMetadataCallTimeout,
		Mutation: defaultMutationCallTimeout,
	}
}

// forMethod returns the bound that applies to one Milvus unary method, named
// without its service prefix.
func (timeouts CallTimeouts) forMethod(method string) time.Duration {
	if slices.Contains(mutationMethodNames, method) {
		return timeouts.Mutation
	}
	return timeouts.Metadata
}

type deadlineReportable struct {
	timeouts CallTimeouts
	logger   *slog.Logger
}

func (reportable deadlineReportable) ClientReporter(
	ctx context.Context,
	callMeta interceptors.CallMeta,
) (interceptors.Reporter, context.Context) {
	timeout := reportable.timeouts.forMethod(callMeta.Method)
	callContext := ctx
	cancel := func() {}
	if needsBound(ctx, timeout) {
		callContext, cancel = context.WithTimeout(ctx, timeout)
	}
	reporter := &deadlineReporter{
		NoopReporter: interceptors.NoopReporter{},
		cancel:       cancel,
		method:       callMeta.FullMethod(),
		timeout:      timeout,
		logger:       reportable.logger,
	}
	return reporter, callContext
}

// needsBound reports whether the call must carry the transport bound. A caller
// that set no deadline needs it, and so does a caller whose own deadline is
// later than the bound, because that caller would otherwise let a wedged call
// block for its full budget. A caller that already asked for less keeps its own
// earlier deadline untouched.
func needsBound(ctx context.Context, timeout time.Duration) bool {
	deadline, hasDeadline := ctx.Deadline()
	if !hasDeadline {
		return true
	}
	return deadline.After(clock.Now().Add(timeout))
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
func DialOptions(logger *slog.Logger, timeouts CallTimeouts) []grpc.DialOption {
	return []grpc.DialOption{
		grpc.WithKeepaliveParams(keepaliveParameters()),
		grpc.WithChainUnaryInterceptor(milvusDeadlineInterceptor(logger, timeouts)),
	}
}

// keepaliveParameters returns the client keepalive policy. It never pings an
// idle connection, because a server running grpc-go's default enforcement
// permits no idle ping at all and closes the connection once a client has sent
// a few of them.
func keepaliveParameters() keepalive.ClientParameters {
	return keepalive.ClientParameters{
		Time:                keepaliveTime,
		Timeout:             keepaliveTimeout,
		PermitWithoutStream: false,
	}
}

func milvusDeadlineInterceptor(logger *slog.Logger, timeouts CallTimeouts) grpc.UnaryClientInterceptor {
	reportable := deadlineReportable{
		timeouts: timeouts,
		logger:   logger,
	}
	return interceptors.UnaryClientInterceptor(reportable)
}
