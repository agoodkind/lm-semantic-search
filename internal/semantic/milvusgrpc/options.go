// Package milvusgrpc defines the gRPC transport policy for Milvus clients.
package milvusgrpc

import (
	"context"
	"errors"
	"log/slog"
	"slices"
	"strings"
	"time"

	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

const (
	// defaultMetadataCallTimeout bounds a Milvus call whose server-side cost does
	// not grow with the number of rows it touches: schema reads, load-state
	// polls, searches, and queries. A call still outstanding after this long is
	// wedged rather than slow, and failing it here is what stops an ingest job
	// from reporting itself as running while nothing moves.
	defaultMetadataCallTimeout = 60 * time.Second

	// defaultMutationCallTimeout is the bound a client applies to a Milvus call
	// that writes or deletes rows when its configuration names no other value.
	// One filter-based Delete covers every row of a conversation, and Milvus
	// evaluates the non-primary relativePath predicate across the loaded
	// collection before it answers, so the matching row count is unbounded and a
	// valid bulk delete on a large collection can run well past the metadata
	// bound. A separate, larger bound keeps the call bounded without turning a
	// valid delete into a milvus_unavailable ingest failure. No fixed duration is
	// provably sufficient for an unbounded row count, so the bound is tunable
	// through CallTimeouts.WithMutation and an operator whose collection is large
	// enough to outlast this default raises it without a rebuild.
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

// mutationMethodNames lists the Milvus unary methods that write, delete, or
// persist collection data and therefore carry the mutation deadline. The list
// was read off the pinned milvuspb MilvusService_ServiceDesc method table
// rather than inferred from method names, so it covers the whole data-mutation
// surface that service declares: the four row-level calls plus TruncateCollection,
// Import, ReplicateMessage, and FlushAll, whose durations all scale with data
// volume rather than with schema size.
//
// Every entry matches a service method name exactly. Exact matching is what
// keeps the administrative deletes on the metadata bound, because a prefix or
// substring rule would sweep DeleteCredential and DeleteUserTags in with Delete.
// Collection, partition, index, alias, database, resource-group, and role calls
// stay on the metadata bound for the same reason.
var mutationMethodNames = []string{
	"Delete",
	"Flush",
	"FlushAll",
	"Import",
	"Insert",
	"ReplicateMessage",
	"TruncateCollection",
	"Upsert",
}

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

// CallObserver receives every unary Milvus method and its request before the
// transport sends it. Callers install one on a construction context when they
// need an isolated audit of a client's backend activity.
type CallObserver func(method string, databaseName string, request proto.Message)

// CallObserverContextKey installs a CallObserver on a Milvus construction
// context. It exists for isolated transport audits such as the live harness.
type CallObserverContextKey struct{}

type callObserverMethodContextKey struct{}

type callObserverStatsHandler struct {
	observer CallObserver
}

func (handler callObserverStatsHandler) TagRPC(
	ctx context.Context,
	info *stats.RPCTagInfo,
) context.Context {
	return context.WithValue(ctx, callObserverMethodContextKey{}, info.FullMethodName)
}

func (handler callObserverStatsHandler) HandleRPC(ctx context.Context, rpcStats stats.RPCStats) {
	payload, ok := rpcStats.(*stats.OutPayload)
	if !ok {
		return
	}
	request, ok := payload.Payload.(proto.Message)
	if !ok {
		return
	}
	method, _ := ctx.Value(callObserverMethodContextKey{}).(string)
	handler.observer(method, transmittedDatabaseName(ctx, request), request)
}

func transmittedDatabaseName(ctx context.Context, request proto.Message) string {
	if named, ok := request.(interface{ GetDbName() string }); ok {
		if databaseName := named.GetDbName(); databaseName != "" {
			return databaseName
		}
	}
	outgoingMetadata, ok := metadata.FromOutgoingContext(ctx)
	if !ok {
		return ""
	}
	return strings.Join(outgoingMetadata.Get("dbname"), ",")
}

func (callObserverStatsHandler) TagConn(
	ctx context.Context,
	_ *stats.ConnTagInfo,
) context.Context {
	return ctx
}

func (callObserverStatsHandler) HandleConn(context.Context, stats.ConnStats) {}

// DefaultCallTimeouts returns the deadline policy a Milvus client uses unless
// its caller tunes the bounds.
func DefaultCallTimeouts() CallTimeouts {
	return CallTimeouts{
		Metadata: defaultMetadataCallTimeout,
		Mutation: defaultMutationCallTimeout,
	}
}

// WithMutation returns the policy with the mutation bound replaced, so an
// operator whose collection is large enough for a valid bulk delete to outlast
// the default raises it from configuration instead of a rebuild.
//
// A non-positive duration keeps the current bound. Configuration that is unset,
// zero, or negative must not remove the bound, because an unbounded mutation is
// the silent hang this policy exists to prevent.
func (timeouts CallTimeouts) WithMutation(mutation time.Duration) CallTimeouts {
	if mutation <= 0 {
		return timeouts
	}
	timeouts.Mutation = mutation
	return timeouts
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
	// context.WithTimeout already implements the shorter-of-two rule this policy
	// needs: it keeps the caller's deadline when that one is earlier and installs
	// the bound otherwise. Deciding between the two here instead would mean
	// comparing an instant against a freshly read clock, and reading the wall
	// clock strips Go's monotonic reading, so a forward wall-clock correction
	// would make a caller deadline look expired and hand a wedged call the full
	// caller budget. The context package compares the two deadlines on the
	// monotonic timer, which no clock correction moves.
	callContext, cancel := context.WithTimeout(ctx, timeout)
	reporter := &deadlineReporter{
		NoopReporter: interceptors.NoopReporter{},
		cancel:       cancel,
		method:       callMeta.FullMethod(),
		timeout:      timeout,
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

// DialOptions returns the keepalive, unary deadline, and optional call observer
// policy for a Milvus client connection.
func DialOptions(
	ctx context.Context,
	logger *slog.Logger,
	timeouts CallTimeouts,
) []grpc.DialOption {
	unaryInterceptors := []grpc.UnaryClientInterceptor{
		milvusDeadlineInterceptor(logger, timeouts),
	}
	dialOptions := []grpc.DialOption{
		grpc.WithKeepaliveParams(keepaliveParameters()),
		grpc.WithChainUnaryInterceptor(unaryInterceptors...),
	}
	observer, _ := ctx.Value(CallObserverContextKey{}).(CallObserver)
	if observer != nil {
		dialOptions = append(dialOptions, grpc.WithStatsHandler(callObserverStatsHandler{observer: observer}))
	}
	return dialOptions
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
