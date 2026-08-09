package milvusgrpc

import (
	"context"
	"encoding/binary"
	"io"
	"log/slog"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/emptypb"
)

type deadlineLogHandler struct {
	records []slog.Record
}

func (handler *deadlineLogHandler) Enabled(context.Context, slog.Level) bool {
	return true
}

func (handler *deadlineLogHandler) Handle(_ context.Context, record slog.Record) error {
	handler.records = append(handler.records, record.Clone())
	return nil
}

func (handler *deadlineLogHandler) WithAttrs([]slog.Attr) slog.Handler {
	return handler
}

func (handler *deadlineLogHandler) WithGroup(string) slog.Handler {
	return handler
}

func TestMilvusDeadlineInterceptorInjectsDeadlineWhenAbsent(t *testing.T) {
	var invoked bool
	invoker := func(
		ctx context.Context,
		_ string,
		_ any,
		_ any,
		_ *grpc.ClientConn,
		_ ...grpc.CallOption,
	) error {
		invoked = true
		deadline, ok := ctx.Deadline()
		if !ok {
			t.Fatal("invoker context has no deadline")
		}
		remaining := time.Until(deadline)
		if remaining <= defaultMetadataCallTimeout-time.Second || remaining > defaultMetadataCallTimeout {
			t.Fatalf("injected deadline remaining = %s, want approximately %s", remaining, defaultMetadataCallTimeout)
		}
		return nil
	}

	interceptor := milvusDeadlineInterceptor(slog.Default(), DefaultCallTimeouts())
	if err := interceptor(context.Background(), "/milvus.test/Call", nil, nil, nil, invoker); err != nil {
		t.Fatalf("interceptor returned error: %v", err)
	}
	if !invoked {
		t.Fatal("interceptor did not invoke the call")
	}
}

// TestMilvusDeadlineInterceptorShortensLongerCallerDeadline pins the upper
// bound the interceptor exists to enforce. A caller that asks for far more time
// than the bound must still lose a wedged call at the bound.
func TestMilvusDeadlineInterceptorShortensLongerCallerDeadline(t *testing.T) {
	callerTimeout := 5 * defaultMetadataCallTimeout
	ctx, cancel := context.WithTimeout(context.Background(), callerTimeout)
	defer cancel()

	var invoked bool
	invoker := func(
		ctx context.Context,
		_ string,
		_ any,
		_ any,
		_ *grpc.ClientConn,
		_ ...grpc.CallOption,
	) error {
		invoked = true
		deadline, ok := ctx.Deadline()
		if !ok {
			t.Fatal("invoker context has no deadline")
		}
		remaining := time.Until(deadline)
		if remaining <= defaultMetadataCallTimeout-time.Second || remaining > defaultMetadataCallTimeout {
			t.Fatalf(
				"deadline remaining = %s, want approximately the %s bound rather than the caller's %s",
				remaining,
				defaultMetadataCallTimeout,
				callerTimeout,
			)
		}
		return nil
	}

	interceptor := milvusDeadlineInterceptor(slog.Default(), DefaultCallTimeouts())
	if err := interceptor(ctx, "/milvus.test/Call", nil, nil, nil, invoker); err != nil {
		t.Fatalf("interceptor returned error: %v", err)
	}
	if !invoked {
		t.Fatal("interceptor did not invoke the call")
	}
}

// deadlineSlack is how much later than the bound a chosen deadline may land
// before the test reads it as the caller's budget rather than the bound. It only
// absorbs the gap between two clock readings, so it stays far below the gap
// between the two budgets under test.
const deadlineSlack = 5 * time.Second

// testControlledClock is the clock these tests control. It takes one reading
// with Go's monotonic timer intact and derives the caller's deadline, the
// stepped wall-clock readings, and the assertions' baseline from it, so a test
// can advance the wall clock by an exact amount with no sleeping and no
// dependence on what the machine's clock does mid-run.
type testControlledClock struct {
	base time.Time
}

func newTestControlledClock() testControlledClock {
	return testControlledClock{base: time.Now()}
}

// deadlineAfter returns a caller deadline one budget past the controlled
// reading. It keeps the monotonic reading, which is what a deadline built by
// context.WithTimeout carries.
func (clock testControlledClock) deadlineAfter(budget time.Duration) time.Time {
	return clock.base.Add(budget)
}

// wallReadingAfter returns the wall-clock reading a process would see once the
// clock has stepped forward by step. Calling UTC strips the monotonic reading,
// which is exactly what internal/clock.Now does, so the result compares on wall
// time the way the removed code's operand did.
func (clock testControlledClock) wallReadingAfter(step time.Duration) time.Time {
	return clock.base.Add(step).UTC()
}

// budgetOf returns how long after the controlled reading an instant falls.
func (clock testControlledClock) budgetOf(instant time.Time) time.Duration {
	return instant.Sub(clock.base)
}

// wallClockDeadlineChoice models the deadline selection this package used to
// perform: pick the caller's deadline when it is already earlier than the bound,
// otherwise install the bound, deciding by measuring the caller's instant
// against a freshly read clock. The reading is a parameter so a test can step
// it; the removed code took it from internal/clock.Now, which returns
// time.Now().UTC() and therefore carries no monotonic reading.
//
// This models the removed code. It is not production code and production does
// not call it. Production performs no clock read at all, so nothing in it can
// observe a stepped clock, and no test can step a clock production never reads.
// TestMilvusCallDeadlineSelectionReadsNoClock in internal/archguard is what
// keeps production off this shape; this function exists to show, deterministically
// and without editing production code, what that shape does after a forward
// wall-clock correction.
func wallClockDeadlineChoice(ctx context.Context, wallClockNow time.Time, timeout time.Duration) time.Time {
	bound := wallClockNow.Add(timeout)
	deadline, hasDeadline := ctx.Deadline()
	if !hasDeadline {
		return bound
	}
	if deadline.After(bound) {
		return bound
	}
	return deadline
}

// TestWallClockDeadlineChoiceEscapesTheBoundAfterAForwardStep steps the clock
// the removed comparison read and shows the defect that motivated replacing it
// with context.WithTimeout. While the clock holds still the comparison is
// correct and the bound wins. After a forward correction larger than the
// caller's remaining budget, the caller's deadline reads as already past the
// bound, the comparison keeps it, and a wedged call inherits the caller's hour.
//
// The step is injected into a model of the removed comparison rather than into
// production, because production reads no clock and Go's public time API cannot
// build a time.Time whose wall and monotonic readings disagree. See
// wallClockDeadlineChoice for why that division is the honest one.
func TestWallClockDeadlineChoiceEscapesTheBoundAfterAForwardStep(t *testing.T) {
	clock := newTestControlledClock()
	const callerBudget = time.Hour
	const forwardStep = 2 * time.Hour
	timeout := DefaultCallTimeouts().Metadata

	ctx, cancel := context.WithDeadline(context.Background(), clock.deadlineAfter(callerBudget))
	defer cancel()

	steady := clock.budgetOf(wallClockDeadlineChoice(ctx, clock.wallReadingAfter(0), timeout))
	if steady > timeout+deadlineSlack {
		t.Fatalf(
			"steady-clock budget = %s, want the %s bound; the removed comparison was correct while the clock held still",
			steady,
			timeout,
		)
	}

	stepped := clock.budgetOf(wallClockDeadlineChoice(ctx, clock.wallReadingAfter(forwardStep), timeout))
	if stepped <= timeout+deadlineSlack {
		t.Fatalf(
			"stepped-clock budget = %s, want the caller's %s; a %s forward wall-clock correction is what made the removed comparison stop bounding the call",
			stepped,
			callerBudget,
			forwardStep,
		)
	}
}

// TestMilvusDeadlineInterceptorBoundsALaterCallerDeadlineUnderTheSameStep runs
// the real interceptor over the same caller deadline and the same elapsed real
// time as the model above, and shows the bound still wins. It cannot step the
// machine's wall clock, so it claims only that the interceptor bounds a later
// caller deadline; the guard test named in wallClockDeadlineChoice is what
// keeps the implementation on a shape a wall-clock step cannot reach.
func TestMilvusDeadlineInterceptorBoundsALaterCallerDeadlineUnderTheSameStep(t *testing.T) {
	clock := newTestControlledClock()
	const callerBudget = time.Hour
	timeouts := DefaultCallTimeouts()

	ctx, cancel := context.WithDeadline(context.Background(), clock.deadlineAfter(callerBudget))
	defer cancel()

	var effectiveDeadline time.Time
	invoker := func(
		ctx context.Context,
		_ string,
		_ any,
		_ any,
		_ *grpc.ClientConn,
		_ ...grpc.CallOption,
	) error {
		deadline, ok := ctx.Deadline()
		if !ok {
			t.Fatal("invoker context has no deadline")
		}
		effectiveDeadline = deadline
		return nil
	}

	interceptor := milvusDeadlineInterceptor(slog.Default(), timeouts)
	if err := interceptor(ctx, "/milvus.test/Call", nil, nil, nil, invoker); err != nil {
		t.Fatalf("interceptor returned error: %v", err)
	}

	effectiveBudget := clock.budgetOf(effectiveDeadline)
	if effectiveBudget > timeouts.Metadata+deadlineSlack || effectiveBudget <= 0 {
		t.Fatalf(
			"effective budget = %s, want the %s bound rather than the caller's %s",
			effectiveBudget,
			timeouts.Metadata,
			callerBudget,
		)
	}
}

func TestMilvusDeadlineInterceptorPreservesEarlierCallerDeadline(t *testing.T) {
	originalDeadline := time.Now().Add(100 * time.Millisecond)
	ctx, cancel := context.WithDeadline(context.Background(), originalDeadline)
	defer cancel()

	invoker := func(
		ctx context.Context,
		_ string,
		_ any,
		_ any,
		_ *grpc.ClientConn,
		_ ...grpc.CallOption,
	) error {
		gotDeadline, ok := ctx.Deadline()
		if !ok {
			t.Fatal("invoker context has no deadline")
		}
		if !gotDeadline.Equal(originalDeadline) {
			t.Fatalf("invoker deadline = %s, want original %s", gotDeadline, originalDeadline)
		}
		return nil
	}

	interceptor := milvusDeadlineInterceptor(slog.Default(), DefaultCallTimeouts())
	if err := interceptor(ctx, "/milvus.test/Call", nil, nil, nil, invoker); err != nil {
		t.Fatalf("interceptor returned error: %v", err)
	}
}

// TestCallTimeoutsForMethod pins the classification against the pinned
// MilvusService method table. The mutating cases are every logical data
// mutation that service declares, so a method whose duration scales with
// matching rows cannot silently fall back to the short metadata bound. The
// remaining cases are the read-only, administrative, credential, and user-tag
// calls that must stay on the short bound, including the two Delete-prefixed
// names an inexact match would sweep in with the row Delete.
func TestCallTimeoutsForMethod(t *testing.T) {
	timeouts := DefaultCallTimeouts()
	cases := []struct {
		method string
		want   time.Duration
	}{
		{method: "Insert", want: timeouts.Mutation},
		{method: "Upsert", want: timeouts.Mutation},
		{method: "Delete", want: timeouts.Mutation},
		{method: "Flush", want: timeouts.Mutation},
		{method: "FlushAll", want: timeouts.Mutation},
		{method: "Import", want: timeouts.Mutation},
		{method: "ReplicateMessage", want: timeouts.Mutation},
		{method: "TruncateCollection", want: timeouts.Mutation},
		{method: "DescribeCollection", want: timeouts.Metadata},
		{method: "Search", want: timeouts.Metadata},
		{method: "Query", want: timeouts.Metadata},
		{method: "DropCollection", want: timeouts.Metadata},
		{method: "DropPartition", want: timeouts.Metadata},
		{method: "DropIndex", want: timeouts.Metadata},
		{method: "DropAlias", want: timeouts.Metadata},
		{method: "DropDatabase", want: timeouts.Metadata},
		{method: "DropResourceGroup", want: timeouts.Metadata},
		{method: "DropRole", want: timeouts.Metadata},
		{method: "DeleteCredential", want: timeouts.Metadata},
		{method: "DeleteUserTags", want: timeouts.Metadata},
	}
	for _, testCase := range cases {
		if got := timeouts.forMethod(testCase.method); got != testCase.want {
			t.Fatalf("forMethod(%q) = %s, want %s", testCase.method, got, testCase.want)
		}
	}
	if timeouts.Mutation <= timeouts.Metadata {
		t.Fatalf("mutation bound %s must exceed the metadata bound %s", timeouts.Mutation, timeouts.Metadata)
	}
}

// TestCallTimeoutsWithMutationIsOperatorTunable pins the tuning path an operator
// uses when a valid bulk delete on a large collection outlasts the built-in
// bound. A non-positive value keeps the built-in bound, because configuration
// that is unset or wrong must never leave a mutation unbounded.
func TestCallTimeoutsWithMutationIsOperatorTunable(t *testing.T) {
	base := DefaultCallTimeouts()
	const configuredMutation = 30 * time.Minute

	tuned := base.WithMutation(configuredMutation)
	if tuned.Mutation != configuredMutation {
		t.Fatalf("tuned mutation bound = %s, want the configured %s", tuned.Mutation, configuredMutation)
	}
	if tuned.Metadata != base.Metadata {
		t.Fatalf("tuning the mutation bound changed the metadata bound to %s, want %s", tuned.Metadata, base.Metadata)
	}

	for _, unusable := range []time.Duration{0, -time.Second} {
		if got := base.WithMutation(unusable).Mutation; got != base.Mutation {
			t.Fatalf("WithMutation(%s) = %s, want the built-in %s bound kept", unusable, got, base.Mutation)
		}
	}
}

func TestMilvusDeadlineInterceptorLogsOnDeadlineExceeded(t *testing.T) {
	handler := &deadlineLogHandler{}
	logger := slog.New(handler)
	wantErr := status.Error(codes.DeadlineExceeded, "test deadline")
	invoker := func(
		context.Context,
		string,
		any,
		any,
		*grpc.ClientConn,
		...grpc.CallOption,
	) error {
		return wantErr
	}

	interceptor := milvusDeadlineInterceptor(logger, DefaultCallTimeouts())
	err := interceptor(context.Background(), "/milvus.test/Call", nil, nil, nil, invoker)
	if err != wantErr {
		t.Fatalf("interceptor error = %v, want %v", err, wantErr)
	}
	if len(handler.records) != 1 {
		t.Fatalf("deadline log records = %d, want 1", len(handler.records))
	}

	record := handler.records[0]
	if record.Level != slog.LevelError {
		t.Fatalf("deadline log level = %s, want ERROR", record.Level)
	}
	attributes := make(map[string]slog.Value)
	record.Attrs(func(attribute slog.Attr) bool {
		attributes[attribute.Key] = attribute.Value
		return true
	})
	if got := attributes["method"].String(); got != "/milvus.test/Call" {
		t.Fatalf("method = %q, want %q", got, "/milvus.test/Call")
	}
	if got := attributes["timeout_ms"].Int64(); got != defaultMetadataCallTimeout.Milliseconds() {
		t.Fatalf("timeout_ms = %d, want %d", got, defaultMetadataCallTimeout.Milliseconds())
	}
	if got := attributes["duration_ms"].Int64(); got < 0 {
		t.Fatalf("duration_ms = %d, want non-negative", got)
	}
	if attributes["err"].Any() == nil {
		t.Fatal("err attribute is nil")
	}
}

func TestDialOptionsReportsEveryUnaryCall(t *testing.T) {
	address := startFakeMilvus(t, time.Millisecond)
	var observedMethod string
	var observedRequest proto.Message
	observerContext := context.WithValue(
		context.Background(),
		CallObserverContextKey{},
		CallObserver(func(method string, _ string, request proto.Message) {
			observedMethod = method
			observedRequest = request
		}),
	)
	options := DialOptions(observerContext, slog.Default(), DefaultCallTimeouts())
	options = append(options, grpc.WithTransportCredentials(insecure.NewCredentials()))
	conn, err := grpc.NewClient("passthrough:///"+address.String(), options...)
	if err != nil {
		t.Fatalf("dial fake Milvus: %v", err)
	}
	t.Cleanup(func() {
		_ = conn.Close()
	})

	if err := invokeFakeMilvus(conn, "DescribeCollection"); err != nil {
		t.Fatalf("invoke fake Milvus: %v", err)
	}
	if observedMethod != "/"+fakeMilvusServiceName+"/DescribeCollection" {
		t.Fatalf("observed method = %q, want DescribeCollection", observedMethod)
	}
	if _, ok := observedRequest.(*emptypb.Empty); !ok {
		t.Fatalf("observed request type = %T, want *emptypb.Empty", observedRequest)
	}
}

func TestDialOptionsReportsTransmittedDatabaseMetadata(t *testing.T) {
	address := startFakeMilvus(t, time.Millisecond)
	observedDatabases := make(chan string, 2)
	observerContext := context.WithValue(
		context.Background(),
		CallObserverContextKey{},
		CallObserver(func(_ string, databaseName string, _ proto.Message) {
			observedDatabases <- databaseName
		}),
	)
	options := DialOptions(observerContext, slog.Default(), DefaultCallTimeouts())
	options = append(options, grpc.WithTransportCredentials(insecure.NewCredentials()))
	conn, err := grpc.NewClient("passthrough:///"+address.String(), options...)
	if err != nil {
		t.Fatalf("dial fake Milvus: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	transmittedContext := metadata.NewOutgoingContext(
		context.Background(),
		metadata.Pairs("dbname", "live_sandbox"),
	)
	if err := invokeFakeMilvusWithContext(transmittedContext, conn, "DescribeCollection"); err != nil {
		t.Fatalf("invoke fake Milvus with database metadata: %v", err)
	}
	if err := invokeFakeMilvus(conn, "DescribeCollection"); err != nil {
		t.Fatalf("invoke fake Milvus without database metadata: %v", err)
	}

	if databaseName := <-observedDatabases; databaseName != "live_sandbox" {
		t.Fatalf("transmitted database = %q, want live_sandbox", databaseName)
	}
	if databaseName := <-observedDatabases; databaseName != "" {
		t.Fatalf("absent transmitted database = %q, want empty", databaseName)
	}
}

// fakeMilvusServiceName mirrors the real Milvus gRPC service name so the
// interceptor classifies the fake server's methods exactly as it classifies the
// deployed ones.
const fakeMilvusServiceName = "milvus.proto.milvus.MilvusService"

// fakeMilvusHandler is the handler contract grpc.ServiceDesc requires. It
// carries only the delay each call spends before it answers.
type fakeMilvusHandler interface {
	callDelay() time.Duration
}

type fakeMilvusServer struct {
	delay time.Duration
}

func (server fakeMilvusServer) callDelay() time.Duration {
	return server.delay
}

func handleDelayedCall(
	srv any,
	ctx context.Context,
	dec func(any) error,
	_ grpc.UnaryServerInterceptor,
) (any, error) {
	request := &emptypb.Empty{}
	if err := dec(request); err != nil {
		return nil, err
	}
	server, ok := srv.(fakeMilvusHandler)
	if !ok {
		return nil, status.Error(codes.Internal, "fake server registered with the wrong handler type")
	}
	timer := time.NewTimer(server.callDelay())
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return nil, status.FromContextError(ctx.Err()).Err()
	case <-timer.C:
		return &emptypb.Empty{}, nil
	}
}

var fakeMilvusServiceDesc = grpc.ServiceDesc{
	ServiceName: fakeMilvusServiceName,
	HandlerType: (*fakeMilvusHandler)(nil),
	Methods: []grpc.MethodDesc{
		{MethodName: "Delete", Handler: handleDelayedCall},
		{MethodName: "TruncateCollection", Handler: handleDelayedCall},
		{MethodName: "DescribeCollection", Handler: handleDelayedCall},
		{MethodName: "DeleteCredential", Handler: handleDelayedCall},
	},
	Metadata: "milvusgrpc_test",
}

// startFakeMilvus serves the fake Milvus methods on a loopback listener. The
// server uses grpc-go's default keepalive enforcement, which is what a
// deployment that never opted into a shorter ping interval runs.
func startFakeMilvus(t *testing.T, delay time.Duration) net.Addr {
	t.Helper()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	server := grpc.NewServer()
	server.RegisterService(&fakeMilvusServiceDesc, fakeMilvusServer{delay: delay})
	go func() {
		_ = server.Serve(listener)
	}()
	t.Cleanup(server.Stop)
	return listener.Addr()
}

func dialFakeMilvus(t *testing.T, address net.Addr, timeouts CallTimeouts) *grpc.ClientConn {
	t.Helper()

	options := DialOptions(context.Background(), slog.Default(), timeouts)
	options = append(options, grpc.WithTransportCredentials(insecure.NewCredentials()))
	conn, err := grpc.NewClient("passthrough:///"+address.String(), options...)
	if err != nil {
		t.Fatalf("dial fake Milvus: %v", err)
	}
	t.Cleanup(func() {
		_ = conn.Close()
	})
	return conn
}

func invokeFakeMilvus(conn *grpc.ClientConn, method string) error {
	return invokeFakeMilvusWithContext(context.Background(), conn, method)
}

func invokeFakeMilvusWithContext(
	ctx context.Context,
	conn *grpc.ClientConn,
	method string,
) error {
	return conn.Invoke(
		ctx,
		"/"+fakeMilvusServiceName+"/"+method,
		&emptypb.Empty{},
		&emptypb.Empty{},
	)
}

// TestDialOptionsApplyMutationTimeoutToSlowMutations drives the real dial
// options against a fake server whose calls outlast the metadata bound. Each
// data mutation must survive on the larger mutation bound while a read-only call
// and a credential call on the same connection still fail fast, which is what
// one flat bound could not do and what a name-prefix classification would get
// wrong for DeleteCredential.
func TestDialOptionsApplyMutationTimeoutToSlowMutations(t *testing.T) {
	const serverDelay = 400 * time.Millisecond
	timeouts := CallTimeouts{
		Metadata: 100 * time.Millisecond,
		Mutation: 5 * time.Second,
	}

	address := startFakeMilvus(t, serverDelay)
	conn := dialFakeMilvus(t, address, timeouts)

	for _, method := range []string{"Delete", "TruncateCollection"} {
		if err := invokeFakeMilvus(conn, method); err != nil {
			t.Fatalf("%s under the %s mutation bound failed: %v", method, timeouts.Mutation, err)
		}
	}

	for _, method := range []string{"DescribeCollection", "DeleteCredential"} {
		err := invokeFakeMilvus(conn, method)
		if status.Code(err) != codes.DeadlineExceeded {
			t.Fatalf(
				"%s error = %v (code %s), want DeadlineExceeded from the %s metadata bound",
				method,
				err,
				status.Code(err),
				timeouts.Metadata,
			)
		}
	}
}

// serverDefaultMinPingInterval is grpc-go's default server-side
// keepalive.EnforcementPolicy MinTime (internal/transport
// defaultKeepalivePolicyMinTime). A client that pings an idle connection, or
// pings more often than this, collects strikes against any server that did not
// opt into a shorter interval.
const serverDefaultMinPingInterval = 5 * time.Minute

// TestKeepaliveParametersRespectDefaultServerEnforcement pins the two knobs a
// server with default enforcement punishes. Wall-clock time cannot prove this
// in a unit test, because a violating policy only earns its disconnect after
// three ping intervals, so the invariant stands in for the wait and
// TestDefaultServerEnforcementDisconnectsIdlePings shows the punishment is real.
func TestKeepaliveParametersRespectDefaultServerEnforcement(t *testing.T) {
	parameters := keepaliveParameters()
	if parameters.PermitWithoutStream {
		t.Fatal(
			"keepalive PermitWithoutStream is true, so the client pings while no call is active " +
				"and a server with default enforcement answers with too_many_pings",
		)
	}
	if parameters.Time < serverDefaultMinPingInterval {
		t.Fatalf(
			"keepalive Time = %s, want at least the default server enforcement minimum of %s",
			parameters.Time,
			serverDefaultMinPingInterval,
		)
	}
	if parameters.Timeout <= 0 {
		t.Fatalf("keepalive Timeout = %s, want a positive ping-ack budget", parameters.Timeout)
	}
}

const (
	http2ClientPreface   = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
	http2FrameHeaderLen  = 9
	http2PingPayloadLen  = 8
	http2GoAwayPrefixLen = 8
	http2SettingsFrame   = 0x4
	http2PingFrame       = 0x6
	http2GoAwayFrame     = 0x7
	http2EnhanceYourCalm = 0xb
	// idlePingCount is how many idle pings it takes to trip a server running
	// default enforcement. The first ping is free because the server has no
	// previous ping timestamp to measure it against; the next three each earn a
	// strike, and grpc-go disconnects once the strikes pass maxPingStrikes (2).
	idlePingCount = 4
)

// writeConnectionFrame writes one connection-level HTTP/2 frame: stream 0, no
// flags. The SETTINGS and PING frames this test sends are both connection
// level.
func writeConnectionFrame(t *testing.T, writer io.Writer, frameType byte, payload []byte) {
	t.Helper()

	frame := make([]byte, http2FrameHeaderLen+len(payload))
	frame[0] = byte(len(payload) >> 16)
	frame[1] = byte(len(payload) >> 8)
	frame[2] = byte(len(payload))
	frame[3] = frameType
	copy(frame[http2FrameHeaderLen:], payload)
	if _, err := writer.Write(frame); err != nil {
		t.Fatalf("write HTTP/2 frame type %d: %v", frameType, err)
	}
}

// readGoAway reads frames until the server sends GOAWAY and returns its error
// code and debug data.
func readGoAway(t *testing.T, reader io.Reader) (uint32, string) {
	t.Helper()

	for {
		header := make([]byte, http2FrameHeaderLen)
		if _, err := io.ReadFull(reader, header); err != nil {
			t.Fatalf("read HTTP/2 frame header: %v", err)
		}
		length := int(header[0])<<16 | int(header[1])<<8 | int(header[2])
		payload := make([]byte, length)
		if _, err := io.ReadFull(reader, payload); err != nil {
			t.Fatalf("read HTTP/2 frame payload: %v", err)
		}
		if header[3] != http2GoAwayFrame {
			continue
		}
		if len(payload) < http2GoAwayPrefixLen {
			t.Fatalf("GOAWAY payload length = %d, want at least %d", len(payload), http2GoAwayPrefixLen)
		}
		errorCode := binary.BigEndian.Uint32(payload[4:http2GoAwayPrefixLen])
		return errorCode, string(payload[http2GoAwayPrefixLen:])
	}
}

// TestDefaultServerEnforcementDisconnectsIdlePings shows what the keepalive
// policy must avoid. The fake Milvus server runs grpc-go's default enforcement,
// which permits no ping at all while no call is active, so three idle pings
// earn an ENHANCE_YOUR_CALM disconnect regardless of how far apart they are:
// the server measures an idle ping against a two-hour window, not against
// EnforcementPolicy.MinTime.
func TestDefaultServerEnforcementDisconnectsIdlePings(t *testing.T) {
	address := startFakeMilvus(t, time.Millisecond)

	conn, err := net.Dial("tcp", address.String())
	if err != nil {
		t.Fatalf("dial raw HTTP/2 connection: %v", err)
	}
	t.Cleanup(func() {
		_ = conn.Close()
	})
	if err := conn.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
		t.Fatalf("set connection deadline: %v", err)
	}

	if _, err := io.WriteString(conn, http2ClientPreface); err != nil {
		t.Fatalf("write HTTP/2 client preface: %v", err)
	}
	writeConnectionFrame(t, conn, http2SettingsFrame, nil)
	for range idlePingCount {
		writeConnectionFrame(t, conn, http2PingFrame, make([]byte, http2PingPayloadLen))
	}

	errorCode, debugData := readGoAway(t, conn)
	if errorCode != http2EnhanceYourCalm {
		t.Fatalf("GOAWAY error code = %d, want ENHANCE_YOUR_CALM (%d)", errorCode, http2EnhanceYourCalm)
	}
	if debugData != "too_many_pings" {
		t.Fatalf("GOAWAY debug data = %q, want %q", debugData, "too_many_pings")
	}
}
