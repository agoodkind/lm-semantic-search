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
	"google.golang.org/grpc/status"
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

func TestCallTimeoutsForMethod(t *testing.T) {
	timeouts := DefaultCallTimeouts()
	cases := []struct {
		method string
		want   time.Duration
	}{
		{method: "Delete", want: timeouts.Mutation},
		{method: "Insert", want: timeouts.Mutation},
		{method: "Upsert", want: timeouts.Mutation},
		{method: "Flush", want: timeouts.Mutation},
		{method: "DescribeCollection", want: timeouts.Metadata},
		{method: "Search", want: timeouts.Metadata},
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
		{MethodName: "DescribeCollection", Handler: handleDelayedCall},
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

	options := DialOptions(slog.Default(), timeouts)
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
	return conn.Invoke(
		context.Background(),
		"/"+fakeMilvusServiceName+"/"+method,
		&emptypb.Empty{},
		&emptypb.Empty{},
	)
}

// TestDialOptionsApplyMutationTimeoutToSlowDelete drives the real dial options
// against a fake server whose calls outlast the metadata bound. Delete must
// survive on the larger mutation bound while an ordinary metadata call on the
// same connection still fails fast, which is what one flat bound could not do.
func TestDialOptionsApplyMutationTimeoutToSlowDelete(t *testing.T) {
	const serverDelay = 400 * time.Millisecond
	timeouts := CallTimeouts{
		Metadata: 100 * time.Millisecond,
		Mutation: 5 * time.Second,
	}

	address := startFakeMilvus(t, serverDelay)
	conn := dialFakeMilvus(t, address, timeouts)

	if err := invokeFakeMilvus(conn, "Delete"); err != nil {
		t.Fatalf("Delete under the %s mutation bound failed: %v", timeouts.Mutation, err)
	}

	err := invokeFakeMilvus(conn, "DescribeCollection")
	if status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf(
			"DescribeCollection error = %v (code %s), want DeadlineExceeded from the %s metadata bound",
			err,
			status.Code(err),
			timeouts.Metadata,
		)
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
