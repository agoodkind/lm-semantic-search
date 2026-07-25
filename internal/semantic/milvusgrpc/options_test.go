package milvusgrpc

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
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

func TestDialOptionsIncludeKeepaliveAndDeadline(t *testing.T) {
	options := DialOptions(slog.Default())
	const wantOptionCount = 2
	if len(options) != wantOptionCount {
		t.Fatalf("DialOptions length = %d, want %d (keepalive + deadline interceptor)", len(options), wantOptionCount)
	}
	for index, option := range options {
		if option == nil {
			t.Fatalf("DialOptions[%d] is nil", index)
		}
	}
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
		if remaining <= defaultMilvusCallTimeout-time.Second || remaining > defaultMilvusCallTimeout {
			t.Fatalf("injected deadline remaining = %s, want approximately %s", remaining, defaultMilvusCallTimeout)
		}
		return nil
	}

	interceptor := milvusDeadlineInterceptor(slog.Default())
	if err := interceptor(context.Background(), "/milvus.test/Call", nil, nil, nil, invoker); err != nil {
		t.Fatalf("interceptor returned error: %v", err)
	}
	if !invoked {
		t.Fatal("interceptor did not invoke the call")
	}
}

func TestMilvusDeadlineInterceptorPreservesExistingDeadline(t *testing.T) {
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

	interceptor := milvusDeadlineInterceptor(slog.Default())
	if err := interceptor(ctx, "/milvus.test/Call", nil, nil, nil, invoker); err != nil {
		t.Fatalf("interceptor returned error: %v", err)
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

	interceptor := milvusDeadlineInterceptor(logger)
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
	if got := attributes["timeout_ms"].Int64(); got != defaultMilvusCallTimeout.Milliseconds() {
		t.Fatalf("timeout_ms = %d, want %d", got, defaultMilvusCallTimeout.Milliseconds())
	}
	if got := attributes["duration_ms"].Int64(); got < 0 {
		t.Fatalf("duration_ms = %d, want non-negative", got)
	}
	if attributes["err"].Any() == nil {
		t.Fatal("err attribute is nil")
	}
}
