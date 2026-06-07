package adapterr

import (
	"context"
	"errors"
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
)

func TestNewMissingArgumentMapsToInvalidArgument(t *testing.T) {
	t.Parallel()

	err := NewMissingArgument("query")
	if err.Class != ClassInvalidArgument {
		t.Fatalf("class = %q, want %q", err.Class, ClassInvalidArgument)
	}
	if !err.SafeForClient {
		t.Fatal("a missing-argument error should be safe for the client")
	}
	if CodeFor(err.Class) != codes.InvalidArgument {
		t.Fatalf("CodeFor(%q) = %v, want %v", err.Class, CodeFor(err.Class), codes.InvalidArgument)
	}
}

func TestInvalidPathMapsToInvalidArgument(t *testing.T) {
	t.Parallel()

	if got := CodeFor(ClassInvalidPath); got != codes.InvalidArgument {
		t.Fatalf("CodeFor(ClassInvalidPath) = %v, want %v", got, codes.InvalidArgument)
	}
}

func TestNewEmbedderBusyIsDistinctFromUnreachable(t *testing.T) {
	t.Parallel()

	cause := errors.New("429 Too Many Requests")
	err := NewEmbedderBusy(cause)
	if err.Class != ClassEmbedderBusy {
		t.Fatalf("class = %q, want %q", err.Class, ClassEmbedderBusy)
	}
	if err.Code != "embedder_busy" {
		t.Fatalf("code = %q, want embedder_busy", err.Code)
	}
	if !strings.Contains(err.Message, "at capacity") {
		t.Fatalf("message = %q, want it to mention capacity rather than unreachable", err.Message)
	}
	if strings.Contains(err.Message, "unreachable") {
		t.Fatalf("a busy endpoint must not read as unreachable: %q", err.Message)
	}
	if !err.SafeForClient {
		t.Fatal("a busy error should be safe for the client")
	}
	if got := CodeFor(ClassEmbedderBusy); got != codes.ResourceExhausted {
		t.Fatalf("CodeFor(ClassEmbedderBusy) = %v, want %v", got, codes.ResourceExhausted)
	}
	if !errors.Is(err, cause) {
		t.Fatal("NewEmbedderBusy should wrap its cause")
	}
}

func TestEmbedderRejectedAndCancelledClasses(t *testing.T) {
	t.Parallel()

	rejected := NewEmbedderRejected(errors.New("400 bad request"))
	if rejected.Class != ClassEmbedderRejected {
		t.Fatalf("rejected class = %q, want %q", rejected.Class, ClassEmbedderRejected)
	}
	if got := CodeFor(ClassEmbedderRejected); got != codes.Internal {
		t.Fatalf("CodeFor(ClassEmbedderRejected) = %v, want %v", got, codes.Internal)
	}

	cancelled := NewEmbedCancelled(nil)
	if cancelled.Class != ClassEmbedCancelled {
		t.Fatalf("cancelled class = %q, want %q", cancelled.Class, ClassEmbedCancelled)
	}
	if got := CodeFor(ClassEmbedCancelled); got != codes.Canceled {
		t.Fatalf("CodeFor(ClassEmbedCancelled) = %v, want %v", got, codes.Canceled)
	}
}

func TestSafeMessageReturnsCleanMessageNotCause(t *testing.T) {
	t.Parallel()

	cause := errors.New(`POST "http://localhost:5400/v1/embeddings": 429 capacity_exceeded`)
	msg := SafeMessage(NewEmbedderUnreachable(cause))
	if strings.Contains(msg, "429") || strings.Contains(msg, "5400") || strings.Contains(msg, "capacity_exceeded") {
		t.Fatalf("SafeMessage leaked implementation detail: %q", msg)
	}
	if !strings.Contains(msg, "unreachable") {
		t.Fatalf("SafeMessage dropped the class message: %q", msg)
	}
	if got := SafeMessage(errors.New("raw boom")); got != "internal error" {
		t.Fatalf("SafeMessage(non-adapter) = %q, want internal error", got)
	}
	if got := SafeMessage(nil); got != "" {
		t.Fatalf("SafeMessage(nil) = %q, want empty", got)
	}
}

func TestIsTransient(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"busy", NewEmbedderBusy(nil), true},
		{"cancelled class", NewEmbedCancelled(nil), true},
		{"context canceled", context.Canceled, true},
		{"context deadline", context.DeadlineExceeded, true},
		{"unreachable", NewEmbedderUnreachable(nil), true},
		{"milvus unavailable", NewMilvusUnavailable(nil), true},
		{"rejected", NewEmbedderRejected(nil), false},
		{"non-adapter", errors.New("boom"), false},
		{"nil", nil, false},
	}
	for _, testCase := range cases {
		if got := IsTransient(testCase.err); got != testCase.want {
			t.Fatalf("%s: IsTransient = %v, want %v", testCase.name, got, testCase.want)
		}
	}
}

// IsInfraFailure covers the self-healing transient set plus a rejected embedder,
// since a rejected config error is global to the pipeline and never a fault of one
// codebase, even though it is not retryable on its own.
func TestIsInfraFailure(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"busy", NewEmbedderBusy(nil), true},
		{"cancelled class", NewEmbedCancelled(nil), true},
		{"context canceled", context.Canceled, true},
		{"unreachable", NewEmbedderUnreachable(nil), true},
		{"milvus unavailable", NewMilvusUnavailable(nil), true},
		{"rejected", NewEmbedderRejected(nil), true},
		{"internal", NewInternal("boom", nil), false},
		{"not indexed", NewNotIndexed("/x", nil), false},
		{"non-adapter", errors.New("boom"), false},
		{"nil", nil, false},
	}
	for _, testCase := range cases {
		if got := IsInfraFailure(testCase.err); got != testCase.want {
			t.Fatalf("%s: IsInfraFailure = %v, want %v", testCase.name, got, testCase.want)
		}
	}
}
