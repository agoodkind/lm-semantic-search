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
		{"milvus unavailable", &AdapterError{Class: ClassMilvusUnavailable}, true},
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
		{"milvus unavailable", &AdapterError{Class: ClassMilvusUnavailable}, true},
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

// TestNewEmbedInputRejectedKeepsFreeTextOutOfTheClientEnvelope proves the safety
// is a property of the constructor rather than of today's callers. A named
// string type still accepts any converted string, so a future caller could hand
// it a provider name, a model name, an endpoint address, or a credential an
// endpoint echoed back in an error. Anything outside the closed reason set has
// to fall back to the unspecified reason.
func TestNewEmbedInputRejectedKeepsFreeTextOutOfTheClientEnvelope(t *testing.T) {
	t.Parallel()

	// The fixture stands in for a credential without being shaped like one, so
	// the repository's secret scanner does not read the test as a real leak.
	const leak = "OpenAI at https://api.example.invalid rejected key REDACTED-CREDENTIAL-FIXTURE"
	err := NewEmbedInputRejected(EmbedInputRejection{
		Reason:   EmbedRejectionReason(leak),
		Limit:    EmbedLimitTokens,
		Measured: ReportedFigure(10000),
		Maximum:  ReportedFigure(8192),
	}, errors.New("cause stays in the daemon log"))

	if err.Code != string(EmbedRejectionUnspecified) {
		t.Fatalf("code = %q, want the unspecified reason %q", err.Code, EmbedRejectionUnspecified)
	}
	for _, fragment := range []string{leak, "OpenAI", "api.example.invalid", "REDACTED-CREDENTIAL-FIXTURE"} {
		if strings.Contains(err.Message, fragment) {
			t.Fatalf("client message %q carries caller-supplied text %q", err.Message, fragment)
		}
		if strings.Contains(err.Code, fragment) {
			t.Fatalf("client code %q carries caller-supplied text %q", err.Code, fragment)
		}
	}
	if !strings.Contains(SafeMessage(err), string(EmbedRejectionUnspecified)) {
		t.Fatalf("client message %q does not name the fallback reason", SafeMessage(err))
	}

	// Every declared reason survives unchanged, so the guard rejects only what it
	// does not recognize.
	for _, reason := range []EmbedRejectionReason{
		EmbedRejectionContextLengthExceeded,
		EmbedRejectionInputBytesExceeded,
		EmbedRejectionInputContainsNUL,
		EmbedRejectionUnspecified,
	} {
		known := NewEmbedInputRejected(EmbedInputRejection{
			Reason:   reason,
			Limit:    EmbedLimitNone,
			Measured: UnreportedFigure(),
			Maximum:  UnreportedFigure(),
		}, nil)
		if known.Code != string(reason) {
			t.Fatalf("declared reason %q came back as %q", reason, known.Code)
		}
	}
}

// TestEmbedInputRejectionNamesTheLimitThatApplied pins each limit kind to the
// figure a caller can act on: a byte ceiling is never reported as a token
// window, and a refusal that arrived without figures says so instead of quoting
// a zero.
func TestEmbedInputRejectionNamesTheLimitThatApplied(t *testing.T) {
	t.Parallel()

	tokens := NewEmbedInputRejected(EmbedInputRejection{
		Reason:   EmbedRejectionContextLengthExceeded,
		Limit:    EmbedLimitTokens,
		Measured: ReportedFigure(10000),
		Maximum:  ReportedFigure(8192),
	}, nil)
	if !strings.Contains(tokens.Message, "10000 tokens") || !strings.Contains(tokens.Message, "8192-token limit") {
		t.Fatalf("token refusal message = %q", tokens.Message)
	}

	bytesRejection := NewEmbedInputRejected(EmbedInputRejection{
		Reason:   EmbedRejectionInputBytesExceeded,
		Limit:    EmbedLimitBytes,
		Measured: ReportedFigure(32769),
		Maximum:  ReportedFigure(32768),
	}, nil)
	if !strings.Contains(bytesRejection.Message, "32769 bytes") || !strings.Contains(bytesRejection.Message, "32768-byte limit") {
		t.Fatalf("byte refusal message = %q", bytesRejection.Message)
	}
	for _, tokenClaim := range []string{" tokens", "-token limit", "model's limit"} {
		if strings.Contains(bytesRejection.Message, tokenClaim) {
			t.Fatalf("byte refusal message %q reports %q, a token limit that did not apply", bytesRejection.Message, tokenClaim)
		}
	}

	unreported := NewEmbedInputRejected(EmbedInputRejection{
		Reason:   EmbedRejectionContextLengthExceeded,
		Limit:    EmbedLimitUnreported,
		Measured: UnreportedFigure(),
		Maximum:  UnreportedFigure(),
	}, nil)
	if !strings.Contains(unreported.Message, "no size figures") {
		t.Fatalf("unreported refusal message = %q, want it to say the figures are missing", unreported.Message)
	}
	if strings.Contains(unreported.Message, "0") {
		t.Fatalf("unreported refusal message %q quotes a zero figure", unreported.Message)
	}
	if unreported.Hint == "" {
		t.Fatal("a size refusal without figures still needs the shorten-and-retry hint")
	}

	nul := NewEmbedInputRejected(EmbedInputRejection{
		Reason:   EmbedRejectionInputContainsNUL,
		Limit:    EmbedLimitNone,
		Measured: UnreportedFigure(),
		Maximum:  UnreportedFigure(),
	}, nil)
	if nul.Message != "embedding input rejected as "+string(EmbedRejectionInputContainsNUL) {
		t.Fatalf("NUL refusal message = %q, want the reason alone with no limit beside it", nul.Message)
	}
}

// TestEmbedInputRejectionSeparatesAReportedFigureFromAnAbsentOne walks every
// combination of the two size figures a provider may report. A figure the
// provider stated is quoted even when it is zero, a figure it never stated is
// called missing, and a figure no size could take is repeated back as such
// instead of passing for either.
func TestEmbedInputRejectionSeparatesAReportedFigureFromAnAbsentOne(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		measured EmbedFigure
		maximum  EmbedFigure
		want     []string
		reject   []string
	}{
		{
			name:     "a measured count without a maximum",
			measured: ReportedFigure(10000),
			maximum:  UnreportedFigure(),
			want:     []string{"10000 tokens", "the model's limit was not reported"},
			reject:   []string{"no size figures"},
		},
		{
			name:     "a maximum without a measured count",
			measured: UnreportedFigure(),
			maximum:  ReportedFigure(8192),
			want:     []string{"8192 tokens", "measured token count was not reported"},
			reject:   []string{"no size figures"},
		},
		{
			name:     "a measured zero is the measurement, not a missing figure",
			measured: ReportedFigure(0),
			maximum:  ReportedFigure(8192),
			want:     []string{"0 tokens", "8192-token limit"},
			reject:   []string{"not reported", "no size figures"},
		},
		{
			name:     "both figures reported as zero are both quoted",
			measured: ReportedFigure(0),
			maximum:  ReportedFigure(0),
			want:     []string{"0 tokens", "0-token limit"},
			reject:   []string{"not reported", "no size figures"},
		},
		{
			name:     "a maximum reported as zero is quoted beside an absent measurement",
			measured: UnreportedFigure(),
			maximum:  ReportedFigure(0),
			want:     []string{"0 tokens", "measured token count was not reported"},
			reject:   []string{"no size figures"},
		},
		{
			name:     "a negative measurement is repeated back, never called absent",
			measured: ReportedFigure(-5),
			maximum:  ReportedFigure(8192),
			want:     []string{"8192 tokens", "reported as -5", "not a possible size"},
			reject:   []string{"not reported", "no size figures"},
		},
		{
			name:     "a negative maximum is repeated back beside the real measurement",
			measured: ReportedFigure(10000),
			maximum:  ReportedFigure(-1),
			want:     []string{"10000 tokens", "reported as -1", "not a possible size"},
			reject:   []string{"not reported", "no size figures"},
		},
		{
			name:     "two unusable figures are both described rather than dropped",
			measured: ReportedFigure(-5),
			maximum:  UnreportedFigure(),
			want:     []string{"reported as -5", "the model's limit was not reported"},
			reject:   []string{"no size figures"},
		},
		{
			name:     "neither figure reported still says the figures are missing",
			measured: UnreportedFigure(),
			maximum:  UnreportedFigure(),
			want:     []string{"no size figures"},
			reject:   []string{"0"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			rejection := NewEmbedInputRejected(EmbedInputRejection{
				Reason:   EmbedRejectionContextLengthExceeded,
				Limit:    EmbedLimitTokens,
				Measured: test.measured,
				Maximum:  test.maximum,
			}, nil)
			for _, fragment := range test.want {
				if !strings.Contains(rejection.Message, fragment) {
					t.Fatalf("message = %q, want it to carry %q", rejection.Message, fragment)
				}
			}
			for _, fragment := range test.reject {
				if strings.Contains(rejection.Message, fragment) {
					t.Fatalf("message = %q, must not carry %q", rejection.Message, fragment)
				}
			}
			if rejection.Hint == "" {
				t.Fatal("a size refusal still needs the shorten-and-retry hint")
			}
		})
	}
}
