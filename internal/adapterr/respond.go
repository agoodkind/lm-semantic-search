package adapterr

import (
	"context"
	"errors"
	"log/slog"

	"goodkind.io/gklog/correlation"
	"google.golang.org/grpc/codes"
)

// Respond classifies err, logs it on the daemon side with the active
// correlation, and returns the gRPC code plus message for the
// boundary's [status.Error] call. A nil err returns ([codes.OK], "").
// Known classes return the safe-for-client message and hint; unknown
// classes return a sanitized message that references the daemon log
// by trace_id and job_id.
func Respond(ctx context.Context, err error) (codes.Code, string) {
	if err == nil {
		return codes.OK, ""
	}
	adapterErr := classify(err)
	logRespond(ctx, adapterErr)
	if adapterErr.SafeForClient {
		return CodeFor(adapterErr.Class), formatKnown(adapterErr, ctx)
	}
	return CodeFor(adapterErr.Class), formatUnknown(ctx)
}

// RespondMCP classifies err, logs it on the daemon side with the
// active correlation, and returns an [*MCPError] the MCP tool layer
// can surface to the client. A nil err returns nil.
func RespondMCP(ctx context.Context, err error) error {
	if err == nil {
		return nil
	}
	adapterErr := classify(err)
	logRespond(ctx, adapterErr)
	corr := correlation.FromContext(ctx)
	message := formatUnknown(ctx)
	if adapterErr.SafeForClient {
		message = formatKnown(adapterErr, ctx)
	}
	return &MCPError{
		Class:   adapterErr.Class,
		Code:    adapterErr.Code,
		Message: message,
		TraceID: string(corr.TraceID),
		JobID:   corr.IdentityAttributeValue("job_id"),
	}
}

func classify(err error) *AdapterError {
	var adapterErr *AdapterError
	if errors.As(err, &adapterErr) {
		return adapterErr
	}
	return NewInternal(err.Error(), err)
}

// SafeMessage returns the client-safe one-line message for err: a known adapter
// class returns its message (plus hint); anything else returns a generic
// message. It carries no wrapped cause or implementation detail, so it is what
// the daemon persists and displays, while the full cause stays in the log.
func SafeMessage(err error) string {
	if err == nil {
		return ""
	}
	adapterErr := classify(err)
	if !adapterErr.SafeForClient {
		return "internal error"
	}
	if adapterErr.Hint != "" {
		return adapterErr.Message + "; " + adapterErr.Hint
	}
	return adapterErr.Message
}

// IsTransient reports whether err is a transient condition that must not be
// persisted as a codebase's terminal failure: an at-capacity embedder or a
// cancellation. The next sync or index attempt resolves it on its own.
func IsTransient(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	adapterErr := classify(err)
	return adapterErr.Class == ClassEmbedderBusy || adapterErr.Class == ClassEmbedCancelled
}

func formatKnown(adapterErr *AdapterError, ctx context.Context) string {
	base := adapterErr.Message
	if adapterErr.Hint != "" {
		base = adapterErr.Message + "; " + adapterErr.Hint
	}
	if refs := formatDiagRefs(ctx); refs != "" {
		return refs + "\n" + base
	}
	return base
}

func formatUnknown(ctx context.Context) string {
	if refs := formatDiagRefs(ctx); refs != "" {
		return refs + "\ninternal error; see daemon logs"
	}
	return "internal error"
}

// formatDiagRefs renders the correlation header from ctx so error messages
// carry a greppable handle to the daemon log. Returns "" when no ids are
// available. The MCP client layer relies on the daemon embedding these in
// the message so it can stay a pure relay.
func formatDiagRefs(ctx context.Context) string {
	corr := correlation.FromContext(ctx)
	return correlation.HeaderLine(corr, "job_id", corr.IdentityAttributeValue("job_id"))
}

func logRespond(ctx context.Context, adapterErr *AdapterError) {
	attrs := []slog.Attr{
		slog.String("class", string(adapterErr.Class)),
		slog.String("code", adapterErr.Code),
		slog.Bool("safe_for_client", adapterErr.SafeForClient),
	}
	if adapterErr.Hint != "" {
		attrs = append(attrs, slog.String("hint", adapterErr.Hint))
	}
	if adapterErr.Message != "" {
		attrs = append(attrs, slog.String("message", adapterErr.Message))
	}
	if adapterErr.Cause != nil {
		attrs = append(attrs, slog.String("cause", adapterErr.Cause.Error()))
	}
	level := slog.LevelWarn
	if adapterErr.Class == ClassInternal {
		level = slog.LevelError
	}
	slog.Default().LogAttrs(ctx, level, "adapter.error.responded", attrs...)
}
