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

// Code returns the stable class code for err, or empty for nil.
func Code(err error) string {
	if err == nil {
		return ""
	}
	return classify(err).Code
}

// IsTransient reports whether err is a self-healing condition: the next sync or
// index attempt resolves it on its own once the dependency recovers. It marks a
// job retryable and, like every shared-infrastructure failure, must not be
// persisted as a codebase's terminal state. The set is an at-capacity or
// unreachable embedder, an unavailable vector store, and a cancellation.
func IsTransient(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	adapterErr := classify(err)
	switch adapterErr.Class {
	case ClassEmbedderBusy, ClassEmbedCancelled, ClassEmbedderUnreachable, ClassMilvusUnavailable:
		return true
	case ClassNotIndexed, ClassUnknownCodebaseID, ClassCollectionMissing,
		ClassCollectionNotReady, ClassSearchResultIncomplete, ClassEmbedderRejected,
		ClassInvalidPath, ClassInvalidArgument, ClassConflictingJob, ClassJobNotFound,
		ClassIndexBudgetExceeded, ClassInternal:
		return false
	default:
		return false
	}
}

// IsInfraFailure reports whether err is a failure of shared infrastructure (the
// embedding pipeline or the vector store) rather than a fault of one codebase's
// own content. Such a failure affects every codebase the same way, so it never
// marks a codebase failed; it belongs on the job and the daemon health record.
// It is the self-healing transient set plus a rejected embedder, which is a
// global config error that a retry alone will not fix but is still not local to
// any codebase.
func IsInfraFailure(err error) bool {
	if err == nil {
		return false
	}
	if IsTransient(err) {
		return true
	}
	return classify(err).Class == ClassEmbedderRejected
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
