package adapterr

import "google.golang.org/grpc/codes"

// Class is the closed-set classification for an [AdapterError].
type Class string

// Class constants.
const (
	// ClassNotIndexed reports a path that the daemon does not yet
	// track as an indexed codebase.
	ClassNotIndexed Class = "not_indexed"

	// ClassUnknownCodebaseID reports a request argument shaped like a codebase
	// id (the "cb_" prefix) that matches no tracked codebase, distinct from a
	// real path that is simply not indexed yet.
	ClassUnknownCodebaseID Class = "unknown_codebase_id"

	// ClassCollectionMissing reports that the Milvus collection for
	// the codebase does not exist.
	ClassCollectionMissing Class = "collection_missing"

	// ClassCollectionNotReady reports that the Milvus collection
	// exists but is still loading or otherwise unavailable to serve
	// reads or writes.
	ClassCollectionNotReady Class = "collection_not_ready"

	// ClassSearchResultIncomplete reports that a semantic search
	// returned without all requested fields.
	ClassSearchResultIncomplete Class = "search_result_incomplete"

	// ClassMilvusUnavailable reports that the Milvus client is not
	// configured or cannot reach the configured address.
	ClassMilvusUnavailable Class = "milvus_unavailable"

	// ClassEmbedderUnreachable reports that the configured embedding
	// endpoint refused or timed out the request.
	ClassEmbedderUnreachable Class = "embedder_unreachable"

	// ClassEmbedderBusy reports that the embedding endpoint answered but
	// is at capacity (rate limited or temporarily unavailable, HTTP
	// 429/503). The endpoint is reachable; the failure is transient and
	// retryable, distinct from ClassEmbedderUnreachable.
	ClassEmbedderBusy Class = "embedder_busy"

	// ClassEmbedderRejected reports that the embedding endpoint answered
	// with a non-429 HTTP error (for example 400/401/500): reachable but
	// rejecting the request, which usually points at a misconfigured model,
	// dimensions, or credentials rather than a transient condition.
	ClassEmbedderRejected Class = "embedder_rejected"

	// ClassEmbedCancelled reports that the embedding request was cancelled
	// (context cancellation or deadline), for example because the job was
	// cancelled or the daemon is shutting down. Not a fault of the endpoint.
	ClassEmbedCancelled Class = "embed_cancelled"

	// ClassInvalidPath reports a path argument that fails validation
	// (empty, malformed, or otherwise unusable).
	ClassInvalidPath Class = "invalid_path"

	// ClassInvalidArgument reports a required non-path argument that is
	// missing or empty (for example a search query or a job id).
	ClassInvalidArgument Class = "invalid_argument"

	// ClassConflictingJob reports that another job already owns the
	// effective indexing operation for this codebase.
	ClassConflictingJob Class = "conflicting_job"

	// ClassJobNotFound reports a job id that the daemon does not
	// track.
	ClassJobNotFound Class = "job_not_found"

	// ClassInternal is the catch-all class for unknown errors. The
	// message is sanitized at the boundary; the operator finds the
	// real cause in the daemon log by grepping trace_id.
	ClassInternal Class = "internal_error"
)

// CodeFor maps a class to its gRPC status code.
func CodeFor(class Class) codes.Code {
	switch class {
	case ClassNotIndexed, ClassJobNotFound, ClassUnknownCodebaseID:
		return codes.NotFound
	case ClassCollectionMissing, ClassCollectionNotReady, ClassConflictingJob:
		return codes.FailedPrecondition
	case ClassMilvusUnavailable, ClassEmbedderUnreachable:
		return codes.Unavailable
	case ClassEmbedderBusy:
		return codes.ResourceExhausted
	case ClassEmbedCancelled:
		return codes.Canceled
	case ClassInvalidPath, ClassInvalidArgument:
		return codes.InvalidArgument
	case ClassSearchResultIncomplete, ClassEmbedderRejected, ClassInternal:
		return codes.Internal
	default:
		return codes.Unknown
	}
}

// NewNotIndexed reports an operation against a codebase the daemon
// does not yet track.
func NewNotIndexed(path string, cause error) *AdapterError {
	return &AdapterError{
		Class:         ClassNotIndexed,
		Message:       "codebase " + quote(path) + " is not indexed",
		Code:          "not_indexed",
		Hint:          "run the index_codebase tool against this path first",
		Cause:         cause,
		SafeForClient: true,
	}
}

// NewUnknownCodebaseID reports a request argument shaped like a codebase id
// that matches no tracked codebase. The message names the id so the caller can
// see the typo rather than getting a path-not-indexed message for a non-path.
func NewUnknownCodebaseID(id string) *AdapterError {
	return &AdapterError{
		Class:         ClassUnknownCodebaseID,
		Message:       "no codebase with id " + quote(id),
		Code:          "unknown_codebase_id",
		Hint:          "list tracked codebases with list_indexing_statuses to find the current id",
		Cause:         nil,
		SafeForClient: true,
	}
}

// NewEmbedderUnreachable reports a transient failure reaching the configured
// embedding endpoint. It is a shared-infrastructure outage that self-heals when
// the endpoint returns, so it never marks a codebase failed.
func NewEmbedderUnreachable(cause error) *AdapterError {
	return &AdapterError{
		Class:         ClassEmbedderUnreachable,
		Message:       "embedding endpoint is unreachable",
		Code:          "embedder_unreachable",
		Hint:          "verify OPENAI_BASE_URL and that the endpoint serves the OpenAI embeddings API",
		Cause:         cause,
		SafeForClient: true,
	}
}

// NewMilvusUnavailable reports a failure reaching the Milvus vector store: a
// metadata call failed or the connection dropped. It is a shared-infrastructure
// outage that self-heals when the store returns, so it never marks a codebase
// failed. The boot-time "not configured" case uses the ErrUnavailable sentinel,
// which carries the same class.
func NewMilvusUnavailable(cause error) *AdapterError {
	return &AdapterError{
		Class:         ClassMilvusUnavailable,
		Message:       "vector store is unavailable",
		Code:          "milvus_unavailable",
		Hint:          "verify MILVUS_ADDRESS and that the vector store is reachable",
		Cause:         cause,
		SafeForClient: true,
	}
}

// NewEmbedderBusy reports a transient embedding failure where the endpoint
// answered but is at capacity (rate limited or temporarily unavailable). It is
// distinct from NewEmbedderUnreachable so a busy endpoint does not read as down.
func NewEmbedderBusy(cause error) *AdapterError {
	return &AdapterError{
		Class:         ClassEmbedderBusy,
		Message:       "embedding endpoint is at capacity (rate limited)",
		Code:          "embedder_busy",
		Hint:          "the endpoint is busy; this retries automatically and the job can be re-run",
		Cause:         cause,
		SafeForClient: true,
	}
}

// NewEmbedderRejected reports that the embedding endpoint answered with a
// non-429 HTTP error: reachable but rejecting the request, typically a
// misconfigured model, dimensions, or credentials.
func NewEmbedderRejected(cause error) *AdapterError {
	return &AdapterError{
		Class:         ClassEmbedderRejected,
		Message:       "embedding endpoint rejected the request",
		Code:          "embedder_rejected",
		Hint:          "check the embedding model name, dimensions, and credentials",
		Cause:         cause,
		SafeForClient: true,
	}
}

// NewEmbedCancelled reports that the embedding request was cancelled (context
// cancellation or deadline), for example a cancelled job or daemon shutdown.
func NewEmbedCancelled(cause error) *AdapterError {
	return &AdapterError{
		Class:         ClassEmbedCancelled,
		Message:       "embedding request was cancelled",
		Code:          "embed_cancelled",
		Hint:          "re-run index_codebase when ready",
		Cause:         cause,
		SafeForClient: true,
	}
}

// NewIndexDataLost reports that a codebase's Milvus collection is gone, so its
// index data is lost and a search cannot run until a rebuild recreates it. It
// maps to FailedPrecondition through [ClassCollectionMissing], which keeps the
// search path off the catch-all internal class.
func NewIndexDataLost(path string, cause error) *AdapterError {
	return &AdapterError{
		Class:         ClassCollectionMissing,
		Message:       "index data for " + quote(path) + " has been lost (collection not found in Milvus)",
		Code:          "collection_missing",
		Hint:          "wait for background repair or re-run index_codebase to rebuild it",
		Cause:         cause,
		SafeForClient: true,
	}
}

// NewInvalidPath reports a path argument that fails validation.
func NewInvalidPath(message string, cause error) *AdapterError {
	return &AdapterError{
		Class:         ClassInvalidPath,
		Message:       message,
		Code:          "invalid_path",
		Hint:          "pass an absolute path to a directory",
		Cause:         cause,
		SafeForClient: true,
	}
}

// NewMissingArgument reports a required argument that the caller left
// empty or omitted. name is the argument the operation needs, so the
// message points the caller at the exact field to supply.
func NewMissingArgument(name string) *AdapterError {
	return &AdapterError{
		Class:         ClassInvalidArgument,
		Message:       "missing required argument " + quote(name),
		Code:          "invalid_argument",
		Hint:          "supply a non-empty " + name,
		Cause:         nil,
		SafeForClient: true,
	}
}

// NewConflictingJob reports a duplicate indexing request the daemon
// rejects in favor of an in-flight job.
func NewConflictingJob(message string, cause error) *AdapterError {
	return &AdapterError{
		Class:         ClassConflictingJob,
		Message:       message,
		Code:          "conflicting_job",
		Hint:          "wait for the existing job to complete or cancel it before retrying",
		Cause:         cause,
		SafeForClient: true,
	}
}

// NewJobNotFound reports an operation against an unknown job id.
func NewJobNotFound(jobID string) *AdapterError {
	return &AdapterError{
		Class:         ClassJobNotFound,
		Message:       "job " + quote(jobID) + " not found",
		Code:          "job_not_found",
		Hint:          "list jobs with list_indexing_jobs to find the current id",
		Cause:         nil,
		SafeForClient: true,
	}
}

// NewInternal wraps an unknown error. The message is recorded in the
// daemon log; the boundary replaces it with a sanitized envelope.
func NewInternal(message string, cause error) *AdapterError {
	return &AdapterError{
		Class:         ClassInternal,
		Message:       message,
		Code:          "internal_error",
		Hint:          "",
		Cause:         cause,
		SafeForClient: false,
	}
}

func quote(value string) string {
	return "\"" + value + "\""
}
