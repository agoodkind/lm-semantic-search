// Package observation reports operation outcomes without source or vector payloads.
package observation

import (
	"context"
	"errors"
	"os"
	"sync/atomic"
	"time"

	"goodkind.io/lm-semantic-search/internal/clock"
)

// Purpose separates ingestion from query and recovery operations.
type Purpose string

const (
	// Unspecified marks operations without a caller purpose.
	Unspecified Purpose = "unspecified"
	// Ingestion marks source ingestion operations.
	Ingestion Purpose = "ingestion"
	// Query marks public search operations.
	Query Purpose = "query"
	// Recovery marks recovery operations.
	Recovery Purpose = "recovery"
)

// Operation identifies the actual measured boundary.
type Operation string

const (
	// Stage measures public staging completion.
	Stage Operation = "stage"
	// EmbeddingAttempt measures one actual embedding SDK request.
	EmbeddingAttempt Operation = "embedding_sdk_attempt"
	// EmbeddingValidation measures returned vector validation.
	EmbeddingValidation Operation = "embedding_validation"
	// IdentitySelection measures catalog identity reuse decisions.
	IdentitySelection Operation = "identity_selection"
	// WriterAdmission measures catalog writer admission.
	WriterAdmission Operation = "writer_admission"
	// CatalogTransaction measures a real SQL transaction.
	CatalogTransaction Operation = "catalog_transaction"
	// UpsertCall measures one logical Milvus SDK Upsert call.
	UpsertCall Operation = "milvus_sdk_upsert"
	// StrongVerification measures strong backend verification.
	StrongVerification Operation = "strong_verification"
	// VerifiedExactScoring measures one combined strong verification and native scoring operation.
	VerifiedExactScoring Operation = "verified_exact_scoring"
)

// Outcome classifies the completed operation without retaining its error text.
type Outcome string

const (
	// Success marks a completed operation without an error.
	Success Outcome = "success"
	// Cancelled marks an operation interrupted by cancellation.
	Cancelled Outcome = "cancelled"
	// DeadlineExceeded marks an operation interrupted by its deadline.
	DeadlineExceeded Outcome = "deadline_exceeded"
	// Failure marks another operation error.
	Failure Outcome = "failure"
)

// Phase identifies a start marker or its corresponding completion.
type Phase string

const (
	// Started marks operation admission.
	Started Phase = "started"
	// Completed marks an operation outcome.
	Completed Phase = "completed"
)

// Scope associates operations with an explicitly identified caller run.
type Scope struct {
	RunID             string
	Generation        uint64
	ProcessID         int
	OperationID       uint64
	ParentOperationID uint64
	Purpose           Purpose
}

type scopeKey struct{}

// WithScope attaches caller-supplied run metadata to ctx.
func WithScope(ctx context.Context, scope Scope) context.Context {
	return context.WithValue(ctx, scopeKey{}, scope)
}

// WithPurpose changes the operation purpose without replacing the run metadata.
func WithPurpose(ctx context.Context, purpose Purpose) context.Context {
	scope := ScopeFromContext(ctx)
	scope.Purpose = purpose
	return WithScope(ctx, scope)
}

// ScopeFromContext returns the run metadata or an unspecified purpose.
func ScopeFromContext(ctx context.Context) Scope {
	scope, found := ctx.Value(scopeKey{}).(Scope)
	if !found {
		return Scope{Purpose: Unspecified}
	}
	return scope
}

// ValidationBoundary separates adapter shape checks from canonical vector checks.
type ValidationBoundary string

const (
	// AdapterValidation measures returned count and dimension checks.
	AdapterValidation ValidationBoundary = "adapter_shape"
	// CanonicalValidation measures finite canonical vector checks.
	CanonicalValidation ValidationBoundary = "canonical_vector"
)

// EmbeddingData separates requested inputs, returned vectors, and validated vectors.
type EmbeddingData struct {
	Attempt    int
	Requested  int
	Returned   int
	Validated  int
	Validation ValidationBoundary
}

// IdentityData reports decisions from one specific catalog lookup or input pass.
type IdentityData struct {
	DuplicateInputs int
	VerifiedReuse   int
	PendingReuse    int
	Missing         int
	SecondLookup    bool
}

// TransactionBoundary identifies the last attempted SQL operation.
type TransactionBoundary string

const (
	// TransactionBegin marks the actual SQL BeginTx boundary.
	TransactionBegin TransactionBoundary = "begin"
	// TransactionWork marks operations inside the SQL transaction.
	TransactionWork TransactionBoundary = "work"
	// TransactionRollback marks the actual SQL Rollback boundary.
	TransactionRollback TransactionBoundary = "rollback"
	// TransactionCommit marks the actual SQL Commit boundary.
	TransactionCommit TransactionBoundary = "commit"
)

// TransactionData identifies the final SQL boundary and rollback outcome.
type TransactionData struct {
	Boundary       TransactionBoundary
	RollbackFailed bool
}

// VectorData separates requested rows from acknowledged SDK rows.
type VectorData struct {
	Requested    int
	Acknowledged int64
	Verified     int
	// ClientSearchDuration includes the complete SDK call and result decoding.
	ClientSearchDuration time.Duration
	// ClientQueryDuration includes bounded vector Query calls and wire decoding.
	ClientQueryDuration time.Duration
	// LocalVerificationDuration includes local identity, vector, and score checks.
	LocalVerificationDuration time.Duration
}

// StageData distinguishes committed receipt reuse from newly staged rows.
type StageData struct {
	Rows             int
	CommittedReceipt bool
}

// Data contains typed values for the event's operation.
type Data struct {
	Embedding   EmbeddingData
	Identity    IdentityData
	Transaction TransactionData
	Vector      VectorData
	Stage       StageData
}

// Event contains metadata and counts; it contains no source, credentials, or vectors.
type Event struct {
	Scope     Scope
	Operation Operation
	Phase     Phase
	Outcome   Outcome
	StartedAt time.Time
	Duration  time.Duration
	Data      Data
}

// Observer accepts synchronous events and must support concurrent callers.
// Callbacks must return promptly and must not mutate the operation context.
type Observer interface {
	Observe(Event)
}

// Span reports one start and one completion for an observed operation.
type Span struct {
	observer Observer
	event    Event
}

var (
	processID       = os.Getpid()
	nextOperationID atomic.Uint64
)

// Start returns an operation context. A nil observer emits no events.
func Start(ctx context.Context, observer Observer, operation Operation) (context.Context, *Span) {
	span := &Span{observer: observer}
	if observer == nil {
		return ctx, span
	}
	scope := ScopeFromContext(ctx)
	scope.ProcessID = processID
	scope.ParentOperationID = scope.OperationID
	scope.OperationID = nextOperationID.Add(1)
	ctx = WithScope(ctx, scope)
	span.event = Event{Scope: scope, Operation: operation, Phase: Started, StartedAt: clock.Now()}
	observer.Observe(span.event)
	return ctx, span
}

// End reports duration, typed data, and a classified error outcome.
func (span *Span) End(ctx context.Context, err error, data Data) {
	if span.observer == nil {
		return
	}
	event := span.event
	event.Phase = Completed
	event.Duration = clock.Now().Sub(event.StartedAt)
	event.Data = data
	event.Outcome = Success
	switch {
	case err != nil && (errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled)):
		event.Outcome = Cancelled
	case err != nil && (errors.Is(err, context.DeadlineExceeded) || errors.Is(ctx.Err(), context.DeadlineExceeded)):
		event.Outcome = DeadlineExceeded
	case err != nil:
		event.Outcome = Failure
	}
	span.observer.Observe(event)
}
