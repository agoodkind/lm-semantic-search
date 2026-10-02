// Package library is the backend-neutral shared search contract. A caller
// declares typed namespaces, writes prepared source occurrences, and searches
// them. The package stores one canonical vector for each distinct embedding
// identity and keeps every source occurrence as a separate metadata record.
//
// The package starts no daemon, opens no network listener, loads no source
// files, and selects no source policy. The caller owns its vector backend
// connection and its embedder.
package library

import (
	"context"
	"time"

	"goodkind.io/lm-semantic-search/library/observation"
)

// StoreDescriptor identifies one immutable vector pool and its catalog. The
// catalog saves the descriptor on first open, and a later open with a different
// descriptor fails with [ErrStoreMismatch].
type StoreDescriptor struct {
	// CatalogPath is the absolute path of the SQLite catalog. Every writer that
	// shares the pool uses the same canonical path.
	CatalogPath string
	// LockPath is the absolute path of the kernel lock file that serializes
	// writers sharing the descriptor.
	LockPath string
	// PoolID identifies the vector pool the catalog binds.
	PoolID string
	// EmbeddingModel is the resolved embedding model identity.
	EmbeddingModel string
	// EmbeddingRevision is the resolved, immutable model revision.
	EmbeddingRevision string
	// Dimension is the vector dimension the model produces.
	Dimension int
	// Normalization identifies the vector normalization the caller applies.
	Normalization string
}

// Embedder turns embedding inputs into dense vectors. It returns exactly one
// vector for every input, in input order, or an error. It never returns a
// shorter slice or a nil vector for an input it refused.
type Embedder interface {
	EmbedBatch(context.Context, []string) ([][]float32, error)
}

// VectorStore is the caller-injected vector backend. The library never closes
// the connection behind it.
type VectorStore interface {
	// PoolIdentity reports the backend location of the pool.
	PoolIdentity() string
	// BindCatalog atomically binds the catalog UUID to the backend pool before
	// any write. A pool already bound to another UUID returns
	// [ErrStoreMismatch].
	BindCatalog(context.Context, string) error
	// PutCanonical writes one canonical vector with deterministic bytes.
	PutCanonical(context.Context, VectorRecord) error
	// VerifyStrong reads each identity with strong consistency and compares its
	// digest and checksum.
	VerifyStrong(context.Context, []VectorIdentity) error
	// ScoreExact returns one finite exact score for every requested ID, in
	// request order, or a typed failure.
	ScoreExact(context.Context, []float32, []string) ([]VectorScore, error)
}

// VerifiedExactScorer optionally verifies canonical identities and vector bytes
// with the same strong read that returns native exact scores. The result contains
// one finite score per identity in request order, or a typed vector failure.
type VerifiedExactScorer interface {
	ScoreExactVerified(context.Context, []float32, []VectorIdentity) ([]VectorScore, error)
}

// ExactScoreLimits optionally declares the backend's bounded score request size.
type ExactScoreLimits interface {
	MaxExactScoreIDs() int
}

// ExactScoreReader verifies and scores canonical vectors for one search.
type ExactScoreReader interface {
	VerifyStrong(context.Context, []VectorIdentity) error
	ScoreExact(context.Context, []float32, []string) ([]VectorScore, error)
}

// ExactScoringSnapshotter optionally creates an immutable reader shared by every
// score block in a search. The reader requires no cursor or close operation.
type ExactScoringSnapshotter interface {
	BeginExactScoring(context.Context) (ExactScoreReader, error)
}

// VectorRecord is one canonical vector write.
type VectorRecord struct {
	ID, IdentityDigest, Checksum string
	Values                       []float32
}

// VectorIdentity identifies one canonical vector for strong verification.
type VectorIdentity struct {
	ID, IdentityDigest, Checksum string
}

// VectorScore is one exact score for a requested vector ID.
type VectorScore struct {
	ID    string
	Score float64
}

// NamespacePolicy selects which writes a namespace accepts. The zero value is
// not a policy. A declaration must choose one.
type NamespacePolicy uint8

const (
	// AppendOnly accepts only appended occurrences. It rejects a batch with
	// mode [Replace] and every Delete.
	AppendOnly NamespacePolicy = iota + 1
	// ReplaceAllowed accepts owner replacement and exact deletion.
	ReplaceAllowed
)

// ScalarType is the declared type of one scalar column. The zero value is not a
// type.
type ScalarType uint8

const (
	// String is a UTF-8 string column with a declared maximum byte length.
	String ScalarType = iota + 1
	// Bool is a boolean column.
	Bool
	// Int64 is a signed 64-bit integer column.
	Int64
)

// ScalarColumn declares one typed scalar column of a namespace.
type ScalarColumn struct {
	Name     string
	Type     ScalarType
	Nullable bool
	// Mutable allows ReprojectScalars to change the column after an occurrence
	// is published.
	Mutable bool
	// MaxLength is the maximum byte length of a [String] value. It is positive
	// for a String column and zero for every other type.
	MaxLength int
}

// NamespaceSpec declares a stable namespace ID, its write policy, and its typed
// scalar columns. Registration accepts an identical declaration again and
// rejects a changed one.
type NamespaceSpec struct {
	ID      string
	Policy  NamespacePolicy
	Scalars []ScalarColumn
}

// ScalarValue is a tagged union. Type selects the field that stores the value,
// and Null marks an explicit null. An absent map entry means the value is
// absent, which is different from null.
type ScalarValue struct {
	Type   ScalarType
	Null   bool
	String string
	Bool   bool
	Int64  int64
}

// FilterOp selects the operator of one [Filter] node. The zero value is not an
// operator.
type FilterOp uint8

const (
	// All matches when every child matches.
	All FilterOp = iota + 1
	// Any matches when at least one child matches.
	Any
	// Not negates its single child. Unknown stays unknown.
	Not
	// Equal matches a column equal to the single value.
	Equal
	// In matches a column equal to any listed value.
	In
	// Range matches a column inside [Lower, Upper). An int64 range includes its
	// lower endpoint and excludes its upper endpoint.
	Range
	// Prefix matches a string column that starts with the literal prefix.
	Prefix
	// IsNull matches a column with an explicit null value.
	IsNull
	// IsPresent matches a column with any value, including null.
	IsPresent
)

// Filter is one node of a typed filter tree. Each node selects exactly one
// operator. A comparison with null evaluates to unknown.
type Filter struct {
	Op           FilterOp
	Column       string
	Children     []Filter
	Values       []ScalarValue
	Lower, Upper *ScalarValue
	Prefix       string
}

// StandardAnalyzer identifies the Milvus 2.6.18 default analyzer: maximal runs
// of alphanumeric characters, lowercased, with no stop words, and CRC-32 IEEE
// token hashes over the first 100 token bytes. A later analyzer change uses a
// new identity string.
const StandardAnalyzer = "milvus-standard-v1"

// SearchMode selects dense or hybrid ranking. The zero value selects the
// default, [Hybrid].
type SearchMode uint8

const (
	// Dense ranks occurrences by exact COSINE score alone.
	Dense SearchMode = iota + 1
	// Hybrid fuses dense and occurrence-weighted BM25 ranks with RRF.
	Hybrid
)

// Config opens a library. A zero budget selects its documented default, and a
// nonzero invalid value fails validation with [ErrInvalidRequest].
type Config struct {
	// Observer receives synchronous operation events. Nil emits no events.
	Observer observation.Observer
	Store    StoreDescriptor
	Vectors  VectorStore
	Embedder Embedder

	// MaxBatchRows bounds the rows of one physical write batch. Default 256.
	MaxBatchRows int
	// MaxBatchBytes bounds the bytes of one physical write batch. Default 8 MiB.
	MaxBatchBytes int64
	// QueryBlockSize bounds the vector IDs of one exact scoring request.
	// Default 512. ExactScoreLimits supplies the bound; other backends use 16,384.
	QueryBlockSize int
	// QueryWorkers bounds concurrent scoring requests. Default 2.
	QueryWorkers int
	// MaxTemporaryBytes bounds the combined query database and verified score
	// stream bytes in the catalog directory. PRAGMA max_page_count reserves the
	// stream budget before admitting each scoring block. Default 1 GiB.
	MaxTemporaryBytes int64
	// SnapshotTTL bounds inactivity between successful cursor pages. Each
	// successful continuation renews the saved expiration. Default 10 minutes.
	SnapshotTTL time.Duration
	// MaxSnapshotBytes bounds the logical result bytes of unexpired search
	// snapshots, measured after expired snapshots are deleted: the lengths of
	// the text columns plus 8 bytes per numeric column. It does not count index
	// pages or free pages. Default 256 MiB.
	MaxSnapshotBytes int64
	// QueryTimeout bounds one search. Default 30 seconds.
	QueryTimeout time.Duration

	// MaxPageSize, MaxQueryBytes, MaxFilterDepth, and MaxFilterValues limit a
	// search request. Zero disables the individual limit. MaxFilterValues
	// counts scalar values across the complete filter tree.
	MaxPageSize     int
	MaxQueryBytes   int
	MaxFilterDepth  int
	MaxFilterValues int

	// AnalyzerIdentity identifies the lexical analyzer. Zero selects
	// [StandardAnalyzer], the only identity this build supports. It must match
	// the saved catalog.
	AnalyzerIdentity string
	// QueryInstructionPrefix is prepended to a query before query embedding
	// only. It never changes stored document identity.
	QueryInstructionPrefix string
	// SearchMode is fixed for each result snapshot.
	SearchMode SearchMode
	// BM25K1 is the BM25 term saturation. Default 1.2.
	BM25K1 float64
	// BM25B is the BM25 length normalization. Nil selects 0.75, and a pointer
	// to zero is valid.
	BM25B *float64
	// RRFK is the reciprocal rank fusion constant. Default 60.
	RRFK int
}

// OccurrenceID identifies one source occurrence.
type OccurrenceID struct {
	Namespace, OwnerID, RowKey string
}

// Occurrence is one prepared source row. SourceText is the returned excerpt,
// SearchText supplies lexical terms, and EmbeddingInput is the exact
// transformed document input sent to the embedder. The three identities stay
// explicit even when their bytes are equal.
type Occurrence struct {
	RowKey         string
	SortKey        string
	SourceText     string
	SearchText     string
	EmbeddingInput string
	Scalars        map[string]ScalarValue
}

// BatchMode selects append or owner replacement. The zero value is not a mode.
type BatchMode uint8

const (
	// Append inserts missing occurrence IDs and accepts a byte-identical replay.
	Append BatchMode = iota + 1
	// Replace publishes one owner's new generation and removes only that
	// owner's previous occurrences.
	Replace
)

// Batch writes one small owner generation in one call.
type Batch struct {
	Namespace        string
	OwnerID          string
	GenerationOrder  uint64
	IdempotencyToken string
	Mode             BatchMode
	Rows             []Occurrence
}

// GenerationKey identifies one staged owner generation.
type GenerationKey struct {
	Namespace, OwnerID string
	GenerationOrder    uint64
	IdempotencyToken   string
}

// StageBatch adds one bounded physical batch to a staged generation. No staged
// occurrence is searchable before its generation commits.
type StageBatch struct {
	Key  GenerationKey
	Mode BatchMode
	Rows []Occurrence
}

// GenerationSeal states the complete staged generation: its row count and its
// canonical sorted row manifest hash.
type GenerationSeal struct {
	RowCount     uint64
	ManifestHash string
}

// ScalarProjection changes declared mutable columns for exact row keys of one
// owner.
type ScalarProjection struct {
	Namespace        string
	OwnerID          string
	ProjectionOrder  uint64
	IdempotencyToken string
	// Rows maps a row key to its mutable column values.
	Rows map[string]map[string]ScalarValue
}

// ProjectionReceipt reports one committed scalar projection.
type ProjectionReceipt struct {
	Namespace, OwnerID string
	ProjectionOrder    uint64
	Fingerprint        string
}

// SearchRequest asks for one page of ranked occurrences.
type SearchRequest struct {
	Namespace     string
	Query         string
	Filter        *Filter
	GroupBy       string
	PerGroupLimit int
	MinScore      float64
	PageSize      int
	Cursor        string
}

// SearchHit is one ranked occurrence.
type SearchHit struct {
	ID         OccurrenceID
	SourceText string
	Scalars    map[string]ScalarValue
	Score      float64
}

// SearchPage is one page of a persisted result snapshot. HasMore is exact.
type SearchPage struct {
	Hits       []SearchHit
	HasMore    bool
	NextCursor string
}

// ApplyReceipt reports one committed owner generation.
type ApplyReceipt struct {
	Namespace, OwnerID string
	GenerationOrder    uint64
	Fingerprint        string
}

// OwnerState reports an owner's committed generation.
type OwnerState struct {
	GenerationOrder               uint64
	IdempotencyToken, Fingerprint string
}

// OwnerOccurrences lists one owner's published occurrences. State is the
// committed generation, ProjectionOrder is the latest saved [ScalarProjection]
// order or zero, and Rows are sorted by row key.
type OwnerOccurrences struct {
	State           OwnerState
	ProjectionOrder uint64
	Rows            []OwnerOccurrence
}

// OwnerOccurrence is one published row key and the generation order that
// published it.
type OwnerOccurrence struct {
	RowKey          string
	GenerationOrder uint64
}

// PrepareRequest asks [PrepareText] to split one selected source text at the
// embedding model's limits.
type PrepareRequest struct {
	// Text is the selected source text. It must be valid UTF-8 and contain at
	// least one non-whitespace character.
	Text string
	// DocumentPrefix is the model's document role prefix. It starts every
	// part's embedding input and counts against both limits.
	DocumentPrefix string
	// MaxTokens is the model's input token limit for one embedding input.
	MaxTokens int
	// MaxBytes is the byte limit for one embedding input. Zero applies no byte
	// limit of its own.
	MaxBytes int
	// Tokenizer counts the tokens the embedding model measures. Nil limits each
	// embedding input to 90 percent of MaxTokens bytes, one byte per token.
	Tokenizer Tokenizer
}

// Tokenizer counts the tokens the embedding model measures for one input,
// including any special tokens the model adds.
type Tokenizer interface {
	CountTokens(context.Context, string) (int, error)
}

// PreparedPart is one model-sized part of a prepared text. ByteStart and
// ByteEnd locate the part in [PrepareRequest.Text], and EmbeddingInput is the
// exact transformed document input for the embedder.
type PreparedPart struct {
	Suffix             string
	ByteStart, ByteEnd int
	EmbeddingInput     string
}
