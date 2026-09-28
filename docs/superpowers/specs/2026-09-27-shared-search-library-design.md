# Shared search library design

## Purpose

Clyde and codebase search need one generic Go library that stores one dense vector for each distinct embedding input and exposes every source occurrence through a separate metadata record. Clyde owns conversation policy, source loading, and Milvus connection configuration. Codebase search owns file extraction and its retention policy. The library knows neither source type.

This design replaces the planned generic conversation RPC cutover. The current generic collection RPC and its deployed data remain available during migration. Removing an old collection requires a separate, verified migration or retirement. The new layout does not preserve the historical TypeScript schema.

## Current behavior

[Vector reuse](../../../internal/semantic/reuse_catalog.go) stores a content vector keyed by content hash and model, but [staging](../../../internal/semantic/staging.go) also inserts a vector with each source row. [Collection search](../../../internal/semantic/collection_search.go) filters native scalar columns on those source rows and ranks no more than `CollectionRankingDepth` candidates. The fixed depth of 16,384 has timed out on production data. The earlier paged hybrid implementation returned 2 rows for a requested 10 in the live harness. The current [daemon registry](../../../internal/store/store.go) uses JSON and JSONL files; it has no transactional occurrence database.

## Public contract

Create the importable package `goodkind.io/lm-semantic-search/library`. Its public contract is backend-neutral. `library/milvus.New` accepts a caller-owned `*milvusclient.Client`; `library/embedded.New` adapts the existing offline vector store. Neither adapter closes a caller-owned connection. The package does not start a daemon, open a network listener, load conversation files, or select a source policy.

```go
type StoreDescriptor struct {
    CatalogPath string
    LockPath string
    PoolID string
    EmbeddingModel string
    EmbeddingRevision string
    Dimension int
    Normalization string
}

type Embedder interface {
    EmbedBatch(context.Context, []string) ([][]float32, error)
}
type VectorStore interface {
    PoolIdentity() string
    BindCatalog(context.Context, string) error
    PutCanonical(context.Context, VectorRecord) error
    VerifyStrong(context.Context, []VectorIdentity) error
    ScoreExact(context.Context, []float32, []string) ([]VectorScore, error)
}
type VectorRecord struct { ID, IdentityDigest, Checksum string; Values []float32 }
type VectorIdentity struct { ID, IdentityDigest, Checksum string }
type VectorScore struct { ID string; Score float64 }
type NamespacePolicy uint8 // AppendOnly or ReplaceAllowed
type ScalarType uint8 // String, Bool, or Int64
type ScalarColumn struct {
    Name string
    Type ScalarType
    Nullable bool
    Mutable bool
    MaxLength int
}
type NamespaceSpec struct { ID string; Policy NamespacePolicy; Scalars []ScalarColumn }
type ScalarValue struct {
    Type ScalarType
    Null bool
    String string
    Bool bool
    Int64 int64
}
type FilterOp uint8 // All, Any, Not, Equal, In, Range, Prefix, IsNull, IsPresent
type Filter struct {
    Op FilterOp
    Column string
    Children []Filter
    Values []ScalarValue
    Lower, Upper *ScalarValue
    Prefix string
}
type SearchMode uint8 // Dense or Hybrid

type Config struct {
    Store StoreDescriptor
    Vectors VectorStore
    Embedder Embedder
    MaxBatchRows int
    MaxBatchBytes int64
    QueryBlockSize int
    QueryWorkers int
    MaxTemporaryBytes int64
    SnapshotTTL time.Duration
    MaxSnapshotBytes int64
    QueryTimeout time.Duration
    MaxPageSize int
    MaxQueryBytes int
    MaxFilterDepth int
    MaxFilterValues int
    AnalyzerIdentity string
    QueryInstructionPrefix string
    SearchMode SearchMode // Dense or Hybrid
    BM25K1 float64
    BM25B *float64
    RRFK int
}

func Open(context.Context, Config) (*Library, error)
func (*Library) Close() error
func (*Library) RegisterNamespace(context.Context, NamespaceSpec) error
func (*Library) Apply(context.Context, Batch) (ApplyReceipt, error)
func (*Library) Stage(context.Context, StageBatch) error
func (*Library) CommitGeneration(context.Context, GenerationKey, GenerationSeal) (ApplyReceipt, error)
func (*Library) AbortGeneration(context.Context, GenerationKey) error
func (*Library) ReprojectScalars(context.Context, ScalarProjection) (ProjectionReceipt, error)
func (*Library) Delete(context.Context, []OccurrenceID) error
func (*Library) Search(context.Context, SearchRequest) (SearchPage, error)
func (*Library) GetOwnerState(context.Context, string, string) (OwnerState, error)
func PrepareText(context.Context, PrepareRequest) ([]PreparedPart, error)
```

`Open` validates the catalog path, lock path, `PoolID`, model and revision, dimensions, and normalization. The SQLite catalog saves a UUID and immutable descriptor. `Vectors.BindCatalog` atomically binds that UUID to the backend pool before any write. The Milvus adapter creates the actual vector collection with UUID, canonical path identity, and writer host in its initial schema description, then describes and compares the winner after success, already-exists, or ambiguous failure. It never overwrites or automatically deletes the binding. The embedded adapter uses atomic filesystem creation and reads the winner. A second catalog UUID returns `ErrStoreMismatch`. Every writer uses the same canonical catalog and kernel lock path. The Milvus adapter declares its database and vector collection, while the embedded adapter declares its local root. `Close` releases only library-owned SQLite handles and workers. The caller retains the vector backend and embedder. `VectorStore.ScoreExact` returns one finite score for every requested ID or a typed failure. The Milvus adapter uses FLAT COSINE with strong reads; the embedded adapter scores selected IDs exactly, without approximate HNSW results. `Config` supplies bounded batch and query settings with documented defaults for zero values; nonzero invalid values fail validation. Initial defaults are 256 rows, 8 MiB write bytes, 512 query IDs per block, 2 query workers, 1 GiB temporary disk, 256 MiB snapshot disk, 10-minute snapshot TTL, 30-second query timeout, BM25 `k1=1.2`, `b=0.75`, and RRF `k=60`. An explicit pointer to zero is valid for BM25 `b`. These are resource defaults, not acceptance latency targets. `SearchMode` selects Dense or Hybrid and is fixed per result snapshot. The analyzer identity matches the saved catalog. `QueryInstructionPrefix` applies to query embedding only.

`NamespaceSpec` declares a stable namespace ID, `AppendOnly` or `ReplaceAllowed` policy, and typed scalar columns. A column has a name, string/bool/int64 type, nullability, and maximum string length. Registration is idempotent for an identical declaration and rejects a changed declaration. `AppendOnly` rejects `Batch.Mode=Replace` and `Delete` inside the library. `ScalarValue` is a tagged union with exactly one string, bool, int64, or explicit null value; an absent map entry means absent. `Filter` is a tagged tree with `All`, `Any`, `Not`, `Equal`, `In`, `Range`, literal string `Prefix`, `IsNull`, and `IsPresent`. Each node selects exactly one operator. The prefix is literal text, never an interpolated SQL wildcard. An int64 range includes its lower endpoint and excludes its upper endpoint. A comparison with null evaluates to unknown; `Not` does not convert unknown to true. The library validates every filter and grouping column against the namespace declaration before it reads vectors. A field cannot inject a Milvus expression.

```go
type OccurrenceID struct { Namespace, OwnerID, RowKey string }
type Occurrence struct {
    RowKey string
    SortKey string
    SourceText string
    SearchText string
    EmbeddingInput string
    Scalars map[string]ScalarValue
}
type BatchMode uint8 // Append or Replace
type Batch struct {
    Namespace string
    OwnerID string
    GenerationOrder uint64
    IdempotencyToken string
    Mode BatchMode
    Rows []Occurrence
}
type GenerationKey struct { Namespace, OwnerID string; GenerationOrder uint64; IdempotencyToken string }
type StageBatch struct { Key GenerationKey; Mode BatchMode; Rows []Occurrence }
type GenerationSeal struct { RowCount uint64; ManifestHash string }
type ScalarProjection struct {
    Namespace string
    OwnerID string
    ProjectionOrder uint64
    IdempotencyToken string
    Rows map[string]map[string]ScalarValue // row key -> mutable columns
}
type ProjectionReceipt struct { Namespace, OwnerID string; ProjectionOrder uint64; Fingerprint string }
type SearchRequest struct {
    Namespace string
    Query string
    Filter *Filter
    GroupBy string
    PerGroupLimit int
    MinScore float64
    PageSize int
    Cursor string
}
type SearchHit struct {
    ID OccurrenceID
    SourceText string
    Scalars map[string]ScalarValue
    Score float64
}
type SearchPage struct { Hits []SearchHit; HasMore bool; NextCursor string }
type ApplyReceipt struct { Namespace, OwnerID string; GenerationOrder uint64; Fingerprint string }
type OwnerState struct { GenerationOrder uint64; IdempotencyToken, Fingerprint string }
type PrepareRequest struct {
    Text string
    DocumentPrefix string
    MaxTokens int
    MaxBytes int
    Tokenizer Tokenizer
}
type Tokenizer interface { CountTokens(context.Context, string) (int, error) }
type PreparedPart struct { Suffix string; ByteStart, ByteEnd int; EmbeddingInput string }
```

`PrepareText` splits text at model limits with stable part suffixes and returns the exact transformed document inputs. Clyde and the codebase adapter select source rows and supply their policies before calling this helper. `Apply` and `Stage` accept already prepared parts and never split again. Importable library packages expose the existing embedding provider implementations.

The exported error contract includes `ErrStoreMismatch`, `ErrInvalidRequest`, `ErrAppendConflict`, `ErrStaleGeneration`, `ErrVectorCorrupt`, `ErrVectorMissing`, `ErrCursorExpired`, `ErrCursorMismatch`, `ErrDeadline`, and `ErrResourceLimit`. Each supports `errors.Is`. A failed write preserves the prior committed generation; a failed search returns no successful page or premature end marker.

`library/embedding` exports `OpenAIConfig{BaseURL string, APIKey string, Model string, Dimension int, RequestTimeout time.Duration, MaxAttempts int, BackoffBase time.Duration}` and `NewOpenAI(context.Context, OpenAIConfig) (library.Embedder, error)`. The factory reuses the current OpenAI-compatible provider and its existing four-attempt, 200-millisecond retry defaults. The caller resolves secret environment or file references and passes the credential value without logging it. The public signature does not import `internal/config`. `ONNXConfig{ModelName string, ModelCacheRoot string}` and `NewONNX(context.Context, ONNXConfig) (library.Embedder, error)` use the existing model preset resolver and cached model assets for the offline profile. The current provider returns `BatchResult{Vectors,Skipped}`; the adapter must reject every skipped or nil input with a typed paused, busy, rejected, or cancellation error. It cannot shorten the returned vector slice or silently omit content.

The existing ONNX provider caches native sessions by model path for the process lifetime. `Library.Close` does not release those sessions. Include the cached sessions in offline memory measurements.

`Append` inserts missing occurrence IDs and accepts a byte-identical replay of the same `GenerationOrder` and `IdempotencyToken`. It rejects any rewrite or omission-based deletion. Clyde uses append-only rows. `Replace` publishes one owner's new generation atomically and removes only that owner's previous occurrences. Codebase search uses this mode. `Stage` accepts bounded physical batches under one `GenerationKey`; no staged occurrence is searchable. `CommitGeneration` compares the staged row count and canonical sorted row manifest hash with `GenerationSeal`, verifies every staged vector, and atomically publishes the complete owner generation. A truncated staged file cannot replace a complete file. `AbortGeneration` removes unpublished staging metadata. `Apply` computes the seal and commits one small batch as a convenience. A lower unknown `GenerationOrder` returns `ErrStaleGeneration`; a known committed token returns its original receipt even after later generations. An equal order with a different token or content returns `ErrAppendConflict`. `Delete` removes exact IDs only for a `ReplaceAllowed` namespace. `GetOwnerState` returns the committed generation and fingerprint. `ApplyReceipt` reports the same committed version.

`ReprojectScalars` changes only declared mutable metadata columns for exact row keys of one owner. It appends a metadata event and atomically advances a separate effective projection with a monotonic `ProjectionOrder`. A known token returns its original receipt; an unknown older order fails. The operation changes no source text, embedding input, vector, immutable role, or occurrence ID and makes zero vector writes. Clyde can update current archive or workspace metadata without assuming that an existing message was edited.

## Shared storage

Use one Milvus collection with `vector_id` as a deterministic primary key, dense vector, `identity_digest`, and `vector_checksum` as stored fields. The key hashes the exact transformed `EmbeddingInput` sent to the provider, resolved embedding model and revision, dimension, normalization, and document role prefix. `SourceText` is the returned excerpt. `SearchText` supplies lexical tokens. The model's transformed document input may differ from both. Store equal text blobs once, but keep their three identities explicit. Two occurrences can share a vector even when their displayed text or scalar metadata differs. The same content under a different resolved model or preprocessing identity uses another key. A content digest collision check compares the full identity bytes saved in SQLite before reuse. Reject nonfinite and zero-norm vectors before a COSINE write.

One descriptor binds one resolved model, revision, dimension, and normalization. A model change creates a new pool and catalog generation; the old pool remains readable until replacement passes acceptance. The library never infers historical model identity from matching dimensions and never silently reopens an existing catalog under another model. A candidate imported vector needs verified exact input and model identity before reuse.

Use a canonical SQLite database in WAL mode for metadata and search snapshots. The initial schema is:

- `store_identity(key PRIMARY KEY, value NOT NULL)` records the descriptor and schema version.
- `namespaces(id PRIMARY KEY, declaration NOT NULL)` records immutable typed declarations.
- `vectors(vector_id PRIMARY KEY, input_hash, input_bytes, model_identity, dimension, normalization, state, generation)` records canonical identities and verified visibility.
- `vector_outbox(vector_id PRIMARY KEY, vector_payload, payload_hash, write_generation, state)` stores a durable payload before the Milvus RPC.
- `owners(namespace, owner_id, generation_order, generation_token, fingerprint, PRIMARY KEY(namespace, owner_id))` records committed owner versions.
- `batch_receipts(namespace, owner_id, generation_order, generation_token, batch_hash, fingerprint, PRIMARY KEY(namespace, owner_id, generation_order, generation_token))` preserves replay receipts after newer generations.
- `staged_generations(namespace, owner_id, generation_order, generation_token, mode, state, PRIMARY KEY(namespace, owner_id, generation_order, generation_token))` and `staged_occurrences(namespace, owner_id, generation_order, generation_token, row_key, payload, PRIMARY KEY(namespace, owner_id, generation_order, generation_token, row_key))` accumulate bounded batches before one owner publication.
- `occurrences(namespace, owner_id, row_key, sort_key, vector_id, source_blob_id, search_hash, source_length, generation_order, PRIMARY KEY(namespace, owner_id, row_key))` records visible rows.
- `source_blobs(blob_id PRIMARY KEY, content)` stores immutable excerpts once.
- `projection_events(namespace, owner_id, projection_order, token, payload_hash, payload, PRIMARY KEY(namespace, owner_id, projection_order))` records metadata changes; `effective_scalars` stores the latest declared mutable values by exact occurrence ID.
- `lexical_content(search_hash PRIMARY KEY, analyzer_identity, document_length)`, `lexical_terms(search_hash, term_hash, tf, PRIMARY KEY(search_hash, term_hash))` with an index on `(term_hash, search_hash)`, and `lexical_occurrences(namespace, owner_id, row_key, search_hash, PRIMARY KEY(namespace, owner_id, row_key))` index shared lexical content and its occurrences.
- `lexical_stats(namespace PRIMARY KEY, generation, corpus_size, total_tokens)` and `lexical_df(namespace, term_hash, df, PRIMARY KEY(namespace, term_hash))` record corpus statistics independently for each namespace. One catalog fixes one analyzer identity.
- `occurrence_scalars(namespace, owner_id, row_key, column_name, type, string_value, int64_value, bool_value, is_null, PRIMARY KEY(namespace, owner_id, row_key, column_name))` has separate typed predicate indexes for string, int64, and bool comparisons.
- `search_snapshots(snapshot_id PRIMARY KEY, namespace, request_hash, visibility_revision, projection_revision, rank_config, expires_at)` and `search_results(snapshot_id, ordinal, namespace, owner_id, row_key, source_blob_id, effective_scalars, score, PRIMARY KEY(snapshot_id, ordinal))` preserve stable pages. Unexpired snapshots pin immutable source blobs and vectors against deletion.

Add indexes for occurrence vector ID, namespace plus owner, namespace plus each declared scalar predicate, and snapshot expiration. Store explicit absence separately from null. Do not serialize all scalars into one JSON field and scan every row for a filter. The temporary query database copies only the queried terms' `lexical_df` rows; it does not decode the entire vocabulary.

SQLite transactions publish occurrence rows and corpus statistics only after each referenced vector has been verified in Milvus. A kernel file lock at `LockPath` serializes all writer processes sharing the descriptor. Do not use an expiring lease against unfenced Milvus writes. In a bounded batch, persist the vector identity, exact vector bytes, and complete write payload in the outbox, flush and commit, call a deterministic Milvus upsert, perform a strong read of primary key, identity digest, and vector checksum, then commit vector visibility and occurrence generation in SQLite. A process crash or ambiguous RPC timeout leaves the outbox entry for replay. Replay verifies the existing vector before retrying the same upsert with identical bytes. A late RPC from a dead lock owner also writes identical bytes. A checksum mismatch returns a typed corruption error. The lock covers recovery and publication. SQLite readers use committed snapshots and do not hold the writer lock during search.

Milvus may retain obsolete physical versions after [upsert](https://milvus.io/docs/upsert-entities.md#Upsert-in-override-mode). The contract is one live canonical vector per identity, with measured compacted storage. Do not claim exactly one physical write or exactly-once physical bytes. Orphan vectors are safe after a crash; delete them only after a full, separately verified reference audit covering committed occurrences, staged generations, and unexpired query snapshots. Normal retention never deletes canonical vectors.

## Complete filtered ranking

The occurrence catalog evaluates the namespace's typed filter against committed occurrences and effective metadata, then materializes eligible occurrence IDs and distinct vector IDs within one committed SQLite snapshot. Search must rank all eligible identities. It cannot silently stop at a fixed candidate count, return a short page when more qualifying rows exist, or substitute unfiltered top results. Query embedding applies `QueryInstructionPrefix` before calling the same resolved model; the prefix never changes stored-document identity. Reject nonfinite or zero-norm query vectors. The library performs [standard filtering](https://milvus.io/docs/filtered-search.md#Standard-filtering) over its metadata snapshot before scoring selected vectors.

The exact baseline partitions eligible vector IDs into bounded Milvus requests. Each request performs exhaustive [FLAT](https://milvus.io/docs/index.md#FLAT) COSINE scoring with the request limit equal to the partition cardinality. Verify that every requested ID appears exactly once before accepting a partition. A disk-backed merge combines all partitions. It expands each vector score to every eligible occurrence, preserving repeated content. Rank dense occurrences by raw COSINE score descending, then occurrence ID ascending. Do not normalize the dense score before fusion. Final fused score orders rows descending, then caller-supplied stable `SortKey` ascending, then occurrence ID ascending. Apply a positive `MinScore` floor after fusion, then one group quota across the entire result, then page by ordinal. The first page writes the full ordered hit projection to `search_results`; subsequent pages read that immutable snapshot with an opaque cursor that encodes snapshot ID, next ordinal, request hash, corpus generation, and rank configuration. `HasMore` is exact. A cursor with a different request, missing snapshot, or expired snapshot returns a typed error. It never silently restarts at page one.

The pinned Milvus 2.6.18 client [search iterator](https://github.com/milvus-io/milvus/blob/v2.6.18/client/milvusclient/iterator.go) calls ordinary Search and does not establish a complete hybrid iterator. The exact partition algorithm does not rely on hybrid iteration. The SDK [serializes the initial schema](https://github.com/milvus-io/milvus/blob/v2.6.18/client/milvusclient/collection_options.go) in CreateCollection, and rootcoord [rejects different parameters](https://github.com/milvus-io/milvus/blob/v2.6.18/internal/rootcoord/create_collection_task.go) for an existing collection. A real two-process first-open race test must validate the binding behavior.

Hybrid search requires occurrence-weighted lexical statistics. The analyzer, term frequency, document length, corpus size `N`, document frequency `df`, and average document length count all committed occurrences in the namespace before the request filter. Publish a statistics generation atomically with occurrence publication or deletion. The implementation may maintain postings per unique lexical text, but duplicate occurrences contribute their multiplicity to corpus statistics and ranked occurrences. The offline adapter uses the same versioned local analyzer; Milvus `RunAnalyzer` validates parity, and the analyzer identity is saved in the catalog. The proposed BM25 term formula is `log(1+(N-df+0.5)/(df+0.5)) * tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl))`. [Milvus 2.6.18](https://github.com/milvus-io/milvus/blob/v2.6.18/internal/storage/stats.go) increments query term frequency for repeated analyzed tokens and multiplies it into inverse document frequency; preserve multiplicity, analyzer hash collisions, and float32 rounding after live parity validation. If query analysis returns no terms or average length is zero, omit the lexical leg without dividing by zero. Rank lexical occurrences by score descending, then occurrence ID ascending. Fuse full dense and lexical occurrence ranks with `1/(k+rank)` per present modality, where `k` is positive, versioned, and stored in the search snapshot. The current Milvus RRF default is 60; the implementation documents its selected value. An occurrence without a lexical match contributes zero for that modality. The existing Milvus BM25 field over unique vector rows and SQLite FTS5 BM25 do not establish parity. No optimization can change the exhaustive baseline's result order. Reject a search with a typed deadline or resource error if exhaustive ranking cannot finish; partial success is forbidden.

Persisted snapshots consume bounded disk with an explicit TTL and cleanup. The first page copies eligible occurrence IDs, effective metadata generation, source blob IDs, corpus statistics, rank configuration, and vector checksums under one SQLite read snapshot into a query database, then releases that read transaction. Selected vectors and source blobs remain immutable and are not deleted by normal retention. Sequential strong Milvus reads use the copied IDs and verify each checksum. The first page fixes namespace declaration, filter, model identity, occurrence visibility revision, metadata projection revision, statistics generation, analyzer, and ranking formula. A later write does not change the page sequence. The library applies a maximum page size, query length, filter depth, and membership count and reports typed validation errors. The limits must cover Clyde's full allowed set and must not become a hidden relevance cutoff.

`MaxPageSize`, `MaxQueryBytes`, `MaxFilterDepth`, and `MaxFilterValues` configure request validation. Zero disables that individual limit; negative values fail configuration. `MaxFilterValues` counts scalar values across the complete filter tree. Exceeding an enabled limit returns `ErrInvalidRequest` before scoring. Resource and deadline budgets still apply. Never truncate a request, filter, or result to satisfy a limit.

## Validation and release gates

Implement public-boundary tests with a temporary SQLite catalog, an isolated real Milvus database, and the real embedding adapter and endpoint. Extract the existing `internal/embedding` provider into a reusable importable adapter instead of requiring Clyde to reimplement it. Do not use mocks, stubs, spies, or a fake Milvus service for persistence and search contracts.

Verify exact content reuse across namespaces; different source text with identical embedding input; distinct model/revision identities; concurrent writers in separate processes; crash and restart after each outbox, RPC, strong-read, and SQLite publication boundary; ambiguous RPC timeout; append replay and mutation rejection; replacement and independent retention; null and absent scalars; group saturation; repeated content; complete pages; stable cursor after writes; and no partial page on resource failure. Inspect Milvus live primary keys and compacted storage. Compare dense and lexical ranks to an independent exhaustive oracle on a seeded corpus. A broken duplicate weighting or missing eligible occurrence must fail a public search assertion.

Measure the isolated real-Milvus baseline against a matched healthy recovered corpus and deployed search for the same query set. The exhaustive design is a correctness baseline, not a measured latency claim. Record p50, p95, worst-case latency, peak memory, snapshot disk bytes, Milvus vector count, and post-compaction storage. Include narrow and broad filters, 20,000 or more eligible IDs, duplicate-heavy content, and group saturation. The release gate requires complete requested pages, deterministic ranking, zero missing occurrences, and no timeout in the agreed workload. The performance gate is no regression against the measured healthy baseline for representative queries unless the user accepts a stated threshold. A failed gate prevents cutover; it does not permit a candidate cutoff or a partial page.
