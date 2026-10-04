# In-memory collection store and in-process embedding provider

Package `collection/memory` implements the `collection.Store` interface on the Go heap. Package `embedding/local` builds an embedding provider that runs an ONNX model inside the calling process. A program that uses both packages searches a collection without a Milvus server and without an embedding service.

## Storage

A `memory.Store` writes no file and opens no connection. The process exit deletes every row. A program that needs the rows after a restart upserts them again.

`EnsureCollection` creates a collection with a fixed vector width and the declared scalar columns. A later call adds declared columns that the collection lacks. A column added to a collection with rows must be nullable.

`Upsert` validates every row of a call before it stores any row. A row with another vector width, a missing value for a column that is not nullable, or a value of another type fails the whole call.

## Search

`Search` computes the cosine similarity between the query vector and every row that matches the filter. The result is exact. The store builds no approximate index.

`Search` orders rows by descending score, then ascending `relativePath`, then ascending ID. It then applies the score floor, the per-group cap, and the limit. The Milvus store applies the same order and the same three rules.

The filter tree uses three-valued logic. A comparison on a null value is unknown, and a row matches only when the whole tree is true. A negated equality on a null value therefore rejects the row.

`Search` ignores the `Query` text of the request, because the store runs no BM25 leg.

## Embedding

`local.New` takes a model name and a cache root. The supported models are `embeddinggemma`, which returns 768-wide vectors, and `bge-small`, which returns 384-wide vectors. `local.Describe` returns the width, the token limit, and the query prefix of a model. The caller prepends the query prefix to a search query and stores documents without a prefix.

The first `local.New` call for a model downloads the model and tokenizer files into the cache root and verifies each pinned SHA-256 checksum. This download is the only network use. A later call reads the cached files and sends no request.

The provider refuses an input longer than the model token limit and reports the input as skipped. It does not embed a prefix of the input.

One loaded model serves every provider with the same model and cache root for the life of the process. The process does not release the loaded model.

## Resource use

These figures come from `BenchmarkSearch` in `collection/memory` on an Apple M5 Max with 768-wide vectors, 1,200 bytes of content per row, and no filter. A separate index build used about 6 cores during the run.

| Rows | Search time | Heap per row |
| --- | --- | --- |
| 10,000 | 18 ms | 4.7 KB |
| 100,000 | 188 ms | 4.7 KB |

Search time grows linearly with the row count, because `Search` reads every row on one goroutine. Heap use is 4 bytes per vector component plus the row text and about 400 bytes of bookkeeping per row. One million such rows need about 4.7 GB of heap and about 2 seconds per search.

## Build requirements

`collection/memory` uses the Go standard library only. `embedding/local` uses cgo and links ONNX Runtime and the tokenizers library, which `make build` installs.
