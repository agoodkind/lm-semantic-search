# Search terms

Search returns source occurrences. Several occurrences can reference one shared
embedding vector without losing their separate source locations.

## Storage

| Term | Meaning |
| --- | --- |
| Source occurrence | One searchable text part at a particular source location. Its identity includes the namespace, owner, and row key. Identical text at two locations remains two occurrences. |
| Owner | The source unit that publishes a generation of occurrences, such as a file or conversation. |
| Namespace | A declared set of occurrences with its own metadata fields and write policy. |
| Embedding input | The exact prepared text sent to the embedding model, including any document prefix. It can differ from the displayed excerpt and lexical search text. |
| Vector | The numeric embedding produced for an input. Its dimension is the number of numeric components. |
| Canonical vector | One live vector record for an embedding identity within a shared pool. Occurrences store references to that record. |
| Embedding identity | The exact prepared input together with the resolved model, model revision, dimension, normalization, and document role. A hash identifies this combination; reuse also checks the saved identity bytes. |
| Vector pool | The backend collection and bound catalog that share canonical vector identities. Separate pools can store the same input independently. |
| Catalog | The SQLite database that stores occurrence identities, source text, metadata, generations, and references to canonical vectors. |
| Vector reuse | Reusing a verified canonical vector for another occurrence with the same embedding identity. Equal numeric vectors from different input identities do not establish reuse. |
| Logical vector reduction | The difference between occurrence count and canonical vector count, compared with storing one vector per occurrence. This count does not measure physical disk savings. |

One hundred occurrences that reference sixty canonical vectors represent forty
fewer logical vector entries, or a 40 percent reduction against that comparison.
Physical storage also includes metadata, indexes, retained backend versions, and
other files. Backend compaction and measured file allocations determine disk
savings.

## Search and pagination

| Term | Meaning |
| --- | --- |
| Eligible occurrence | A committed occurrence that matches the query's namespace and metadata filters. Staged rows are not searchable. |
| Backend scoring batch | A bounded group of eligible vector IDs scored in one backend operation. The search scores further batches until every eligible vector has a score. The batch size is not a result limit. |
| Result page | One requested slice of the ordered occurrence results. Page size controls how many results the caller receives at once. |
| Cursor | A continuation token for a saved result snapshot. Subsequent pages read that snapshot's ordering. |
| Complete traversal | Every result allowed by the filters, score floor, and group quota appears exactly once, and the final page reports no more results. Cursor expiration before completion is a failure. Repeated text at different source locations remains separate results. |
| Repeated result | The same occurrence identity appears more than once during one traversal. Shared vectors and repeated source text do not by themselves constitute repeated results. |
| Result ordinal | A result's zero-based position in the complete ordered snapshot. Its position is separate from its source message index or row key. |
| Score | The numeric value used to rank an occurrence. Dense search uses cosine similarity; hybrid search combines dense and lexical ranks. |
| Score difference | The change in a result's score between compared executions. Even a small difference can exchange adjacent result positions. Equal result membership does not prove equal ordering. |
| Cold search | A measured execution that includes the vector verification or other initialization excluded from its paired warm execution. The experiment must specify which state was reset. |
| Warm search | A measured execution that reuses specified verified or initialized state. Warm and cold results still require the same membership and ordering checks. |

A backend batch of 512 or 2,048 vector IDs does not restrict a query to that many
results. A separate result page size of 100 requires 1,787 pages for 178,543
results. The final page contains 43 results. These values illustrate distinct
counts; they are not a current production corpus size or recommended setting.

A complete comparison checks occurrence identities, scores, ordering, page
continuation, and final exhaustion separately. Latency or lower memory use does
not establish result correctness.
