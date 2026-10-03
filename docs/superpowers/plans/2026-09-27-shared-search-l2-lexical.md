# Rank lexical matches by source occurrence

## Goal

Implement the [design's lexical contract](../specs/2026-09-27-shared-search-library-design.md). L2 can develop after L0 while L1 develops storage. L1 registers the L2 migration at integration. The L2 live gate requires L1's strict harness and L3's public search implementation.

## Current behavior

Current collection search uses Milvus hybrid ranking over stored rows. The new shared vector pool stores one vector per distinct embedding identity. Lexical corpus statistics count source occurrences.

## Constraints

- L2 owns `library/lexical_schema.go` and lexical package files. It does not edit L1's `library/catalog_schema.go` concurrently.
- Analyze `Occurrence.SearchText`. Do not derive lexical terms from transformed embedding input or display source text.
- Compute corpus size, document frequency, and average length for the full active namespace corpus before request filters. Each distinct occurrence contributes to corpus statistics even when its text matches another occurrence.
- The embedded backend uses the library's local analyzer. Milvus `RunAnalyzer` verifies parity; it is not an offline runtime dependency.

## Tasks

### 1. Store indexed lexical terms and corpus statistics

Files:

- Create `library/lexical_schema.go`, `library/lexical_analyzer.go`, and `library/lexical_index.go`.
- Create `test/live/library_lexical_live_test.go`.

Steps:

1. Define `lexical_terms(search_hash, term_hash, tf)` with indexes for term lookup, `lexical_df(namespace, term_hash, df)`, and namespace corpus totals for `N` and total analyzed tokens. Record analyzer identity with lexical content.
2. Store term frequency once per identical `SearchText` and analyzer identity. Maintain occurrence membership separately. Update `N`, document frequencies, and total tokens transactionally with owner publication and deletion.
3. Freeze lexical corpus generation with each query snapshot. Recompute only affected terms during Replace and Delete; do not decode the entire vocabulary map for one change.
4. Test two source occurrences with identical text and one with different text. Assert `N`, term document frequency, length totals, and postings after append, replace, metadata reprojection, and delete. Reprojection must leave lexical statistics unchanged.

### 2. Score repeated query terms

Files:

- Create `library/lexical_rank.go` and extend `test/live/library_lexical_live_test.go` after L3 implements public `Search`.

Steps:

1. Implement the pinned Milvus BM25 arithmetic from the design, including repeated query token multiplicity, hash collisions, float32 rounding points, `k1 > 0`, and `0 <= b <= 1`.
2. Return no lexical ranked rows for an empty analyzed query or zero average document length. Avoid division by zero.
3. Accumulate selected occurrence scores with bounded memory and spill to disk when needed. Resolve matching postings through the indexed term keys.
4. Compare production analyzer terms and score ordering with Milvus `RunAnalyzer` and an independent per-occurrence BM25 oracle on duplicate-heavy input. Validate the configured analyzer hash and numeric parameters.

## Verification

- Compile L2 with L1 after the lexical migration is registered. Run `make library-live-l2` after L3 implements public `Search`; the strict real-dependency runner rejects zero matching tests and skips.
- Run `make test && make check` after schema integration.
- Assert equal BM25 scores and ordering for duplicate occurrences, repeated query terms, empty text, and corpus membership changes. Report numeric parity tolerances observed from the pinned server instead of silently changing the formula.
