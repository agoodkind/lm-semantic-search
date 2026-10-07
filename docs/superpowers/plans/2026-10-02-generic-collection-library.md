# Generic collection library implementation plan

## Goal

One generic library in lm-semantic-search stores items (text, embedding, typed labels) in Milvus and returns the top matches for a query with label filters. Code search in the LMS daemon and conversation search in Clyde both use the library. Conversation search reads the existing corpus in `conv_chunks_09cfca5e` (5,926 conversations, 3,571,604 rows) without re-embedding. Code search reads the existing `hybrid_code_chunks_*` collections. Every component of the shared search restoration that this plan does not keep is removed.

An in-memory backend behind the same library interface is a follow-up plan.

A dense index fix for `conv_chunks_09cfca5e` (LMS-707) is a second follow-up plan. The 2026-09-18 restore recreated the vector index as `AUTOINDEX`, which Milvus 2.6.18 resolves to HNSW_SQ SQ4U. Measured recall@10 dropped from 0.912 (HNSW float, before the restore) to 0.840. The follow-up builds HNSW float on a test copy, compares it with the baseline queries, and asks the operator before rebuilding the live index.

## Current behavior

Verified on 2026-10-02 between 8:40 PM and 9:30 PM Pacific.

- Clyde search fails. Installed Clyde is `fc0c1f114` (main). It searches only the embedded store copied to `~/.local/state/clyde/library/codebase/catalog.sqlite` (467,322 rows, 229 conversations). Every query copies every eligible row into a temporary SQLite database and fails at the 30 second query timeout or the 1 GiB `MaxTemporaryBytes` limit.
- `~/.config/clyde/config.toml` sets `search_enabled = true` and `ingestion_enabled = false` with the embedded store keys. The previous file is `~/.config/clyde/config.toml.bak-search-restore-20261002`.
- The LMS daemon process is a Codex candidate build (`701794f`) under `/Volumes/Chaos Storage/Codex/LMS-Clyde-shared-search/resume/bin/lms-production-scoring-candidate/`, started by `resume/installed-lms/supervise-installed-lms.rb`. Its gRPC service has no conversation methods.
- `~/Library/LaunchAgents/io.goodkind.lm-semantic-search-daemon.plist` sets `CLAUDE_CONTEXT_CODEBASE_STORE=library` and points state, config, context, socket, and `TMPDIR` at Chaos Storage. `~/.local/state/lm-semantic-search/sockets/lm-semantic-search-daemon.sock` is a symlink to the Codex socket.
- Code search reads `lms_library_codebase` (281,911 rows) in the Codex test Milvus `validation-native-20260930-milvus` on `localhost:39630`, with a 7.6 GB catalog on Chaos Storage.
- The pre-restoration LMS state is intact in `~/.local/state/lm-semantic-search` (last written 2026-09-28 19:32). The pre-restoration data is intact in the main Milvus `milvus-standalone` on port 19530: `conv_chunks_09cfca5e` and the `hybrid_code_chunks_*` collections.
- The last LMS commit that both old Clyde and old code search ran against is `bdf15019`. Clyde `a5baa9154` pins it.

### Origin of the removed work

| Repository | Range | Author trailer | Content |
| --- | --- | --- | --- |
| lm-semantic-search | `a671122b` (#311) | Codex | Shared search library design, including exhaustive ranking |
| lm-semantic-search | 2026-09-28 to 2026-09-29 | Claude Opus 5.5 | `library/` package, exhaustive `Library.Search`, embedded store, `CLAUDE_CONTEXT_CODEBASE_STORE=library` |
| lm-semantic-search | 2026-09-29 to 2026-10-02 | Codex | Library code store, conversation API removal (`9284e20e`), native scoring work |
| clyde | #385 to #413 except the keepers in Task 2 | Codex | Embedded conversation ingestion and search |

## Constraints

- Do not re-embed existing rows. Do not write to `conv_chunks_09cfca5e` or any `hybrid_code_chunks_*` collection until Task 6.
- Keep the Milvus column names of existing collections. Existing rows must stay readable without migration.
- Ranking uses Milvus top-K: dense ANN and BM25 sparse requests fused by RRF (the `bdf15019` behavior). Never score every row per query.
- Add no new tests. Verify each task by running real searches and comparing results before and after. Keep restored tests only when they pass; delete restored tests that fail because their code was removed. Do not port tests with the conversation code.
- Build each repository with `make build`. Sign every commit (`git commit -S`). Ship each task as its own pull request. Deploy only from `main`.
- Every delete in Task 9 requires the operator's explicit confirmation for that item.

## Baseline queries

Record these before Task 3 and reuse them in every later verification.

- Code search: the `search_code` MCP tool with five fixed queries on `/Users/agoodkind/Sites/tack`, `/Users/agoodkind/Sites/clyde-dev/clyde`, and `/Users/agoodkind/Sites/lm-semantic-search`. Save the top 10 result paths and line ranges per query.
- Conversation search: `clyde conversation search --query <q> --limit 10` for five fixed queries, plus one with `--provider codex`, one with a workspace, and one that includes archived conversations. Save the conversation IDs and message indexes.

## Tasks

### 1. Restore lm-semantic-search to bdf15019 plus unrelated fixes

Depends on nothing.

Behavior:
- The tree equals `bdf15019` plus these unrelated commits, applied in order: `ef1857c2` (LMS-715), `26c9a39b` (LMS-716), `40d911a3` (LMS-717), `582d81e3` (LMS-723), `a6b23d19` (gksyntax recipe), `c02703e9` (CBM pin), `27199d42`, `51920df4`, `ea378bf9` (graph shutdown).
- `library/`, `cmd/library-live-gate/`, `test/import/`, `test/live/library_*`, `internal/daemon/library_code_*`, `CLAUDE_CONTEXT_CODEBASE_STORE`, and the 2026-09-27 shared search design and plans are gone.
- The conversation gRPC methods are back.

Steps:
1. `git switch -c restore-target bdf15019`.
2. `git cherry-pick -S` each keeper commit above. Resolve conflicts in favor of the `bdf15019` search code.
3. `git switch -c restore-pre-library origin/main`.
4. `git read-tree -u --reset restore-target` and commit with the subject `Restore lm-semantic-search to bdf15019 with unrelated fixes since then`.
5. Open the pull request. List every reverted range from the origin table in the body.

Verification:
- Run: `make build`.
- Expect: success.
- Run: `git diff restore-target restore-pre-library --stat`.
- Expect: no output.

### 2. Restore Clyde to a5baa9154 plus unrelated fixes

Depends on Task 1 being merged.

Behavior:
- The tree equals `a5baa9154` plus #390 (`1caa961fb`), #391 (`ebf2df824`), #392 (`a4f37c931`), #393 (`a0886ed5a`), and #396 (`50275e644`), each applied with `git cherry-pick -S -m 1`.
- #399 (`dc7bae620`) mixes provider parser fixes with search changes. Keep only its `internal/providers/cursor/parser`, `internal/providers/zed/parser`, and `internal/providers/zed/store` hunks and their test data.
- `go.mod` requires the Task 1 merge commit of lm-semantic-search.

Steps:
1. Build the target tree from `a5baa9154` with the cherry-picks above on a scratch branch.
2. Apply the #399 provider hunks with `git checkout dc7bae620 -- <paths>`. Drop any hunk that imports search packages.
3. On a branch from `origin/main`, run `git read-tree -u --reset` with the target tree and commit with the subject `Restore Clyde to a5baa9154 with unrelated fixes since then`.
4. Update `go.mod` and `go.sum` to the Task 1 merge commit.

Verification:
- Run: `make build`.
- Expect: success.

### 3. Return the installed services to the pre-restoration layout

Depends on Tasks 1 and 2 being merged. This task is the restore checkpoint: conversation search and code search both behave as on 2026-09-28.

Files:
- Modify: `~/Library/LaunchAgents/io.goodkind.lm-semantic-search-daemon.plist`
- Modify: `~/.config/clyde/config.toml`
- Remove: the symlink `~/.local/state/lm-semantic-search/sockets/lm-semantic-search-daemon.sock`

Steps:
1. Record the baseline queries against the running services.
2. Stop the Codex supervisor and its daemon: send `TERM` to the `supervise-installed-lms.rb` process. Confirm that the `lm-semantic-search-daemon` started from Chaos Storage exits.
3. Rewrite the plist: remove `CLAUDE_CONTEXT_CODEBASE_STORE`, and remove the `CLAUDE_CONTEXTD_*_ROOT`, `CLAUDE_CONTEXTD_SOCKET_PATH`, and `TMPDIR` entries that point at Chaos Storage.
4. Remove the socket symlink.
5. Install LMS from the Task 1 merge with `lm-semantic-search install`, then `launchctl kickstart -k gui/$(id -u)/io.goodkind.lm-semantic-search-daemon`.
6. Restore the `[conversation.semantic]` block from `~/.config/clyde/config.toml.bak-search-restore-20261002` with `search_enabled = true` and `ingestion_enabled = false`.
7. Install Clyde from the Task 2 merge with `make deploy`.

Verification:
- Run: the code search baseline queries.
- Expect: results from the `hybrid_code_chunks_*` collections, served by the daemon with `~/.local/state/lm-semantic-search` as its state root.
- Run: the conversation search baseline queries.
- Expect: each returns within 5 seconds with conversations from before 2026-09-28. Save these results as the new conversation baseline.

### 4. Extract the generic collection library in lm-semantic-search

Depends on Task 3.

Files:
- Create: `collection/` (public package `goodkind.io/lm-semantic-search/collection`)
- Create: `collection/milvus/` (Milvus backend)
- Create: `embedding/` (public embedding providers moved from `internal/embedding`)
- Modify: `internal/semantic/collection_search.go`, `collection_filter_expr.go`, `collection.go`, `insert_batch.go`, `chunk_metadata.go`
- Modify: `internal/model/types.go`
- Modify: `internal/daemon/manager_collection_search.go`, `semantic_index.go`, `semantic_backend.go`

Behavior:
- `collection.Declaration` defines the item ID column and typed scalar columns (moved from `model.CollectionDeclaration` and `model.ScalarColumn`).
- `collection.Filter` is the existing filter tree from `semantic/collection_filter_expr.go`.
- `collection.Store` is the backend interface with `Search`, `Upsert`, `Delete`, `Query`, and `EnsureCollection`.
- `collection/milvus.Store.Search` runs the existing `rankCollectionCandidates` hybrid request: a dense ANN request on `vector` and a BM25 request on `sparse_vector`, each limited to `CollectionRankingDepth` with the compiled filter, fused by RRF. It then runs `selectRankedCandidates` and `loadRankedHits`.
- `collection.Hit` returns the ID, content, score, `relativePath`, `startLine`, `endLine`, `fileExtension`, `metadata`, `splitPart`, and a map of declared scalars.
- Rows carry declared scalars as a map. `model.StoredChunk` loses its conversation fields. A generic writer that reads the declaration replaces `conversationScalarColumns`.
- `semantic.Service` loses `conversationCollectionPrefix`, `isConversationCollection`, and the conversation branches in `SearchCollection` (`IsConversationDeclaration`, `ensureConversationScalarColumnsOnce`, `resolveLegacyConversationGroups`). Callers pass the collection name and declaration.
- The daemon code search uses `collection.Store`. Code search results do not change.

Steps:
1. Move the types, the filter compiler, and the Milvus search and load functions into `collection` and `collection/milvus` without behavior changes.
2. Replace the conversation fields of `model.StoredChunk` with a scalar map. Change the insert path to write declared scalars by name.
3. Move `internal/embedding` to the public `embedding` package. Keep `internal/embedding/onnx` internal.
4. Point the daemon at the new packages.

Verification:
- Run: `make build`.
- Expect: success.
- Run: the code search baseline queries on a daemon built from this branch.
- Expect: identical top 10 paths, line ranges, and order.

### 5. Search conversations from Clyde through the collection library

Depends on Task 4 being merged.

Files:
- Create: `clyde/internal/conversation/vectorsearch/` (conversation declaration, filter mapping, collection naming, hit conversion)
- Modify: `clyde/internal/daemon/conversation_semantic_runtime.go`, `search_engine_hits.go`, `conversation_search_source.go`
- Modify: `clyde/internal/config/conversation_config.go`

Behavior:
- `vectorsearch.Declaration()` returns the conversation columns of `conv_chunks_09cfca5e`: `conversationId`, `parentConversationId`, `role`, `provider`, `workspaceRoot`, `archived`, `timestampUnix`, `messageIndex`, `loadRules`.
- `vectorsearch.CollectionName(collectionID)` returns `"conv_chunks_"` plus the first 8 hex characters of MD5 of the trimmed collection ID, matching `internal/semantic/service.go:359` at `bdf15019`.
- Filter mapping follows `bdf15019` `semantic/conversation_filter_expr.go`: providers to `provider in`, workspace roots to `workspaceRoot in`, roles lowercased to `role in`, conversation IDs to `conversationId in`, parent to `parentConversationId ==`, from and until to `timestampUnix >=` and `<`, message index bounds to `messageIndex >=` and `<`, and archived to `archived ==`. `min_score` is a floor after ranking. A per-conversation limit becomes `GroupBy=conversationId`.
- The dense request embeds the query with the NV-EmbedCode query prefix. The BM25 request uses the raw query.
- An in-process type implements the existing `conversationSemanticSearchClient` interface. `engineSearchMatches` and `resolveEngineHits` keep their behavior.
- Config adds `milvus_address`, `milvus_database`, `embedding_base_url`, `embedding_model`, and `embedding_dimension` under `[conversation.semantic]`. Search no longer uses `socket_path`.

Steps:
1. Port the conversation filter, collection naming, and hit conversion from `bdf15019` into `vectorsearch`.
2. Implement the in-process search client over `collection/milvus` and `embedding`.
3. Build the client in `startConversationSemanticRuntime` instead of dialing LMS.

Verification:
- Run: `make build`, then `make deploy`.
- Run: the conversation search baseline queries.
- Expect: the same top 10 conversation IDs and message indexes as the Task 3 baseline.

### 6. Move conversation ingestion into Clyde

Depends on Task 5 being merged.

Files:
- Create: `clyde/internal/conversation/vectorsearch/ingest.go`, `rows.go`, `state.go`
- Modify: `clyde/internal/daemon/conversation_semantic_sync.go`

Behavior:
- Port from `bdf15019` into Clyde: `upsertConversationDocuments` and `conversationDocumentsToStoredChunks` (`manager_conversations.go`), `newConversationStoredChunk` (`manager_conversation_tools.go`), the text and storable rules (`manager_conversation_text.go`, `manager_conversation_storable.go`), manifest and fingerprint handling, delete, backfill (`semantic/conversation_backfill.go`), and stored-row assembly (`localvec/conversation.go`, `semantic/conversation_batch.go`, `semantic/conversation_state.go`).
- Row IDs, `relativePath` formats (`conv/<id>/<message>[/<part>]`, `convtool/<id>/<message>/<tool>[/<part>]`, `convthink/<id>/<message>[/<part>]`), `startLine = 0`, `endLine = 0`, `fileExtension = ""`, and the metadata JSON stay the same, so new rows match existing rows.
- Existing vectors are reused by content hash, as in `bdf15019`.

Verification:
- Run: `make build`, then `make deploy` with `ingestion_enabled = true`.
- Expect: the first pass sends zero embedding requests for conversations unchanged since 2026-09-28.
- Run: `clyde conversation search` for a phrase from a conversation started today.
- Expect: that conversation in the results.

### 7. Remove conversation code from lm-semantic-search

Depends on Task 6 being deployed with ingestion on.

Files:
- Delete: the conversation files that Task 1 restored (`internal/daemon/*conversation*`, `internal/semantic/conversation_*`, `internal/localvec/conversation.go`, `docs/conversationingest`) and their tests.
- Modify: `proto/lmsemanticsearch/v1/service.proto` (remove the conversation methods and messages), `internal/daemon/grpc_server.go`, `internal/localvec/format.go` (drop conversation fields from `row`).

Verification:
- Run: `make build`, then reinstall LMS.
- Run: the code search and conversation search baseline queries.
- Expect: unchanged results.

### 8. Confirm no service uses the Codex test Milvus

Depends on Task 3.

Steps:
1. Confirm that no Clyde or LMS config contains port 39630.

Verification:
- Run: `lsof -nP -iTCP:39630` while running a conversation search and a code search.
- Expect: no connection from `clyde` or `lm-semantic-search-daemon`.

### 9. Delete the shared search restoration artifacts

Depends on Tasks 3 and 8. Ask the operator before each item.

Items:
1. The Codex runtime: the `supervise-installed-lms.rb` supervisor, the `reconstruction-proxy` on `127.0.0.1:5400`, and `~/Library/LaunchAgents/io.goodkind.search-reconstruction-proxy.plist`.
2. The Codex test Milvus stack: containers `validation-native-20260930-milvus`, `-minio`, and `-etcd`, and volumes `validation-native-20260930-milvus-data` (20.57 GB), `-minio-data` (9.44 GB), and `-etcd-data` (147.6 MB).
3. `/Volumes/Chaos Storage/Codex/LMS-Clyde-shared-search/`.
4. The embedded store copy in `~/.local/state/clyde/library/` and `~/.local/state/clyde/conversation-semantic/`.
5. The 13 `conv_chunks_*` collections with 0 to 6 rows in the main Milvus. Keep `conv_chunks_09cfca5e`.
6. Clyde documents from 2026-09-26 and 2026-09-27 under `docs/superpowers/` about the embedded store, if Task 2 kept any.
7. Git branches and worktrees from this work, through the cleanup-git skill.

Verification:
- Run: `docker ps`, `docker volume ls`, and `launchctl list`.
- Expect: no `validation-native-20260930` container or volume and no `search-reconstruction-proxy` job.
