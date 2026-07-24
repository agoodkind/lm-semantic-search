# Offline profile

The offline profile runs indexing and search entirely on the local machine, with no Docker, no GPU, and no hosted model server. It replaces the Milvus vector store with an on-disk local index and the hosted embedding model with an in-process ONNX model. A user on a low-spec laptop gets natural-language code search at lower precision than the default profile.

The profile is opt-in and changes nothing for users on the default. An absent or `standard` profile keeps the Milvus store and the hosted embedder.

## Enable it

Run the operator CLI to switch a machine to offline:

```
lm-semantic-search profile offline
```

Select the embedding model with `--model`:

```
lm-semantic-search profile offline --model bge-small
```

Valid model names are `embeddinggemma` (the default) and `bge-small`. The command writes `"profile": "offline"` and the chosen `"offlineEmbeddingModel"` into the daemon `config.json`, preserving every other key. The environment variables `CLAUDE_CONTEXT_PROFILE` and `OFFLINE_EMBEDDING_MODEL` override the file. Restart the daemon after changing the profile, because the daemon reads it at startup.

## Search engine

The store keeps one on-disk vector index per collection under the state directory. Small collections use exact search, so results are the true nearest neighbors. Once a collection grows past a size threshold, the store switches to an approximate index (Hierarchical Navigable Small World, HNSW) that keeps query time low as the corpus grows to millions of vectors, at a small recall cost. This is what lets the offline profile scale to large repositories.

The code graph is unchanged and already offline, so structural search for definitions, callers, and references works with zero external dependencies.

## Model delivery

The default model is `embeddinggemma`, a code-specialized embedding model. The daemon downloads the selected model and tokenizer on first use, verifies a pinned checksum, and caches them under the state directory. After that first fetch, the daemon runs fully offline. The model is not committed to the repository and is not required to install the binary.

The first offline run needs network access to fetch the selected model. Every run after the model is cached is fully offline.

## Limits

Offline search is dense-only. It does not include the default profile's sparse keyword leg or hybrid reranking, so recall on exact-symbol queries is lower.

Above the exact-search threshold, the approximate index returns near-exact results rather than guaranteed exact nearest neighbors. Below the threshold, results are exact.

## Switching back

Offline collections are stored separately from the shared Milvus collections and are not readable by the upstream tool. Switching an offline-indexed codebase back to `standard` needs a forced reindex, because the two profiles do not share an index. This is a deliberate rebuild, not a silent break.
