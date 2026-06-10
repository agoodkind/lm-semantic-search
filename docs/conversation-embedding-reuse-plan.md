# Plan: stop re-embedding conversation chunks that already exist

## What the system does today

The daemon keeps a semantic index of every Claude and Codex conversation in one
collection, `conv_chunks_09cfca5e` (codebase id `cb_1780899687_8497ffa5700d`).
clyde sends a manifest of every conversation with a content fingerprint, the
engine replies with the conversation ids whose content changed, and clyde sends
the documents for those ids. The engine embeds them with a 7B model on the GPU.

The GPU stays busy almost continuously. A conversation embed job runs every five
to seven minutes, and the embedder keeps producing vectors for text that is
already stored in the collection.

## Evidence

All figures below are measured from the live state under
`~/.local/state/lm-semantic-search` and the source at the cited file and line.

Manifest and backlog (from `clyde-daemon.jsonl` feeder lines and
`registry.json`):

- clyde's manifest lists about 1294 conversations.
- About 1208 are reported needed on every pass (795 Codex, 413 Claude).
- 89 are checkpointed in the merkle snapshot, whose stored config digest matches
  the registry digest, so the checkpoint loads correctly.
- The needed count falls about 16 per hour, while new active conversations are
  added about 8 per hour, so the net drain is roughly 7 to 8 per hour. At that
  rate the one-time backlog takes three to six days to clear.
- 88 percent of upsert attempts are rejected with `conflicting active job`,
  because a job is almost always already running.

Per-job embed volume (from `daemon.jsonl` chunk-cache deltas and
`semantic.reindex` spans):

- A single changed conversation regularly embeds 500 to 3647 chunks in one job.
- The largest single-conversation job ran 22.5 minutes, made 114 insert
  batches, and embedded 3647 chunks, all from one conversation.
- Job duration tracks the size of the conversation embedded, not the count of
  conversations. Jobs that embed one large conversation run 600 to 1352
  seconds. Jobs that embed two to five small conversations finish in 10 to 48
  seconds.

Correction to one earlier reading: the `chunks_generated` value of about 44000
that appears on every job is `semantic.Count` of the whole collection, set in
`codebaseTotals` (`internal/daemon/manager_delta.go:262`). It is the collection
size, not the number of chunks embedded in that run.

## Root cause

The cause has three layers. The first is the dominant cost.

### Layer 1: a changed conversation re-embeds in full

When a conversation changes, the engine deletes every stored chunk for that
conversation and embeds every chunk again. There is no message-level delta.

- `conversationItemSource.indexOne` turns every delivered document for the
  conversation into chunks (`internal/daemon/item_source.go:98-117`).
- `conversationDocumentsToStoredChunks` produces at least one chunk per message
  (`internal/daemon/manager_conversations.go:536-562`).
- `removalFor` returns a prefix delete of `conv/<id>/`, which drops all prior
  rows for the conversation (`internal/daemon/item_source.go:119-125`).
- `handleChangedFile` issues that prefix delete and embeds the full new chunk
  set in one reindex call (`internal/daemon/manager_delta.go:510-548`).

A conversation with 1000 stored messages that gains one message deletes 1000
chunks and embeds 1001. The active conversation grows on every pass, so it is
re-embedded in full every few minutes.

### Layer 2: conversation ingests never reuse stored vectors

The engine already has a vector reuse mechanism keyed by content hash. The code
build path loads existing vectors and skips the embedder for any chunk whose
content hash is unchanged. Conversation ingests never use it.

- `runDeltaSync` sets `reuse: nil` unconditionally
  (`internal/daemon/manager_delta.go:221`).
- `runBootstrap` loads reuse vectors only from absorbed child codebases and
  sibling worktrees (`internal/daemon/manager_delta.go:317-329`). A conversation
  collection has neither, so its reuse map stays empty.
- `LoadSnapshotForConfig` returns an empty snapshot when the stored config
  digest does not match, which classifies every conversation as added and forces
  a full re-embed (`internal/merkle/snapshot.go:240-258`).

The collection already holds about 44000 vectors for this corpus. The backlog
work is re-embedding text whose vector already exists in the same collection.

### Layer 3: the job writes the full checkpoint once per diff item

The per-item loop writes the entire merkle snapshot to disk and emits a progress
update for every item in the diff, including items that were skipped because no
documents were delivered for them.

- `applyDeltaChanges` calls `writeCheckpoint` and `reportDeltaProgress` after
  every `handleChangedFile` that returns without fallback or completion,
  including skipped items (`internal/daemon/manager_delta.go:493-506`).
- `writeCheckpoint` serializes and writes the whole snapshot each time
  (`internal/daemon/manager_delta.go:566-569`).

A job whose diff has 1209 items but delivers documents for 3 performs about 1210
full-snapshot writes and 1209 progress updates while embedding 3 conversations.
This work inflates job duration, which keeps the engine busy, which is why 88
percent of feeder upserts are rejected and the backlog drains slowly.

## Fix

### Primary: reuse stored conversation vectors by content hash

A reused vector skips the embedder. The engine keys reuse on
`contentVectorKey(content)`, a SHA-256 of the chunk text alone, with no path,
message index, role, or id mixed in (`internal/semantic/reuse.go:25-28`). The
same key is used when stored vectors are loaded
(`internal/semantic/reuse.go:52-101`) and when the embed step checks each chunk
before calling the model (`internal/semantic/staging.go:177-197`). A conversation
chunk's text is a deterministic byte slice of the message text
(`splitConversationText`, `internal/daemon/manager_conversations.go:536-598`), so
a re-delivered unchanged message produces the same chunk text, the same hash, and
a reuse hit. Conversation ingests leave this off today: `runDeltaSync` sets
`reuse` to nil (`internal/daemon/manager_delta.go:220-221`), and `runBootstrap`
loads reuse only from child and sibling code collections, which a conversation
collection never has.

Turn reuse on for conversation ingests, scoped per conversation. For each
conversation about to be reindexed, read its existing vectors from the live
collection by relative-path prefix `conv/<id>/`, build the content-hash to vector
map, and pass it into that conversation's reindex. The reindex deletes the
conversation's prior rows and then inserts the new chunks
(`internal/semantic/service.go:199-232`); the embed step takes a reused vector for
any chunk whose hash is already in the map and calls the model only for the rest
(`internal/semantic/staging.go:177-212`). The map is loaded into memory before the
delete runs, so the delete that follows does not affect it.

Scope the read per conversation rather than loading the whole collection. The
collection holds about 44000 vectors, so a whole-collection load on every job
would move hundreds of megabytes for no reason. A prefix-scoped read returns only
the one conversation's chunks, which is the exact set the reindex needs.

This covers three cases with one mechanism. A changed conversation reuses its
unchanged messages and embeds only the new or edited ones. A backlog conversation
whose vectors already exist in the live collection reuses all of them and embeds
nothing. A genuinely new conversation finds no prior vectors and embeds every
chunk, which is correct.

Result: a changed conversation embeds only its new or changed chunks, and the
one-time backlog drains with almost no embedding. This removes the dominant GPU
cost in Layer 1 and the migration cost in Layer 2.

### Secondary: checkpoint only after real progress

Write the checkpoint and emit progress only for items that change
`state.working`, which means items that were embedded or removed, not items that
were skipped. This preserves crash resume, which only needs a checkpoint after
real progress, and removes the roughly 1206 redundant full-snapshot writes per
job. Shorter jobs lower the engine-busy fraction and let the backlog drain
faster.

### Optional: raise the feeder batch cap

`conversationSemanticMaxDocsPerBatch` is 800
(`internal/daemon` clyde side, `internal/daemon/conversation_semantic_sync.go:18`).
With reuse making embeds cheap and jobs short, a larger cap clears the one-time
backlog faster. This is lower priority once reuse lands, because reuse makes the
backlog nearly free regardless of batch size.

## Implementation details

The primary fix is two pieces: a prefix-scoped reuse loader and one call site that
uses it.

Add a prefix-scoped reuse loader in `internal/semantic`, next to
`loadReuseVectorsFromCollection` (`internal/semantic/reuse.go:52-101`). It queries
one collection for rows whose `relativePath` starts with a given prefix, outputs
the content and dense-vector columns, and returns `map[string][]float32` keyed by
`contentVectorKey(content)`. The existing `QueryIterator` read path in that file
and the prefix expression used by `deleteByRelativePathPrefix`
(`internal/semantic/removal.go`) supply both halves, so the loader reuses the same
prefix filter the delete already uses.

Call the loader per conversation, before the delete. The site is
`handleChangedFile` (`internal/daemon/manager_delta.go:510-548`), after `indexOne`
builds the new chunks and before `applyReindexForState` runs the delete and
insert. Read the conversation's existing vectors for `conv/<id>/` from the live
collection, then pass that map as the reuse argument for that one reindex call.

Read from the live conversation collection for both ingest paths. The delta path
(`runDeltaSync`) serves steady state and the current backlog, because the merkle
seed is non-empty so the cold-start conversations route through delta rather than
bootstrap. The bootstrap path (`runBootstrap`) runs only when no usable checkpoint
exists. Both should read prior vectors from the live collection, which is where
the existing 44000 vectors live.

## Tests, written before the fix

- A conversation with N messages is embedded once, then re-delivered with one
  new message. Assert the embedder receives one new chunk and reuses the rest,
  by checking the reused versus embedded split in the reindex progress.
- A job whose diff has many undelivered items writes the checkpoint a bounded
  number of times, not once per diff item.
- A manifest re-sync after a checkpoint reset, against an already-populated
  collection, reuses stored vectors and embeds close to zero.

## Verification

- `make test` and `make lint` in lm-semantic-search.
- After deploy, watch `jobs.jsonl`: per-job embedded chunk count falls to the
  size of the changed delta, the needed count drains quickly, and the
  engine-busy fraction in `clyde-daemon.jsonl` falls.

## Scope and risks

The primary and secondary fixes are engine-side in lm-semantic-search. The
optional batch-cap change is clyde-side.

The content-hash match is confirmed. Reuse keys on the SHA-256 of the chunk text
only (`internal/semantic/reuse.go:25-28`), the load and the per-chunk lookup use
that same key (`reuse.go:52-101`, `staging.go:177-197`), and a conversation
chunk's text is a deterministic byte slice of the message text
(`internal/daemon/manager_conversations.go:536-598`). An unchanged message reuses
its stored vector. No field in the hash would block the match.

Reading vectors before the delete is confirmed feasible. A reindex deletes first
and inserts second (`internal/semantic/service.go:199-232`), and the reuse map is
an in-memory Go map populated before the delete, so the Milvus delete does not
reach it (`internal/semantic/staging.go:177-212`). `LoadReuseVectors` and the new
prefix-scoped loader accept any collection name and have no exclusion for the
conversation collection.

The remaining cost is one Milvus read per reindexed conversation. A prefix-scoped
query is bounded by the conversation's own chunk count and runs only for changed
conversations, so it is small next to the embedding it removes.

The delete-and-reinsert of all of a conversation's rows still happens; only the
embedding is skipped. Milvus row writes are cheap next to GPU embedding, so this
is acceptable. A true message-level delta that inserts only the new rows is a
larger change and is not needed to remove the GPU cost.
