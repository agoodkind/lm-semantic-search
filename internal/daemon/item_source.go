package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"maps"

	"goodkind.io/lm-semantic-search/internal/discovery"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// itemSource is the one part of the indexing routine that differs by kind. The
// shared delta and bootstrap routine asks a source to list the current items
// with a content fingerprint each, to produce one item's chunks on request, to
// name the store rows that drop when an item changes or leaves, and to name the
// progress unit. A code source walks the filesystem and reads files; a
// conversation source reads the manifest and documents the daemon was handed.
type itemSource interface {
	// capture lists the current items as itemID -> content fingerprint.
	capture(ctx context.Context) (merkle.Snapshot, error)
	// indexOne produces the stored chunks and fingerprint for one item.
	indexOne(ctx context.Context, itemID string) (indexer.OneFileResult, error)
	// removalFor maps item ids to the store removal that drops their prior rows.
	removalFor(itemIDs []string) semantic.Removal
	// reuseSource names where one item's already-embedded vectors live: the
	// collection and the relativePath prefix that scopes the read. ok=false
	// means the item has no per-item reuse source and every chunk embeds. A
	// conversation returns its live collection and conv/<id>/ prefix, so a
	// changed conversation re-embeds only its changed chunks.
	reuseSource(itemID string) (collectionName string, relativePathPrefix string, ok bool)
	// unit is the human progress noun, "file" or "document".
	unit() string
}

// codeItemSource lists and reads a filesystem codebase. It is the byte-for-byte
// behavior the daemon ran before the routine became source-driven: capture is a
// merkle walk and indexOne is one file read and split.
type codeItemSource struct {
	runner        indexingRunner
	canonicalPath string
	config        model.IndexConfig
	// onRules receives the walk's resolved ignore rules each capture, so the
	// manager can persist them without a second walk. Nil disables reporting.
	onRules func(discovery.IgnoreRules)
}

func newCodeItemSource(runner indexingRunner, canonicalPath string, config model.IndexConfig, onRules func(discovery.IgnoreRules)) codeItemSource {
	return codeItemSource{runner: runner, canonicalPath: canonicalPath, config: config, onRules: onRules}
}

func (source codeItemSource) capture(ctx context.Context) (merkle.Snapshot, error) {
	snapshot, rules, err := merkle.Capture(ctx, source.canonicalPath, source.config)
	if err != nil {
		slog.ErrorContext(ctx, "capture code snapshot failed", "path", source.canonicalPath, "err", err)
		return merkle.Snapshot{}, fmt.Errorf("capture code snapshot for %s: %w", source.canonicalPath, err)
	}
	if source.onRules != nil {
		source.onRules(rules)
	}
	return snapshot, nil
}

func (source codeItemSource) indexOne(ctx context.Context, relativePath string) (indexer.OneFileResult, error) {
	type indexOneOutcome struct {
		result indexer.OneFileResult
		err    error
	}
	done := make(chan indexOneOutcome, 1)
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				var empty indexer.OneFileResult
				done <- indexOneOutcome{result: empty, err: fmt.Errorf("index code file panic: %v", recovered)}
			}
		}()
		result, err := source.runner.IndexOne(ctx, source.canonicalPath, relativePath, source.config)
		done <- indexOneOutcome{result: result, err: err}
	}()

	select {
	case <-ctx.Done():
		return indexer.OneFileResult{}, fmt.Errorf("index code file %s cancelled: %w", relativePath, ctx.Err())
	case outcome := <-done:
		if outcome.err == nil {
			return outcome.result, nil
		}
		slog.ErrorContext(ctx, "index code file failed", "path", relativePath, "err", outcome.err)
		return indexer.OneFileResult{}, fmt.Errorf("index code file %s: %w", relativePath, outcome.err)
	}
}

func (source codeItemSource) removalFor(itemIDs []string) semantic.Removal {
	return semantic.RemovePaths(itemIDs)
}

// reuseSource reports no per-item reuse for code files: a code file's chunks
// share one relativePath and change together, so there is nothing unchanged to
// reuse within one file. Cross-collection reuse (worktree siblings, merge-down
// children) stays on the bootstrap path.
func (source codeItemSource) reuseSource(string) (string, string, bool) {
	return "", "", false
}

func (source codeItemSource) unit() string {
	return "file"
}

// conversationItemSource lists and reads conversation documents the daemon was
// handed over the wire. capture returns the manifest clyde sent (every
// conversation id with its content fingerprint), and indexOne returns the chunks
// for the documents clyde delivered for one conversation. A conversation's
// messages span many rows under one conv/<id>/ prefix, so its removal is a
// prefix delete rather than the code file's exact path.
type conversationItemSource struct {
	// collectionName is the live conversation collection, the reuse source for
	// a changed conversation's already-embedded vectors.
	collectionName string
	manifest       map[string]string
	documents      map[string][]model.ConversationDocument
	splitterID     string
}

func newConversationItemSource(collectionName string, manifest map[string]string, documents []model.ConversationDocument) conversationItemSource {
	byID := make(map[string][]model.ConversationDocument, len(manifest))
	for _, document := range documents {
		conversationID := document.ConversationID
		byID[conversationID] = append(byID[conversationID], document)
	}
	return conversationItemSource{collectionName: collectionName, manifest: manifest, documents: byID, splitterID: ""}
}

func (source conversationItemSource) capture(_ context.Context) (merkle.Snapshot, error) {
	files := make(map[string]string, len(source.manifest))
	maps.Copy(files, source.manifest)
	return merkle.Snapshot{ConfigDigest: "", Files: files, Inodes: nil}, nil
}

func (source conversationItemSource) indexOne(_ context.Context, conversationID string) (indexer.OneFileResult, error) {
	documents, delivered := source.documents[conversationID]
	if !delivered || len(documents) == 0 {
		// clyde asked for this conversation's documents but delivered none, so it
		// cannot be embedded this pass. Skip it without advancing the checkpoint, so
		// the next manifest sync still classifies it changed and clyde resends.
		return indexer.OneFileResult{Chunks: nil, FileHash: "", Skipped: true, SkipReason: indexer.SkipUnreadable, Removed: false}, nil
	}
	chunks, err := conversationDocumentsToStoredChunks(documents)
	if err != nil {
		return indexer.OneFileResult{Chunks: nil, FileHash: "", Skipped: false, SkipReason: indexer.SkipNone, Removed: false}, err
	}
	return indexer.OneFileResult{
		Chunks:     chunks,
		FileHash:   source.manifest[conversationID],
		Skipped:    false,
		SkipReason: indexer.SkipNone,
		Removed:    false,
	}, nil
}

func (source conversationItemSource) removalFor(itemIDs []string) semantic.Removal {
	prefixes := make([]string, 0, len(itemIDs))
	for _, conversationID := range itemIDs {
		prefixes = append(prefixes, conversationRelativePathPrefix(conversationID))
	}
	return semantic.RemovePrefixes(prefixes)
}

// reuseSource points one conversation's reuse read at its own rows in the live
// collection: the conv/<id>/ prefix the reindex is about to delete. Loaded
// before the delete, those vectors let unchanged messages skip the embedder.
func (source conversationItemSource) reuseSource(conversationID string) (string, string, bool) {
	if source.collectionName == "" {
		return "", "", false
	}
	return source.collectionName, conversationRelativePathPrefix(conversationID), true
}

func (source conversationItemSource) unit() string {
	return "document"
}
