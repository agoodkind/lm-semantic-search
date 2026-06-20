package daemon

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"goodkind.io/gklog/correlation"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/spans"
	"goodkind.io/lm-semantic-search/internal/store"
)

const (
	conversationCanonicalPathPrefix = "chat:///"
	conversationChunkMaxBytes       = 60000
)

type conversationJobKind string

const (
	conversationJobKindUpsert conversationJobKind = "upsert"
	conversationJobKindDelete conversationJobKind = "delete"
)

// conversationJobPayload carries the work for one conversation job. An upsert
// holds the full manifest (every conversation id with its content fingerprint)
// and the documents clyde delivered for the changed ids; the shared routine
// diffs the manifest against the stored checkpoint and embeds only the changed
// conversations. A delete holds one conversation id to drop.
type conversationJobPayload struct {
	Kind           conversationJobKind
	CollectionName string
	Manifest       map[string]string
	Documents      []model.ConversationDocument
	ConversationID string
}

// RegisterConversationCollection records a virtual document collection that is
// addressed by logical collection id rather than a filesystem directory.
func (manager *Manager) RegisterConversationCollection(ctx context.Context, collectionID string) (model.Codebase, error) {
	trimmedCollectionID := strings.TrimSpace(collectionID)
	if trimmedCollectionID == "" {
		return model.Codebase{}, errors.New("collection id is required")
	}
	canonicalPath := conversationCanonicalPath(trimmedCollectionID)

	manager.mu.Lock()
	defer manager.mu.Unlock()

	if codebase, found := manager.findConversationCollectionLocked(trimmedCollectionID); found {
		return codebase, nil
	}

	collectionName := ""
	if manager.semantic != nil {
		collectionName = manager.semantic.ConversationCollectionName(trimmedCollectionID)
	}
	if collectionName == "" {
		return model.Codebase{}, errors.New("conversation collection name is unavailable")
	}

	codebase := newCodebaseRecord(canonicalPath)
	codebase.Kind = model.CodebaseKindDocument
	codebase.Status = model.CodebaseStatusIndexed
	codebase.EffectiveConfig = manager.enrichIndexConfig(emptyAutoIndexConfig())
	codebase.EffectiveConfig.IgnoreDigest = digestIndexConfig(codebase.EffectiveConfig)
	codebase.CollectionName = collectionName
	codebase.UpdatedAt = clock.Now()
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		slog.ErrorContext(ctx, "persist conversation collection registration failed", "collection_id", trimmedCollectionID, "err", err)
		return model.Codebase{}, fmt.Errorf("persist conversation collection %s: %w", trimmedCollectionID, err)
	}
	return codebase, nil
}

// SyncConversationManifest diffs clyde's full conversation manifest against the
// stored checkpoint and returns the ids the engine needs: the conversations new
// or changed since the last successful ingest. clyde then sends documents for
// only those ids. The engine owns drift, so clyde keeps no change-tracking
// state and a slow first embed runs exactly once.
func (manager *Manager) SyncConversationManifest(ctx context.Context, collectionID string, manifest map[string]string) ([]string, error) {
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		return nil, err
	}

	configDigest := codebase.EffectiveConfig.IgnoreDigest
	legacyDigest := manager.legacyDigestForCodebase(codebase.ID)
	seed := merkle.LoadSnapshotForConfig(manager.merklePath(codebase.ID), configDigest, legacyDigest)
	current := merkle.Snapshot{ConfigDigest: configDigest, Files: manifest, Inodes: nil}
	diff := merkle.DiffSnapshots(seed, current)

	needed := make([]string, 0, len(diff.Added)+len(diff.Modified))
	needed = append(needed, diff.Added...)
	needed = append(needed, diff.Modified...)
	sort.Strings(needed)
	return needed, nil
}

// upsertConversationDocuments queues an asynchronous ingest. When manifest is
// nil it is derived from the delivered documents, so a caller that hands over a
// complete set need not compute fingerprints itself.
func (manager *Manager) upsertConversationDocuments(ctx context.Context, collectionID string, documents []model.ConversationDocument, manifest map[string]string, client model.ClientInfo) (model.Job, error) {
	for _, document := range documents {
		if strings.TrimSpace(document.ConversationID) == "" {
			return model.Job{}, errors.New("conversation id is required")
		}
	}
	if manifest == nil {
		manifest = manifestFromDocuments(documents)
	}
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		return model.Job{}, err
	}
	payload := conversationJobPayload{
		Kind:           conversationJobKindUpsert,
		CollectionName: codebase.CollectionName,
		Manifest:       manifest,
		Documents:      documents,
		ConversationID: "",
	}
	return manager.queueConversationJob(ctx, codebase, client, payload)
}

// DeleteConversation queues an asynchronous delete for one conversation id.
func (manager *Manager) DeleteConversation(ctx context.Context, collectionID string, conversationID string) (model.Job, error) {
	return manager.deleteConversation(ctx, collectionID, conversationID, model.ClientInfo{Name: "", PID: 0})
}

// SearchConversations searches a registered virtual conversation collection.
func (manager *Manager) SearchConversations(ctx context.Context, collectionID string, query string, limit int32, filter conversationSearchFilter, perConversationLimit int32) ([]model.StoredChunk, error) {
	trimmedCollectionID := strings.TrimSpace(collectionID)

	manager.mu.Lock()
	codebase, found := manager.findConversationCollectionLocked(trimmedCollectionID)
	manager.mu.Unlock()
	if !found {
		return nil, nil
	}
	return manager.searchConversationCollectionFiltered(ctx, codebase, query, limit, filter, perConversationLimit)
}

// SearchWithinConversation retrieves one conversation's matching rows plus the
// content fingerprint the engine has embedded for it. An empty fingerprint
// means the conversation is not indexed; a fingerprint differing from the
// conversation's current one means the index trails the transcript. Either way
// the caller decides whether to literal-scan newer content.
func (manager *Manager) SearchWithinConversation(ctx context.Context, collectionID string, conversationID string, query string, limit int32, filter conversationSearchFilter) ([]model.StoredChunk, string, error) {
	trimmedConversationID := strings.TrimSpace(conversationID)
	if trimmedConversationID == "" {
		return nil, "", errors.New("conversation id is required")
	}
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		return nil, "", err
	}
	filter.ConversationIDs = []string{trimmedConversationID}
	chunks, err := manager.searchConversationCollectionFiltered(ctx, codebase, query, limit, filter, 0)
	if err != nil {
		return nil, "", err
	}
	return chunks, manager.conversationIndexedFingerprint(codebase, trimmedConversationID), nil
}

// conversationIndexedFingerprint reads the checkpointed content fingerprint
// for one conversation from the collection's merkle snapshot. Empty when the
// engine has never embedded the conversation.
func (manager *Manager) conversationIndexedFingerprint(codebase model.Codebase, conversationID string) string {
	snapshot := merkle.LoadSnapshotForConfig(
		manager.merklePath(codebase.ID),
		codebase.EffectiveConfig.IgnoreDigest,
		manager.legacyDigestForCodebase(codebase.ID),
	)
	return snapshot.Files[conversationID]
}

const (
	// conversationSearchInflationFactor over-fetches before the per-conversation
	// cap drops rows, so the cap does not starve the final result list below the
	// requested limit when more matching rows exist.
	conversationSearchInflationFactor = 4
	// conversationSearchInflationCap bounds the over-fetch so it never grows
	// unbounded with the requested limit.
	conversationSearchInflationCap = int32(500)
)

// inflatedConversationSearchLimit returns how many rows to fetch before
// post-retrieval filtering. Every scope dimension is now pushed into Milvus, so
// the only post-retrieval row-dropper that can starve the result is the
// per-conversation cap. When that cap is off, the requested limit is exact;
// when it is on, the fetch is inflated by a bounded factor so the cap has a
// surplus to draw the final limit from.
func inflatedConversationSearchLimit(limit int32, perConversationLimit int32) int32 {
	if perConversationLimit <= 0 {
		return limit
	}
	inflated := min(limit*conversationSearchInflationFactor, conversationSearchInflationCap)
	return max(inflated, limit)
}

// searchConversationCollectionFiltered is the one retrieval path under both
// conversation search RPCs. When the vector store is up, every scope dimension
// is pushed into Milvus as a native scalar-column expression and the engine
// returns the result already reduced to the requested limit: it pages the ranked
// search by offset so the per-conversation cap and min_score fill the limit
// deterministically instead of starving it, reusing one query embedding across
// pages. When the store is down, the literal chunk cache ranks a bounded
// over-fetch and reduces it in process.
func (manager *Manager) searchConversationCollectionFiltered(ctx context.Context, codebase model.Codebase, query string, limit int32, filter conversationSearchFilter, perConversationLimit int32) ([]model.StoredChunk, error) {
	if limit <= 0 {
		limit = 10
	}

	if manager.semantic != nil && manager.semantic.Available() {
		chunks, err := manager.semantic.SearchConversationCollectionCapped(ctx, codebase.CollectionName, query, limit, perConversationLimit, filter.MinScore, filter.toSemanticFilter())
		if err == nil {
			manager.noteDependencyHealthy()
			return chunks, nil
		}
		manager.noteDependencyFailure(err)
		slog.ErrorContext(ctx, "search conversation collection failed", "collection", codebase.CollectionName, "err", err)
	}

	fetchLimit := inflatedConversationSearchLimit(limit, perConversationLimit)
	return manager.searchConversationChunkCache(ctx, codebase, query, limit, fetchLimit, filter, perConversationLimit)
}

func (manager *Manager) deleteConversation(ctx context.Context, collectionID string, conversationID string, client model.ClientInfo) (model.Job, error) {
	trimmedConversationID := strings.TrimSpace(conversationID)
	if trimmedConversationID == "" {
		return model.Job{}, errors.New("conversation id is required")
	}
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		return model.Job{}, err
	}
	payload := conversationJobPayload{
		Kind:           conversationJobKindDelete,
		CollectionName: codebase.CollectionName,
		Manifest:       nil,
		Documents:      nil,
		ConversationID: trimmedConversationID,
	}
	return manager.queueConversationJob(ctx, codebase, client, payload)
}

func (manager *Manager) queueConversationJob(ctx context.Context, codebase model.Codebase, client model.ClientInfo, payload conversationJobPayload) (model.Job, error) {
	var emptyJob model.Job

	manager.mu.Lock()
	current, found := manager.codebases[codebase.ID]
	if !found {
		manager.mu.Unlock()
		return emptyJob, fmt.Errorf("conversation collection not tracked: %s", codebase.CanonicalPath)
	}
	if activeJob, active, err := manager.activeConversationJobLocked(current); err != nil {
		manager.mu.Unlock()
		return emptyJob, err
	} else if active {
		manager.mu.Unlock()
		return emptyJob, fmt.Errorf("conflicting active job %s for conversation collection %s", activeJob.ID, current.CanonicalPath)
	}

	now := clock.Now()
	job := newQueuedJob(
		current.ID,
		current.CanonicalPath,
		current.CanonicalPath,
		client,
		string(jobOperationConversationIngest),
		false,
		current.EffectiveConfig,
		now,
	)
	current.Status = model.CodebaseStatusIndexing
	current.ActiveJobID = job.ID
	current.UpdatedAt = now
	manager.codebases[current.ID] = current
	manager.conversationJobs[job.ID] = payload
	if err := manager.saveLocked(); err != nil {
		delete(manager.conversationJobs, job.ID)
		manager.mu.Unlock()
		return emptyJob, err
	}
	if err := manager.appendJobLocked("conversation_ingest", job); err != nil {
		delete(manager.conversationJobs, job.ID)
		manager.mu.Unlock()
		return emptyJob, err
	}
	manager.mu.Unlock()

	ctx = spans.Attach(
		ctx,
		correlation.IdentityAttribute{Key: "job_id", Value: job.ID},
		correlation.IdentityAttribute{Key: "codebase_id", Value: current.ID},
	)
	manager.runJobAsync(ctx, job.ID)
	return job, nil
}

func (manager *Manager) activeConversationJobLocked(codebase model.Codebase) (model.Job, bool, error) {
	var emptyJob model.Job
	if codebase.ActiveJobID == "" {
		return emptyJob, false, nil
	}
	activeJob, found := manager.jobs[codebase.ActiveJobID]
	if !found {
		return emptyJob, false, nil
	}
	switch activeJob.State {
	case model.JobStateCompleted, model.JobStateFailed, model.JobStateCancelled:
		return emptyJob, false, nil
	case model.JobStateQueued, model.JobStateRunning, model.JobStateCancelling:
		return activeJob, true, nil
	default:
		return emptyJob, false, fmt.Errorf("unknown job state %s for active job %s", activeJob.State, activeJob.ID)
	}
}

// runConversationIngest runs one conversation job. An upsert flows through the
// same delta-then-bootstrap routine code uses, with a conversation source
// feeding the manifest and documents; a delete drops one conversation's rows.
func (manager *Manager) runConversationIngest(ctx context.Context, job model.Job) {
	payload, found := manager.conversationJobPayload(job.ID)
	if !found {
		manager.updateJobFailed(ctx, job.ID, errors.New("conversation job payload missing"))
		return
	}

	select {
	case <-ctx.Done():
		manager.updateJobCancelled(ctx, job.ID)
		return
	default:
	}

	switch payload.Kind {
	case conversationJobKindDelete:
		manager.runConversationDelete(ctx, job, payload)
	case conversationJobKindUpsert:
		source := newConversationItemSource(payload.CollectionName, payload.Manifest, payload.Documents)
		if manager.runDeltaSync(ctx, job, source) {
			return
		}
		manager.runBootstrap(ctx, job, source)
	default:
		manager.updateJobFailed(ctx, job.ID, fmt.Errorf("unknown conversation job kind %s", payload.Kind))
	}
}

// runConversationDelete drops one conversation's rows from the live collection
// and the literal-fallback chunk cache, then marks the job complete. It does not
// touch the merkle checkpoint: a later manifest sync that omits the id converges
// the same removal idempotently.
func (manager *Manager) runConversationDelete(ctx context.Context, job model.Job, payload conversationJobPayload) {
	if manager.semantic != nil {
		if err := manager.semantic.DeleteConversation(ctx, payload.CollectionName, payload.ConversationID); err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
				manager.updateJobCancelled(ctx, job.ID)
				return
			}
			manager.updateJobFailed(ctx, job.ID, err)
			return
		}
	}
	if err := manager.dropConversationFromChunkCache(ctx, job.CodebaseID, payload.ConversationID); err != nil {
		manager.updateJobFailed(ctx, job.ID, err)
		return
	}
	manager.finishConversationDelete(job.ID)
}

func (manager *Manager) finishConversationDelete(jobID string) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	job, found := manager.jobs[jobID]
	if !found {
		delete(manager.conversationJobs, jobID)
		return
	}
	now := clock.Now()
	job.State = model.JobStateCompleted
	job.UpdatedAt = now
	job.CompletedAt = &now
	job.Progress.Phase = "completed"
	job.Progress.OverallPercent = 100
	job.Progress.LastEventAt = now
	job.Progress.HeartbeatAt = now
	delete(manager.conversationJobs, jobID)
	if err := manager.appendJobLocked("job_completed", job); err != nil {
		slog.Error("append completed conversation delete event failed", "job_id", jobID, "err", err)
	}

	codebase, found := manager.codebases[job.CodebaseID]
	if !found {
		return
	}
	codebase.Status = model.CodebaseStatusIndexed
	codebase.ActiveJobID = ""
	codebase.UpdatedAt = now
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		slog.Error("write registry after conversation delete failed", "job_id", jobID, "err", err)
	}
}

func (manager *Manager) searchConversationChunkCache(ctx context.Context, codebase model.Codebase, query string, limit int32, fetchLimit int32, filter conversationSearchFilter, perConversationLimit int32) ([]model.StoredChunk, error) {
	chunks, err := store.ReadChunks(manager.chunkPath(codebase.ID))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []model.StoredChunk{}, nil
		}
		slog.ErrorContext(ctx, "read conversation chunk cache failed", "codebase_id", codebase.ID, "err", err)
		return nil, fmt.Errorf("read conversation chunk cache for %s: %w", codebase.ID, err)
	}
	scoped := make([]model.StoredChunk, 0, len(chunks))
	for _, chunk := range chunks {
		if filter.matchesScope(chunk) {
			scoped = append(scoped, chunk)
		}
	}
	// Rank fetchLimit rows, not limit, so the per-conversation cap has a surplus
	// to draw the final limit from rather than truncating an already-limited
	// ranking. fetchLimit equals limit when the cap is off.
	ranked := rankChunks(scoped, query, fetchLimit, nil, "")
	return applyConversationSearchFilter(ranked, filter, perConversationLimit, limit), nil
}

// mergeConversationChunkCache keeps the literal-fallback chunk cache complete
// across an incremental ingest. It keeps the prior chunks for conversations
// still present in the manifest and not re-sent this run, drops conversations no
// longer in the manifest, replaces re-sent conversations with their fresh
// chunks, and writes the union back. The cache backs conversation search when
// the vector store is down.
func (manager *Manager) mergeConversationChunkCache(ctx context.Context, codebaseID string, newChunks []model.StoredChunk, manifest map[string]string) error {
	chunkPath := manager.chunkPath(codebaseID)
	existing, err := store.ReadChunks(chunkPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			existing = []model.StoredChunk{}
		} else {
			slog.ErrorContext(ctx, "read conversation chunk cache failed", "codebase_id", codebaseID, "err", err)
			return fmt.Errorf("read conversation chunk cache for %s: %w", codebaseID, err)
		}
	}

	resentIDs := make(map[string]struct{})
	for _, conversationID := range conversationIDsFromChunks(newChunks) {
		resentIDs[conversationID] = struct{}{}
	}

	kept := make([]model.StoredChunk, 0, len(existing)+len(newChunks))
	for _, chunk := range existing {
		conversationID := strings.TrimSpace(chunk.ConversationID)
		if conversationID == "" {
			continue
		}
		if _, stillPresent := manifest[conversationID]; !stillPresent {
			continue
		}
		if _, resent := resentIDs[conversationID]; resent {
			continue
		}
		kept = append(kept, chunk)
	}
	kept = append(kept, newChunks...)

	if err := store.WriteChunks(chunkPath, kept); err != nil {
		slog.ErrorContext(ctx, "write conversation chunk cache failed", "codebase_id", codebaseID, "err", err)
		return fmt.Errorf("write conversation chunk cache for %s: %w", codebaseID, err)
	}
	return nil
}

func (manager *Manager) dropConversationFromChunkCache(ctx context.Context, codebaseID string, conversationID string) error {
	chunkPath := manager.chunkPath(codebaseID)
	chunks, err := store.ReadChunks(chunkPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		slog.ErrorContext(ctx, "read conversation chunk cache failed", "codebase_id", codebaseID, "err", err)
		return fmt.Errorf("read conversation chunk cache for %s: %w", codebaseID, err)
	}
	chunks = dropConversationChunks(chunks, []string{conversationID})
	if err := store.WriteChunks(chunkPath, chunks); err != nil {
		slog.ErrorContext(ctx, "write conversation chunk cache failed", "codebase_id", codebaseID, "err", err)
		return fmt.Errorf("write conversation chunk cache for %s: %w", codebaseID, err)
	}
	return nil
}

func conversationIDsFromChunks(chunks []model.StoredChunk) []string {
	seen := make(map[string]struct{})
	conversationIDs := make([]string, 0)
	for _, chunk := range chunks {
		conversationID := strings.TrimSpace(chunk.ConversationID)
		if conversationID == "" {
			continue
		}
		if _, found := seen[conversationID]; found {
			continue
		}
		seen[conversationID] = struct{}{}
		conversationIDs = append(conversationIDs, conversationID)
	}
	return conversationIDs
}

func dropConversationChunks(chunks []model.StoredChunk, conversationIDs []string) []model.StoredChunk {
	prefixes := conversationRelativePathPrefixes(conversationIDs)
	if len(prefixes) == 0 {
		return chunks
	}

	kept := make([]model.StoredChunk, 0, len(chunks))
	for _, chunk := range chunks {
		if chunkHasConversationPrefix(chunk, prefixes) {
			continue
		}
		kept = append(kept, chunk)
	}
	return kept
}

func conversationRelativePathPrefixes(conversationIDs []string) []string {
	seen := make(map[string]struct{})
	prefixes := make([]string, 0, len(conversationIDs))
	for _, conversationID := range conversationIDs {
		trimmedConversationID := strings.TrimSpace(conversationID)
		if trimmedConversationID == "" {
			continue
		}
		prefix := conversationRelativePathPrefix(trimmedConversationID)
		if _, found := seen[prefix]; found {
			continue
		}
		seen[prefix] = struct{}{}
		prefixes = append(prefixes, prefix)
	}
	return prefixes
}

func chunkHasConversationPrefix(chunk model.StoredChunk, prefixes []string) bool {
	for _, prefix := range prefixes {
		if strings.HasPrefix(chunk.RelativePath, prefix) {
			return true
		}
	}
	return false
}

func (manager *Manager) conversationJobPayload(jobID string) (conversationJobPayload, bool) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	payload, found := manager.conversationJobs[jobID]
	return payload, found
}

// manifestFromDocuments derives a content fingerprint per conversation from a
// complete delivered document set, so a caller that hands over every document
// need not compute fingerprints. The fingerprint covers each message's index,
// role, and text in message order.
func manifestFromDocuments(documents []model.ConversationDocument) map[string]string {
	byID := make(map[string][]model.ConversationDocument)
	order := make([]string, 0)
	for _, document := range documents {
		conversationID := strings.TrimSpace(document.ConversationID)
		if conversationID == "" {
			continue
		}
		if _, seen := byID[conversationID]; !seen {
			order = append(order, conversationID)
		}
		byID[conversationID] = append(byID[conversationID], document)
	}
	manifest := make(map[string]string, len(order))
	for _, conversationID := range order {
		manifest[conversationID] = fingerprintConversationDocuments(byID[conversationID])
	}
	return manifest
}

func fingerprintConversationDocuments(documents []model.ConversationDocument) string {
	sorted := make([]model.ConversationDocument, len(documents))
	copy(sorted, documents)
	sort.Slice(sorted, func(first int, second int) bool {
		return sorted[first].MessageIndex < sorted[second].MessageIndex
	})
	hasher := sha256.New()
	for _, document := range sorted {
		hasher.Write([]byte(strconv.Itoa(int(document.MessageIndex))))
		hasher.Write([]byte{0})
		hasher.Write([]byte(document.Role))
		hasher.Write([]byte{0})
		hasher.Write([]byte(document.Text))
		hasher.Write([]byte{0})
	}
	return hex.EncodeToString(hasher.Sum(nil))
}

func conversationDocumentsToStoredChunks(documents []model.ConversationDocument) ([]model.StoredChunk, error) {
	chunks := make([]model.StoredChunk, 0, len(documents))
	for _, document := range documents {
		conversationID := strings.TrimSpace(document.ConversationID)
		if conversationID == "" {
			return nil, errors.New("conversation id is required")
		}
		parentConversationID := strings.TrimSpace(document.ParentConversationID)
		pieces := splitConversationText(document.Text)
		for partIndex, piece := range pieces {
			chunks = append(chunks, model.StoredChunk{
				Content:              piece,
				RelativePath:         conversationRelativePath(conversationID, document.MessageIndex, partIndex, len(pieces) > 1),
				StartLine:            0,
				EndLine:              0,
				Language:             "",
				FileExtension:        "",
				ConversationID:       conversationID,
				ParentConversationID: parentConversationID,
				MessageIndex:         document.MessageIndex,
				Role:                 document.Role,
				TimestampUnix:        document.TimestampUnix,
				WorkspaceRoot:        document.WorkspaceRoot,
				Score:                0,
			})
		}
	}
	return chunks, nil
}

func conversationRelativePath(conversationID string, messageIndex int32, partIndex int, multipart bool) string {
	basePath := fmt.Sprintf("conv/%s/%d", conversationID, messageIndex)
	if !multipart {
		return basePath
	}
	return fmt.Sprintf("%s/%d", basePath, partIndex)
}

func conversationRelativePathPrefix(conversationID string) string {
	return "conv/" + conversationID + "/"
}

func splitConversationText(text string) []string {
	if len(text) <= conversationChunkMaxBytes {
		return []string{text}
	}
	pieces := make([]string, 0, (len(text)+conversationChunkMaxBytes-1)/conversationChunkMaxBytes)
	start := 0
	for start < len(text) {
		end := start + conversationChunkMaxBytes
		if end >= len(text) {
			pieces = append(pieces, text[start:])
			break
		}
		for end > start && !utf8.RuneStart(text[end]) {
			end--
		}
		if end == start {
			_, size := utf8.DecodeRuneInString(text[start:])
			end = start + size
		}
		pieces = append(pieces, text[start:end])
		start = end
	}
	return pieces
}

func (manager *Manager) findConversationCollectionLocked(collectionID string) (model.Codebase, bool) {
	canonicalPath := conversationCanonicalPath(collectionID)
	for _, codebase := range manager.codebases {
		if codebase.Kind != model.CodebaseKindDocument {
			continue
		}
		if codebase.CanonicalPath == canonicalPath {
			return codebase, true
		}
	}
	var emptyCodebase model.Codebase
	return emptyCodebase, false
}

func conversationCanonicalPath(collectionID string) string {
	return conversationCanonicalPathPrefix + collectionID
}
