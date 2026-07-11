package daemon

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"goodkind.io/gklog/correlation"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/spans"
)

const (
	conversationCanonicalPathPrefix = "chat:///"
	conversationChunkMaxBytes       = 60000
	conversationToolSummaryMaxBytes = 2000
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
	// Absence is the upsert's caller-declared policy for a conversation the
	// manifest omits. It is meaningful only for an upsert; a delete sets it
	// explicitly to absenceRetain (also the zero value) but never consults it.
	Absence absencePolicy
	// Reexamine forces delivered conversations with absent or stale derived
	// markers into this run's changed set even when their fingerprints are
	// unchanged. It is meaningful only for an upsert and stays false for the
	// normal sync.
	Reexamine bool
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
		// Roll the in-memory record back when the registry write fails, mirroring
		// the adopt and worktree paths, so a failed persist does not leave a
		// codebase that later lookups treat as registered until restart.
		delete(manager.codebases, codebase.ID)
		slog.ErrorContext(ctx, "persist conversation collection registration failed", "collection_id", trimmedCollectionID, "err", err)
		return model.Codebase{}, fmt.Errorf("persist conversation collection %s: %w", trimmedCollectionID, err)
	}
	// A persisted codebase record pairs with one observer signal so no saveLocked
	// path silently skips invalidation; for a document collection it is a no-op
	// delete, which keeps the invariant uniform across every record write.
	manager.observer.Invalidate(codebase.ID)
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

	manager.mu.Lock()
	cursor := manager.conversationSyncCursors[codebase.ID]
	needed, nextCursor := capNeededConversations(diff.Added, diff.Modified, manager.config.MaxConversationsPerIngest, cursor)
	if nextCursor != "" {
		manager.conversationSyncCursors[codebase.ID] = nextCursor
	}
	manager.mu.Unlock()
	return needed, nil
}

// One shared cursor tracks pre-sort rotation order across modified overflow and
// added windows. A cursor past a list's end wraps to that list's start so
// rotation stays live without per-list cursors.
func capNeededConversations(added []string, modified []string, limit int, cursor string) ([]string, string) {
	if limit <= 0 {
		needed := make([]string, 0, len(added)+len(modified))
		needed = append(needed, added...)
		needed = append(needed, modified...)
		sort.Strings(needed)
		return needed, cursor
	}

	capped := make([]string, 0, min(limit, len(added)+len(modified)))
	nextCursor := ""
	if len(modified) > limit {
		modifiedWindow := firstN(rotateAfter(modified, cursor), limit)
		capped = append(capped, modifiedWindow...)
		if len(modifiedWindow) > 0 {
			nextCursor = modifiedWindow[len(modifiedWindow)-1]
		}
		sort.Strings(capped)
		return capped, nextCursor
	}

	sortedModified := append([]string(nil), modified...)
	sort.Strings(sortedModified)
	capped = append(capped, sortedModified...)
	if len(sortedModified) > 0 {
		nextCursor = sortedModified[len(sortedModified)-1]
	}
	remainder := limit - len(capped)
	if remainder > 0 {
		addedWindow := firstN(rotateAfter(added, cursor), remainder)
		capped = append(capped, addedWindow...)
		if len(addedWindow) > 0 {
			nextCursor = addedWindow[len(addedWindow)-1]
		}
	}
	sort.Strings(capped)
	return capped, nextCursor
}

func rotateAfter(values []string, cursor string) []string {
	rotated := append([]string(nil), values...)
	sort.Strings(rotated)
	if len(rotated) == 0 || cursor == "" {
		return rotated
	}
	start := sort.SearchStrings(rotated, cursor)
	if start < len(rotated) && rotated[start] == cursor {
		start++
	}
	if start >= len(rotated) {
		return rotated
	}
	return append(append([]string(nil), rotated[start:]...), rotated[:start]...)
}

func firstN(values []string, limit int) []string {
	if limit >= len(values) {
		return values
	}
	return values[:limit]
}

// upsertConversationDocuments queues an asynchronous ingest. When manifest is
// nil it is derived from the delivered documents, so a caller that hands over a
// complete set need not compute fingerprints itself.
func (manager *Manager) upsertConversationDocuments(ctx context.Context, collectionID string, documents []model.ConversationDocument, manifest map[string]string, client model.ClientInfo, absence absencePolicy, reexamine bool) (model.Job, error) {
	for _, document := range documents {
		if strings.TrimSpace(document.ConversationID) == "" {
			return model.Job{}, errors.New("conversation id is required")
		}
	}
	if manifest == nil {
		// Deriving the manifest from only the delivered documents is safe under
		// retain: an omitted conversation is kept either way. Under an authoritative
		// (delete-on-absence) upsert it is dangerous, because the derived manifest
		// lists only the delivered ids, so every other indexed conversation would be
		// treated as absent and deleted. Require an explicit manifest there.
		if absence == absenceDeleteGuarded {
			return model.Job{}, errors.New("authoritative conversation upsert requires an explicit manifest")
		}
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
		Absence:        absence,
		Reexamine:      reexamine,
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
// the caller decides whether to refresh newer content.
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

func (manager *Manager) backfillConversationScalars(ctx context.Context, collectionID string, enrichment semantic.ConversationEnrichment, dryRun bool) (changed int, orphan int, err error) {
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		return 0, 0, err
	}
	if manager.semantic == nil {
		return 0, 0, semantic.ErrUnavailable
	}
	changed, orphan, err = manager.semantic.BackfillConversationEnrichment(ctx, codebase.CollectionName, enrichment, dryRun)
	if err != nil {
		slog.ErrorContext(ctx, "backfill conversation scalars failed", "collection_id", collectionID, "collection", codebase.CollectionName, "changed", changed, "orphan", orphan, "err", err)
		return changed, orphan, fmt.Errorf("backfill conversation scalars for %s: %w", collectionID, err)
	}
	return changed, orphan, nil
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

// searchConversationCollectionFiltered is the one retrieval path under both
// conversation search RPCs. Every scope dimension is pushed into Milvus as a
// native scalar-column expression and the engine returns the result already
// reduced to the requested limit: it pages the ranked search by offset so the
// per-conversation cap and min_score fill the limit deterministically instead
// of starving it, reusing one query embedding across pages.
func (manager *Manager) searchConversationCollectionFiltered(ctx context.Context, codebase model.Codebase, query string, limit int32, filter conversationSearchFilter, perConversationLimit int32) ([]model.StoredChunk, error) {
	if limit <= 0 {
		limit = 10
	}

	if manager.semantic == nil || !manager.semantic.Available() {
		manager.noteDependencyFailure(semantic.ErrUnavailable)
		return nil, semantic.ErrUnavailable
	}
	chunks, err := manager.semantic.SearchConversationCollectionCapped(ctx, codebase.CollectionName, query, limit, perConversationLimit, filter.MinScore, filter.toSemanticFilter())
	if err != nil {
		manager.noteDependencyFailure(err)
		slog.ErrorContext(ctx, "search conversation collection failed", "collection", codebase.CollectionName, "err", err)
		return nil, fmt.Errorf("search conversation collection %s: %w", codebase.CollectionName, err)
	}
	manager.noteDependencyHealthy()
	return chunks, nil
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
		// A delete removes exactly one conversation and never runs the
		// manifest-absence branch, so Absence is unused here; set it explicitly to
		// absenceRetain (also the zero value) to satisfy exhaustruct.
		Absence: absenceRetain,
		// A delete never re-examines documents; it carries none.
		Reexamine: false,
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
		emptyAdmissionBudget,
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
	// Pair the record write with one observer signal so no saveLocked path skips
	// invalidation; for a document collection it is a no-op delete.
	manager.observer.Invalidate(current.ID)
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

	if manager.semantic == nil || !manager.semantic.Available() {
		manager.updateJobFailed(ctx, job.ID, semantic.ErrUnavailable)
		return
	}

	switch payload.Kind {
	case conversationJobKindDelete:
		manager.runConversationDelete(ctx, job, payload)
	case conversationJobKindUpsert:
		source := newConversationItemSource(payload.CollectionName, payload.Manifest, payload.Documents, manager.semantic, payload.Absence, payload.Reexamine)
		source.derivedVersions = loadConversationDerivedMarkers(
			conversationDerivedMarkerPath(manager.merklePath(job.CodebaseID)),
		)
		// The second return is the code path's graph-index task; a conversation
		// collection never produces one, so there is nothing to discard here.
		if handled, _ := manager.runDeltaSync(ctx, job, source); handled {
			return
		}
		_ = manager.runBootstrap(ctx, job, source)
	default:
		manager.updateJobFailed(ctx, job.ID, fmt.Errorf("unknown conversation job kind %s", payload.Kind))
	}
}

// runConversationDelete drops one conversation's rows from the live collection,
// then marks the job complete. It does not touch the merkle checkpoint: a later
// manifest sync that omits the id converges the same removal idempotently.
func (manager *Manager) runConversationDelete(ctx context.Context, job model.Job, payload conversationJobPayload) {
	select {
	case <-ctx.Done():
		manager.updateJobCancelled(ctx, job.ID)
		return
	default:
	}

	if manager.semantic == nil || !manager.semantic.Available() {
		manager.updateJobFailed(ctx, job.ID, semantic.ErrUnavailable)
		return
	}
	if err := manager.semantic.DeleteConversation(ctx, payload.CollectionName, payload.ConversationID); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
			manager.updateJobCancelled(ctx, job.ID)
			return
		}
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
	// Pair the record write with one observer signal so no saveLocked path skips
	// invalidation; for a document collection it is a no-op delete.
	manager.observer.Invalidate(codebase.ID)
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
		for _, tool := range document.Tools {
			hasher.Write([]byte(tool.Name))
			hasher.Write([]byte{0})
			hasher.Write([]byte(tool.InputJSON))
			hasher.Write([]byte{0})
			hasher.Write([]byte(tool.Command))
			hasher.Write([]byte{0})
			hasher.Write([]byte(tool.LangHint))
			hasher.Write([]byte{0})
			hasher.Write([]byte(tool.Output))
			hasher.Write([]byte{0})
			hasher.Write([]byte(strconv.FormatBool(tool.IsError)))
			hasher.Write([]byte{0})
		}
		hasher.Write([]byte(document.Thinking))
		hasher.Write([]byte{0})
	}
	return hex.EncodeToString(hasher.Sum(nil))
}

func conversationDocumentsToStoredChunks(ctx context.Context, documents []model.ConversationDocument) ([]model.StoredChunk, error) {
	dispatcher := newConversationToolDispatcher()
	chunks := make([]model.StoredChunk, 0, len(documents))
	for _, document := range documents {
		conversationID := strings.TrimSpace(document.ConversationID)
		if conversationID == "" {
			return nil, errors.New("conversation id is required")
		}
		parentConversationID := strings.TrimSpace(document.ParentConversationID)
		pieces := splitConversationText(document.Text)
		for partIndex, piece := range pieces {
			chunks = append(chunks, newConversationStoredChunk(
				document,
				conversationID,
				parentConversationID,
				conversationRelativePath(conversationID, document.MessageIndex, partIndex, len(pieces) > 1),
				piece,
				"",
				0,
				0,
			))
		}
		for toolIndex, toolCall := range document.Tools {
			toolBasePath := conversationToolCallPath(conversationID, document.MessageIndex, toolIndex)
			chunks = append(chunks, splitConversationDerivedContent(
				document,
				conversationID,
				parentConversationID,
				toolBasePath+"/tok",
				conversationToolTokenContent(toolCall),
			)...)
			if toolCall.Command != "" {
				chunks = append(chunks, splitConversationDerivedContent(
					document,
					conversationID,
					parentConversationID,
					toolBasePath+"/cmd",
					toolCall.Command,
				)...)
			}
			extension := conversationToolExtension(toolCall.LangHint)
			if toolCall.InputJSON != "" {
				inputChunks, err := splitConversationToolPayload(ctx, dispatcher, document, conversationID, parentConversationID, toolBasePath+"/in", "tool"+extension, toolCall.InputJSON)
				if err != nil {
					return nil, err
				}
				chunks = append(chunks, inputChunks...)
			}
			if toolCall.Output != "" {
				outputChunks, err := splitConversationToolPayload(ctx, dispatcher, document, conversationID, parentConversationID, toolBasePath+"/out", "tool"+extension, toolCall.Output)
				if err != nil {
					return nil, err
				}
				chunks = append(chunks, outputChunks...)
			}
		}
		if document.Thinking != "" {
			chunks = append(chunks, splitConversationDerivedContent(
				document,
				conversationID,
				parentConversationID,
				conversationThinkingPath(conversationID, document.MessageIndex),
				document.Thinking,
			)...)
		}
	}
	return chunks, nil
}

type conversationMessageDiff struct {
	documents       []model.ConversationDocument
	removalPaths    []string
	removalPrefixes []string
}

// diffConversationMessages treats a delivered message as unchanged only when
// the stored role equals document.Role and the assembled stored text equals
// document.Text, which is also the text conversationDocumentsToStoredChunks
// stores after multipart splitting. Stale stored indices must be deleted here
// because the conversation source uses absenceRetain, so an absent message row
// would otherwise survive forever once the conversation fingerprint advances.
// New messages also carry exact removals because legacy rows without
// messageIndex are invisible to storedState but still live under the same
// conv/<id>/<message> paths; genuinely new messages pay one no-op delete.
func diffConversationMessages(ctx context.Context, conversationID string, documents []model.ConversationDocument, storedState map[int32]semantic.StoredMessageState, reuse map[string][]float32) (conversationMessageDiff, error) {
	diff := conversationMessageDiff{
		documents:       make([]model.ConversationDocument, 0, len(documents)),
		removalPaths:    make([]string, 0),
		removalPrefixes: make([]string, 0),
	}
	delivered := make(map[int32]struct{}, len(documents))
	for _, document := range documents {
		delivered[document.MessageIndex] = struct{}{}
		stored, found := storedState[document.MessageIndex]
		if found {
			matches, err := conversationDocumentMatchesStored(ctx, document, stored, reuse)
			if err != nil {
				return diff, err
			}
			if matches {
				continue
			}
		}
		diff.documents = append(diff.documents, document)
		diff.addRemoval(conversationID, document.MessageIndex)
	}

	staleIndexes := make([]int32, 0)
	for messageIndex := range storedState {
		if _, found := delivered[messageIndex]; found {
			continue
		}
		staleIndexes = append(staleIndexes, messageIndex)
	}
	slices.Sort(staleIndexes)
	for _, staleIndex := range staleIndexes {
		diff.addRemoval(conversationID, staleIndex)
	}
	return diff, nil
}

func conversationDocumentMatchesStored(ctx context.Context, document model.ConversationDocument, stored semantic.StoredMessageState, reuse map[string][]float32) (bool, error) {
	if stored.Role != document.Role || stored.Text != document.Text {
		return false, nil
	}
	documentHasDerived := len(document.Tools) > 0 || document.Thinking != ""
	if !stored.HasDerivedContent && !documentHasDerived {
		return true, nil
	}
	if !documentHasDerived {
		return false, nil
	}
	chunks, err := conversationDocumentsToStoredChunks(ctx, []model.ConversationDocument{document})
	if err != nil {
		return false, err
	}
	currentHasDerived := false
	for _, chunk := range chunks {
		if !isDerivedConversationChunk(chunk) {
			continue
		}
		currentHasDerived = true
		if _, found := reuse[semantic.ContentVectorKey(chunk.Content)]; !found {
			return false, nil
		}
	}
	if stored.HasDerivedContent != currentHasDerived {
		return false, nil
	}
	return true, nil
}

func isDerivedConversationChunk(chunk model.StoredChunk) bool {
	return strings.HasPrefix(chunk.RelativePath, "convtool/") || strings.HasPrefix(chunk.RelativePath, "convthink/")
}

// addRemoval emits both the exact path and the slash-suffixed prefix for one
// message. The pair is load-bearing: the exact path deletes a single-part
// row, the slash prefix deletes multipart part rows across shape transitions,
// and a bare prefix without the slash would like-match sibling indices
// (conv/x/12 matches conv/x/120).
func (diff *conversationMessageDiff) addRemoval(conversationID string, messageIndex int32) {
	relativePath := conversationRelativePath(conversationID, messageIndex, 0, false)
	toolPath := conversationToolMessagePath(conversationID, messageIndex)
	thinkingPath := conversationThinkingPath(conversationID, messageIndex)
	diff.removalPaths = append(diff.removalPaths, relativePath, toolPath, thinkingPath)
	diff.removalPrefixes = append(diff.removalPrefixes, relativePath+"/", toolPath+"/", thinkingPath+"/")
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
