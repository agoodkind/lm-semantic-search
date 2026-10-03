package daemon

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strconv"
	"strings"

	"goodkind.io/gklog/correlation"
	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/spans"
)

const (
	conversationCanonicalPathPrefix = "chat:///"
	conversationChunkMaxBytes       = 60000
)

// resolveConversationChunkBudget picks the effective byte cap for splitting
// conversation text. The budget flows in from the owning manager as an optional
// argument; when absent or non-positive it defaults to the varchar-safe cap. The
// optional shape keeps the many direct test call sites unchanged while the
// production path passes the manager's per-run budget explicitly.
func resolveConversationChunkBudget(chunkByteBudget []int) int {
	if len(chunkByteBudget) > 0 && chunkByteBudget[0] > 0 {
		return chunkByteBudget[0]
	}
	return conversationChunkMaxBytes
}

type conversationJobKind string

const (
	conversationJobKindUpsert conversationJobKind = "upsert"
	conversationJobKindDelete conversationJobKind = "delete"
)

// conversationJobPayload is the work of one document collection job. An upsert
// lists the full manifest (every item id with its content fingerprint) and the
// content delivered for the changed ids: conversation documents from the
// conversation RPC, or validated client rows from the generic RPC. The shared
// routine diffs the manifest against the stored checkpoint and embeds only the
// changed items. A delete removes the rows of the one item in ItemID: a
// conversation id from the conversation RPC, or a client item id from the
// generic RPC.
type conversationJobPayload struct {
	Kind           conversationJobKind
	CollectionName string
	Manifest       map[string]string
	Documents      []model.ConversationDocument
	Rows           []collectionRow
	ItemID         string
	// Absence is the upsert's caller-declared policy for a conversation the
	// manifest omits. It is meaningful only for an upsert; a delete sets it
	// explicitly to absenceRetain (also the zero value) but never consults it.
	Absence absencePolicy
	// Backfill forces delivered conversations whose expected derived rows are
	// ABSENT into this run's changed set even when their fingerprints are
	// unchanged, and skips conversations whose derived rows are all present. It is
	// presence-based and meaningful only for an upsert; the normal sync leaves it
	// false.
	Backfill bool
	// Force rebuilds EVERY delivered conversation regardless of presence, with
	// vector reuse disabled, so present rows re-embed. It is meaningful only for an
	// upsert and stays false for the normal sync. When both flags are set, Force
	// wins.
	Force bool
}

// SyncConversationManifest diffs clyde's full conversation manifest against the
// stored checkpoint and returns the ids the engine needs: the conversations new
// or changed since the last successful ingest. clyde then sends documents for
// only those ids. The engine owns drift, so clyde keeps no change-tracking
// state and a slow first embed runs exactly once.
func (manager *Manager) SyncConversationManifest(ctx context.Context, collectionID string, manifest map[string]string) ([]string, error) {
	codebase, err := manager.resolveConversationCollection(ctx, collectionID)
	if err != nil {
		return nil, err
	}
	return manager.syncCollectionManifest(ctx, codebase, manifest), nil
}

// syncCollectionManifest diffs a document collection's manifest against its
// stored checkpoint and returns the new and changed item ids, capped per ingest
// by capNeededConversations with the collection's rotation cursor.
func (manager *Manager) syncCollectionManifest(ctx context.Context, codebase model.Codebase, manifest map[string]string) []string {
	configDigest := codebase.EffectiveConfig.IgnoreDigest
	seed := manager.loadLiveCheckpoint(ctx, codebase, configDigest).snapshot
	current := merkle.Snapshot{ConfigDigest: configDigest, Files: manifest, Inodes: nil}
	diff := merkle.DiffSnapshots(seed, current)

	manager.mu.Lock()
	cursor := manager.conversationSyncCursors[codebase.ID]
	needed, nextCursor := capNeededConversations(diff.Added, diff.Modified, manager.config.MaxConversationsPerIngest, cursor)
	if nextCursor != "" {
		manager.conversationSyncCursors[codebase.ID] = nextCursor
	}
	manager.mu.Unlock()
	return needed
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

// upsertConversationDocuments queues an asynchronous ingest through the
// generic document collection path. When manifest is nil it is derived from
// the delivered documents with fingerprintConversationDocuments. A caller that
// hands over a complete set then need not compute fingerprints itself.
func (manager *Manager) upsertConversationDocuments(ctx context.Context, collectionID string, documents []model.ConversationDocument, manifest map[string]string, client model.ClientInfo, absence absencePolicy, backfill bool, force bool) (model.Job, error) {
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
	codebase, err := manager.resolveConversationCollection(ctx, collectionID)
	if err != nil {
		return model.Job{}, err
	}
	return manager.queueCollectionUpsert(ctx, codebase, client, collectionUpsert{
		Manifest:  manifest,
		Documents: documents,
		Rows:      nil,
		Absence:   absence,
		Backfill:  backfill,
		Force:     force,
	})
}

// collectionUpsert is one delivery into a document collection: the manifest
// and either conversation documents or validated client rows.
type collectionUpsert struct {
	Manifest  map[string]string
	Documents []model.ConversationDocument
	Rows      []collectionRow
	Absence   absencePolicy
	Backfill  bool
	Force     bool
}

// queueCollectionUpsert queues the asynchronous ingest of one delivery into a
// registered document collection. Both the conversation RPC and the generic
// item RPC queue their upserts here.
func (manager *Manager) queueCollectionUpsert(ctx context.Context, codebase model.Codebase, client model.ClientInfo, upsert collectionUpsert) (model.Job, error) {
	payload := conversationJobPayload{
		Kind:           conversationJobKindUpsert,
		CollectionName: codebase.CollectionName,
		Manifest:       upsert.Manifest,
		Documents:      upsert.Documents,
		Rows:           upsert.Rows,
		ItemID:         "",
		Absence:        upsert.Absence,
		Backfill:       upsert.Backfill,
		Force:          upsert.Force,
	}
	return manager.queueConversationJob(ctx, codebase, client, payload)
}

// DeleteConversation queues an asynchronous delete for one conversation id.
func (manager *Manager) DeleteConversation(ctx context.Context, collectionID string, conversationID string) (model.Job, error) {
	return manager.deleteConversation(ctx, collectionID, conversationID, model.ClientInfo{Name: "", PID: 0})
}

// SearchConversations searches a registered virtual conversation collection
// through the generic collection search. It converts the conversation filter
// to the typed filter tree, and a per-conversation limit becomes a per-group
// cap on conversationId. An unregistered collection returns no results and
// registers nothing.
func (manager *Manager) SearchConversations(ctx context.Context, collectionID string, query string, limit int32, filter conversationSearchFilter, perConversationLimit int32) ([]model.StoredChunk, error) {
	if refusal := manager.maintenanceRefusal(); refusal != nil {
		return nil, refusal
	}
	trimmedCollectionID := strings.TrimSpace(collectionID)

	manager.mu.Lock()
	codebase, found := manager.findConversationCollectionLocked(trimmedCollectionID)
	manager.mu.Unlock()
	if !found {
		return nil, nil
	}
	hits, err := manager.searchRegisteredCollection(ctx, trimmedCollectionID, codebase, filter.collectionSearchRequest(trimmedCollectionID, query, limit, perConversationLimit))
	if err != nil {
		return nil, err
	}
	return collectionHitChunks(hits), nil
}

// SearchWithinConversation retrieves one conversation's matching rows plus the
// content fingerprint the engine has embedded for it. It registers the
// collection first, scopes the generic collection search to the one
// conversation id, and reads the fingerprint through
// [Manager.CollectionItemState]. An empty fingerprint means the conversation
// is not indexed; a fingerprint differing from the conversation's current one
// means the index trails the transcript. Either way the caller decides whether
// to refresh newer content.
func (manager *Manager) SearchWithinConversation(ctx context.Context, collectionID string, conversationID string, query string, limit int32, filter conversationSearchFilter) ([]model.StoredChunk, string, error) {
	trimmedConversationID := strings.TrimSpace(conversationID)
	if trimmedConversationID == "" {
		return nil, "", errors.New("conversation id is required")
	}
	if refusal := manager.maintenanceRefusal(); refusal != nil {
		return nil, "", refusal
	}
	codebase, err := manager.resolveConversationCollection(ctx, collectionID)
	if err != nil {
		return nil, "", err
	}
	trimmedCollectionID := strings.TrimSpace(collectionID)
	filter.ConversationIDs = []string{trimmedConversationID}
	hits, err := manager.searchRegisteredCollection(ctx, trimmedCollectionID, codebase, filter.collectionSearchRequest(trimmedCollectionID, query, limit, 0))
	if err != nil {
		return nil, "", err
	}
	fingerprint, err := manager.CollectionItemState(ctx, trimmedCollectionID, trimmedConversationID)
	if err != nil {
		return nil, "", err
	}
	return collectionHitChunks(hits), fingerprint, nil
}

// collectionHitChunks returns the stored chunk of every hit, in order. The
// conversation RPCs build their response fields from the chunk's decoded
// metadata.
func collectionHitChunks(hits []semantic.CollectionHit) []model.StoredChunk {
	chunks := make([]model.StoredChunk, 0, len(hits))
	for _, hit := range hits {
		chunks = append(chunks, hit.Chunk)
	}
	return chunks
}

// backfillConversationScalars fills workspaceRoot and archived on the rows of a
// conversation collection through the generic scalar backfill. values maps a
// conversation id to its workspaceRoot and archived values. The collection
// resolves the way every conversation RPC resolves it.
func (manager *Manager) backfillConversationScalars(ctx context.Context, collectionID string, values map[string]map[string]collection.ScalarValue, dryRun bool) (int, int, error) {
	codebase, err := manager.resolveConversationCollection(ctx, collectionID)
	if err != nil {
		return 0, 0, err
	}
	declaration := semantic.ConversationDeclaration()
	return manager.runScalarBackfill(ctx, codebase, collection.ScalarBackfill{
		ItemColumn:         declaration.ItemIDColumn,
		Columns:            declaredColumnsNamed(declaration, semantic.ConversationWorkspaceRootColumn, semantic.ConversationArchivedColumn),
		Values:             values,
		LegacyPathFamilies: semantic.ConversationLegacyPathFamilies,
		DryRun:             dryRun,
	})
}

// deleteConversation queues the removal of one conversation's rows through the
// generic item delete. The collection resolves the way every conversation RPC
// resolves it.
func (manager *Manager) deleteConversation(ctx context.Context, collectionID string, conversationID string, client model.ClientInfo) (model.Job, error) {
	trimmedConversationID := strings.TrimSpace(conversationID)
	if trimmedConversationID == "" {
		return model.Job{}, errors.New("conversation id is required")
	}
	codebase, err := manager.resolveConversationCollection(ctx, collectionID)
	if err != nil {
		return model.Job{}, err
	}
	return manager.queueItemDelete(ctx, codebase, trimmedConversationID, client)
}

func (manager *Manager) queueConversationJob(ctx context.Context, codebase model.Codebase, client model.ClientInfo, payload conversationJobPayload) (model.Job, error) {
	manager.policyMutationMutex.Lock()
	policyLocked := true
	defer func() {
		if policyLocked {
			manager.policyMutationMutex.Unlock()
		}
	}()

	var emptyJob model.Job

	manager.mu.Lock()
	current, found := manager.codebases[codebase.ID]
	if !found {
		manager.mu.Unlock()
		return emptyJob, fmt.Errorf("conversation collection not tracked: %s", codebase.CanonicalPath)
	}
	activeJob, active, err := manager.activeConversationJobLocked(current)
	if err != nil {
		manager.mu.Unlock()
		return emptyJob, err
	}
	if active {
		// Coalesce an upsert onto the depth-1 pending slot instead of refusing, so a
		// backfill and a normal ingest on the same collection do not contend; the slot
		// drains into a fresh job on the active job's terminal transition. A delete
		// cannot losslessly fold into the single upsert-shaped slot, so it keeps the
		// refuse behavior.
		if payload.Kind == conversationJobKindUpsert {
			manager.mergePendingConversationPayloadLocked(current.ID, payload)
			manager.mu.Unlock()
			return activeJob, nil
		}
		manager.mu.Unlock()
		return emptyJob, adapterr.NewActiveJobConflict(activeJob.ID, fmt.Sprintf("conflicting active job %s for conversation collection %s", activeJob.ID, current.CanonicalPath))
	}

	job, err := manager.enqueueConversationJobLocked(current, client, payload)
	if err != nil {
		manager.mu.Unlock()
		return emptyJob, err
	}
	manager.mu.Unlock()

	ctx = spans.Attach(
		ctx,
		correlation.IdentityAttribute{Key: "job_id", Value: job.ID},
		correlation.IdentityAttribute{Key: "codebase_id", Value: current.ID},
	)
	manager.policyMutationMutex.Unlock()
	policyLocked = false
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
	case model.JobStateQueued, model.JobStateRunning, model.JobStatePaused, model.JobStateCancelling:
		return activeJob, true, nil
	default:
		return emptyJob, false, fmt.Errorf("unknown job state %s for active job %s", activeJob.State, activeJob.ID)
	}
}

// runConversationIngest runs one document collection job. An upsert runs the
// same delta-then-bootstrap routine code uses, with the collection item source
// that documentItemSource builds from the saved declaration. A delete drops one
// item's rows.
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
		source := manager.documentItemSource(job.CodebaseID, payload)
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

// runConversationDelete drops one item's rows from the live collection, then
// marks the job complete. The saved declaration selects the rows: its item id
// column, plus the legacy conversation path prefixes for the conversation
// declaration. The delete leaves the merkle checkpoint unchanged. A later
// manifest sync that omits the id converges the checkpoint.
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
	removal := manager.itemSelector(job.CodebaseID).removal([]string{payload.ItemID})
	if err := manager.semantic.DeleteItemRows(ctx, payload.CollectionName, removal); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
			manager.updateJobCancelled(ctx, job.ID)
			return
		}
		manager.updateJobFailed(ctx, job.ID, err)
		return
	}
	manager.finishConversationDelete(ctx, job.ID)
}

func (manager *Manager) finishConversationDelete(ctx context.Context, jobID string) {
	manager.policyMutationMutex.Lock()
	defer manager.policyMutationMutex.Unlock()

	manager.transitionMutex.Lock()
	manager.mu.Lock()
	job, found := manager.jobs[jobID]
	if !found || isTerminalJobState(job.State) {
		delete(manager.conversationJobs, jobID)
		manager.mu.Unlock()
		manager.transitionMutex.Unlock()
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
	jobEvent := model.JobEvent{Event: "job_completed", OccurredAt: clock.Now(), Job: job}
	manager.mu.Unlock()
	journalErr := manager.writeJobTransition(jobEvent)
	manager.mu.Lock()
	manager.jobs[jobID] = job
	if journalErr != nil {
		slog.ErrorContext(ctx, "append completed conversation delete event failed", "job_id", jobID, "err", journalErr)
	}
	delete(manager.conversationJobs, jobID)
	codebase, found := manager.codebases[job.CodebaseID]
	if !found {
		manager.mu.Unlock()
		manager.transitionMutex.Unlock()
		return
	}
	// Clear ActiveJobID only when it still points at this job, so a raced or
	// duplicate terminal transition never clobbers a drained successor.
	codebase.Status = model.CodebaseStatusIndexed
	if codebase.ActiveJobID == jobID {
		codebase.ActiveJobID = ""
	}
	codebase.UpdatedAt = now
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		slog.ErrorContext(ctx, "write registry after conversation delete failed", "job_id", jobID, "err", err)
	}
	// Pair the record write with one observer signal so no saveLocked path skips
	// invalidation; for a document collection it is a no-op delete.
	manager.observer.Invalidate(codebase.ID)
	// drainPendingJobLocked no-ops unless ActiveJobID was cleared above, so a raced
	// transition that did not own the slot never drains a duplicate.
	drainedJobID, drained := manager.drainPendingJobLocked(ctx, codebase.ID)
	codebaseID := codebase.ID
	manager.mu.Unlock()
	manager.transitionMutex.Unlock()
	if drained {
		manager.runDrainedJob(ctx, codebaseID, drainedJobID)
	}
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
			hasher.Write([]byte(tool.Display))
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

// conversationDocumentsToStoredChunks is the single derived-chunk regeneration
// entry point: every path that turns delivered documents into stored chunks
// (both the per-item indexOne loop and the full-conversation fallback) routes
// through it. It is a package var rather than a plain func only so a same-package
// test can wrap it to count regenerations and lock the chokepoint invariant that
// the up-front presence classifier (forcedWorkSet) regenerates nothing.
// Production never reassigns it.
var conversationDocumentsToStoredChunks = func(_ context.Context, documents []model.ConversationDocument, chunkByteBudget ...int) ([]model.StoredChunk, error) {
	budget := resolveConversationChunkBudget(chunkByteBudget)
	chunks := make([]model.StoredChunk, 0, len(documents))
	for _, document := range documents {
		conversationID := strings.TrimSpace(document.ConversationID)
		if conversationID == "" {
			return nil, errors.New("conversation id is required")
		}
		parentConversationID := strings.TrimSpace(document.ParentConversationID)
		chunks = appendStorableConversationField(
			chunks,
			document.Text,
			budget,
			func(piece string, partIndex int, multipart bool) model.StoredChunk {
				return newConversationStoredChunk(
					document,
					conversationID,
					parentConversationID,
					conversationRelativePath(conversationID, document.MessageIndex, partIndex, multipart),
					piece,
					"",
					0,
					0,
				)
			},
		)
		for toolIndex, toolCall := range document.Tools {
			toolBasePath := conversationToolCallPath(conversationID, document.MessageIndex, toolIndex)
			chunks = appendContinuedStorableField(
				chunks,
				conversationToolContent(toolCall),
				budget,
				strings.TrimSpace(toolCall.Name),
				func(piece string, partIndex int, multipart bool) model.StoredChunk {
					relativePath := toolBasePath
					if multipart {
						relativePath = fmt.Sprintf("%s/%d", toolBasePath, partIndex)
					}
					return newConversationStoredChunk(
						document,
						conversationID,
						parentConversationID,
						relativePath,
						piece,
						"",
						0,
						0,
					)
				},
			)
		}
		chunks = append(chunks, splitConversationDerivedContent(
			document,
			conversationID,
			parentConversationID,
			conversationThinkingPath(conversationID, document.MessageIndex),
			document.Thinking,
			budget,
		)...)
	}
	return chunks, nil
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
