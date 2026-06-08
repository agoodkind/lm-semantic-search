package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"unicode/utf8"

	"goodkind.io/gklog/correlation"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/metrics"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/spans"
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

type conversationJobPayload struct {
	Kind           conversationJobKind
	CollectionName string
	Chunks         []model.StoredChunk
	DocumentCount  int
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

// UpsertConversationDocuments queues an asynchronous ingest for pre-chunked
// conversation documents in a virtual document collection.
func (manager *Manager) UpsertConversationDocuments(ctx context.Context, collectionID string, documents []model.ConversationDocument) (model.Job, error) {
	return manager.upsertConversationDocuments(ctx, collectionID, documents, model.ClientInfo{Name: "", PID: 0})
}

func (manager *Manager) upsertConversationDocuments(ctx context.Context, collectionID string, documents []model.ConversationDocument, client model.ClientInfo) (model.Job, error) {
	chunks, err := conversationDocumentsToStoredChunks(documents)
	if err != nil {
		return model.Job{}, err
	}
	codebase, err := manager.RegisterConversationCollection(ctx, collectionID)
	if err != nil {
		return model.Job{}, err
	}
	payload := conversationJobPayload{
		Kind:           conversationJobKindUpsert,
		CollectionName: codebase.CollectionName,
		Chunks:         chunks,
		DocumentCount:  len(documents),
		ConversationID: "",
	}
	return manager.queueConversationJob(ctx, codebase, client, payload)
}

// DeleteConversation queues an asynchronous delete for one conversation id.
func (manager *Manager) DeleteConversation(ctx context.Context, collectionID string, conversationID string) (model.Job, error) {
	return manager.deleteConversation(ctx, collectionID, conversationID, model.ClientInfo{Name: "", PID: 0})
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
		Chunks:         nil,
		DocumentCount:  0,
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

	manager.updateConversationJobProgress(job.ID, payload)
	var err error
	switch payload.Kind {
	case conversationJobKindUpsert:
		if manager.semantic != nil {
			err = manager.semantic.UpsertConversationChunks(ctx, payload.CollectionName, payload.Chunks)
		}
	case conversationJobKindDelete:
		if manager.semantic != nil {
			err = manager.semantic.DeleteConversation(ctx, payload.CollectionName, payload.ConversationID)
		}
	default:
		err = fmt.Errorf("unknown conversation job kind %s", payload.Kind)
	}
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
			manager.updateJobCancelled(ctx, job.ID)
			return
		}
		manager.updateJobFailed(ctx, job.ID, err)
		return
	}
	manager.updateConversationJobCompleted(job.ID, payload)
}

func (manager *Manager) conversationJobPayload(jobID string) (conversationJobPayload, bool) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	payload, found := manager.conversationJobs[jobID]
	return payload, found
}

func (manager *Manager) updateConversationJobProgress(jobID string, payload conversationJobPayload) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	job, found := manager.jobs[jobID]
	if !found {
		return
	}
	if job.State != model.JobStateQueued && job.State != model.JobStateRunning {
		return
	}

	now := clock.Now()
	job.State = model.JobStateRunning
	job.UpdatedAt = now
	job.Progress.Phase = conversationJobPhase(payload)
	job.Progress.FilesTotal = safeInt32(payload.DocumentCount)
	job.Progress.FilesProcessed = 0
	job.Progress.ChunksGenerated = safeInt32(len(payload.Chunks))
	job.Progress.CollectionRowsWritten = 0
	job.Progress.LastEventAt = now
	job.Progress.HeartbeatAt = now
	manager.jobs[jobID] = job
}

func (manager *Manager) updateConversationJobCompleted(jobID string, payload conversationJobPayload) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	job, found := manager.jobs[jobID]
	if !found {
		delete(manager.conversationJobs, jobID)
		return
	}

	now := clock.Now()
	metrics.JobCompleted()
	if payload.Kind == conversationJobKindUpsert && len(payload.Chunks) > 0 {
		manager.noteDependencyHealthyLocked()
	}
	job.State = model.JobStateCompleted
	job.UpdatedAt = now
	job.CompletedAt = &now
	job.Progress.Phase = "completed"
	job.Progress.OverallPercent = 100
	job.Progress.FilesProcessed = safeInt32(payload.DocumentCount)
	job.Progress.FilesTotal = safeInt32(payload.DocumentCount)
	job.Progress.ChunksGenerated = safeInt32(len(payload.Chunks))
	job.Progress.CollectionRowsWritten = safeInt32(len(payload.Chunks))
	job.Progress.LastEventAt = now
	job.Progress.HeartbeatAt = now
	delete(manager.conversationJobs, jobID)
	if err := manager.appendJobLocked("job_completed", job); err != nil {
		slog.Error("append completed conversation job event failed", "job_id", jobID, "err", err)
	}

	codebase, found := manager.codebases[job.CodebaseID]
	if !found {
		return
	}
	codebase.Status = model.CodebaseStatusIndexed
	codebase.ActiveJobID = ""
	codebase.LastSuccessfulRun = &model.IndexRunSummary{
		IndexedFiles: safeInt32(payload.DocumentCount),
		TotalChunks:  safeInt32(len(payload.Chunks)),
		Status:       "completed",
		CompletedAt:  now,
		SkippedFiles: nil,
	}
	codebase.UpdatedAt = now
	manager.codebases[codebase.ID] = codebase
	if err := manager.saveLocked(); err != nil {
		slog.Error("write registry after completed conversation job failed", "job_id", jobID, "err", err)
	}
}

func conversationJobPhase(payload conversationJobPayload) string {
	if payload.Kind == conversationJobKindDelete {
		return "Deleting conversation documents..."
	}
	return "Ingesting conversation documents..."
}

func conversationDocumentsToStoredChunks(documents []model.ConversationDocument) ([]model.StoredChunk, error) {
	chunks := make([]model.StoredChunk, 0, len(documents))
	for _, document := range documents {
		conversationID := strings.TrimSpace(document.ConversationID)
		if conversationID == "" {
			return nil, errors.New("conversation id is required")
		}
		pieces := splitConversationText(document.Text)
		for partIndex, piece := range pieces {
			chunks = append(chunks, model.StoredChunk{
				Content:        piece,
				RelativePath:   conversationRelativePath(conversationID, document.MessageIndex, partIndex, len(pieces) > 1),
				StartLine:      0,
				EndLine:        0,
				Language:       "",
				FileExtension:  "",
				ConversationID: conversationID,
				MessageIndex:   document.MessageIndex,
				Role:           document.Role,
				TimestampUnix:  document.TimestampUnix,
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
