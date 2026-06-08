package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
	"time"

	"goodkind.io/gklog/correlation"
	"goodkind.io/gklog/version"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/pbconv"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// toDependencyHealth converts the daemon's cached shared-dependency health into
// its protobuf form for JSON consumers. Timestamps stay UTC on the wire and are
// omitted while zero, so a never-degraded record carries no since stamp.
func toDependencyHealth(health dependencyHealth) *pb.DependencyHealth {
	result := &pb.DependencyHealth{
		Degraded:      health.Degraded(),
		Mode:          string(health.Mode),
		Since:         nil,
		LastHealthyAt: nil,
	}
	if !health.Since.IsZero() {
		result.Since = timestamppb.New(health.Since)
	}
	if !health.LastHealthyAt.IsZero() {
		result.LastHealthyAt = timestamppb.New(health.LastHealthyAt)
	}
	return result
}

// appendCorrelationRef prefixes one compact diagnostics line to a display
// text so every successful response starts with a greppable correlation
// header. Extras are key/value pairs for ids the trace context does not
// already carry, such as codebase_id and job_id.
func appendCorrelationRef(displayText string, ctx context.Context, extras ...string) string {
	corr := correlation.FromContext(ctx)
	line := correlation.HeaderLine(corr, extras...)
	if line == "" {
		return displayText
	}
	if strings.TrimSpace(displayText) == "" {
		return line
	}
	return line + "\n" + displayText
}

// envelopeText composes the human-facing display text for a read surface as the
// shared envelope: the dependency-health banner (only when a shared dependency is
// degraded), then the correlation header, then the body, joined with single
// newlines. It is the one place the banner is prepended, so every surface shows
// exactly one banner and the body renderers never carry it. The caller passes the
// health snapshot it already read so the banner and the body agree.
func (server *GRPCServer) envelopeText(ctx context.Context, health dependencyHealth, body string, extras ...string) string {
	withHeader := appendCorrelationRef(body, ctx, extras...)
	banner := renderHealthBanner(health, server.manager.config)
	if banner == "" {
		return withHeader
	}
	if strings.TrimSpace(withHeader) == "" {
		return banner
	}
	return banner + "\n" + withHeader
}

// jobIDOf returns the id of job or "" when job is nil, so callers can fold
// optional job ids into appendCorrelationRef without nil checks.
func jobIDOf(job *model.Job) string {
	if job == nil {
		return ""
	}
	return job.ID
}

// codebaseIDOf returns the id of codebase or "" when found is false.
func codebaseIDOf(found bool, codebase model.Codebase) string {
	if !found {
		return ""
	}
	return codebase.ID
}

// beginRPC opens the per-RPC correlation span, emits daemon.rpc.started,
// and returns the derived context plus a deferred completion function
// that recovers panics through [adapterr.Respond] and emits
// daemon.rpc.completed with status code and duration.
func beginRPC(ctx context.Context, method string) (context.Context, func(*error)) {
	corr := correlation.FromIncomingMetadata(ctx).Child()
	ctx = correlation.WithContext(ctx, corr)
	started := clock.Now()
	slog.InfoContext(ctx, "daemon.rpc.started", "method", method)
	return ctx, func(errPtr *error) {
		if recovered := recover(); recovered != nil {
			*errPtr = status.Error(adapterr.Respond(ctx, adapterr.NewInternal("daemon panic", fmt.Errorf("panic: %v", recovered))))
		}
		logRPCDone(ctx, method, started, *errPtr)
	}
}

func logRPCDone(ctx context.Context, method string, started time.Time, err error) {
	level := slog.LevelInfo
	code := "OK"
	if err != nil {
		level = slog.LevelWarn
		code = status.Code(err).String()
	}
	slog.LogAttrs(ctx, level, "daemon.rpc.completed",
		slog.String("method", method),
		slog.String("status", code),
		slog.Int64("duration_ms", clock.Now().Sub(started).Milliseconds()),
	)
}

// classifyManagerError promotes a raw manager error into a typed
// [adapterr.AdapterError] based on the message substrings the
// manager layer uses today. Unrecognized errors pass through; the
// boundary wraps them as [adapterr.ClassInternal].
func classifyManagerError(path string, err error) error {
	if err == nil {
		return nil
	}
	var adapterErr *adapterr.AdapterError
	if errors.As(err, &adapterErr) {
		return err
	}
	text := err.Error()
	switch {
	case strings.Contains(text, "conflicting active job"):
		return adapterr.NewConflictingJob(text, err)
	case strings.Contains(text, "codebase not tracked"):
		return adapterr.NewNotIndexed(path, err)
	case strings.Contains(text, "invalid file extensions in extensionFilter"):
		return adapterr.NewInvalidPath(text, err)
	case strings.Contains(text, "index data for '"):
		return adapterr.NewInvalidPath(text, err)
	}
	return err
}

// requireNonEmpty reports an InvalidArgument error when a required request
// field is empty or whitespace, so the daemon rejects a missing path, query, or
// job id loudly instead of treating it as a normal not-found. argument names the
// field for the error envelope; pathLike picks the path-specific hint over the
// generic missing-argument hint.
func requireNonEmpty(ctx context.Context, value string, argument string, pathLike bool) error {
	if strings.TrimSpace(value) != "" {
		return nil
	}
	if pathLike {
		return status.Error(adapterr.Respond(ctx, adapterr.NewInvalidPath("codebase path is required", nil)))
	}
	return status.Error(adapterr.Respond(ctx, adapterr.NewMissingArgument(argument)))
}

// GRPCServer exposes the daemon manager through the generated gRPC service.
type GRPCServer struct {
	manager  *Manager
	shutdown func()
}

// NewGRPCServer builds the daemon's gRPC service implementation.
func NewGRPCServer(manager *Manager, shutdown func()) *GRPCServer {
	return &GRPCServer{
		manager:  manager,
		shutdown: shutdown,
	}
}

// Version reports daemon build metadata.
func (server *GRPCServer) Version(ctx context.Context, request *pb.VersionRequest) (resp *pb.VersionResponse, err error) {
	ctx, done := beginRPC(ctx, "Version")
	defer done(&err)
	_ = request
	_ = ctx
	return &pb.VersionResponse{
		Version:   version.String(),
		Commit:    version.Commit,
		BuildTime: version.BuildTime,
	}, nil
}

// StartIndex registers a new indexing request with the daemon.
func (server *GRPCServer) StartIndex(ctx context.Context, request *pb.StartIndexRequest) (resp *pb.StartIndexResponse, err error) {
	ctx, done := beginRPC(ctx, "StartIndex")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetPath(), "absolutePath", true); argErr != nil {
		return nil, argErr
	}
	job, codebase, deduplicated, overlapsCodebaseID, callErr := server.manager.StartIndex(ctx, request.GetPath(), pbClient(request.GetClient()), pbconv.FromStartIndexConfig(request), request.GetForce())
	if callErr != nil {
		return nil, status.Error(adapterr.Respond(ctx, classifyManagerError(request.GetPath(), callErr)))
	}
	mergeNote := server.startIndexMergeNote(request.GetPath(), codebase)
	return &pb.StartIndexResponse{
		JobId:              job.ID,
		CodebaseId:         codebase.ID,
		State:              string(job.State),
		Deduplicated:       deduplicated,
		CanonicalPath:      codebase.CanonicalPath,
		OverlapsCodebaseId: overlapsCodebaseID,
		DisplayText:        appendCorrelationRef(renderStartIndex(request.GetPath(), codebase, job, deduplicated, overlapsCodebaseID, mergeNote), ctx, "codebase_id", codebase.ID, "job_id", job.ID),
	}, nil
}

// startIndexMergeNote describes a containment relationship the StartIndex call
// resolved: a merge-up redirect when the requested path is covered by the
// returned (larger) codebase, or a merge-down reuse note when the requested
// path roots above already-indexed sub-folders. It returns an empty string for
// an ordinary, non-overlapping index.
func (server *GRPCServer) startIndexMergeNote(requestedPath string, codebase model.Codebase) string {
	if canonical, err := canonicalizePath(requestedPath); err == nil && codebase.CanonicalPath != "" {
		coveringRoot := filepath.Clean(codebase.CanonicalPath)
		if canonical != coveringRoot && pathCovers(coveringRoot, canonical) {
			return fmt.Sprintf("🔁 '%s' is covered by the larger index '%s'; syncing that index to include this subtree instead of building a separate one.", requestedPath, codebase.CanonicalPath)
		}
	}
	descendants := server.manager.IndexedDescendants(requestedPath)
	if len(descendants) == 0 {
		return ""
	}
	names := make([]string, 0, len(descendants))
	for _, child := range descendants {
		names = append(names, child.CanonicalPath)
	}
	return "🔗 Reusing already-indexed " + plural("sub-folder", len(names)) + ": " + strings.Join(names, ", ") + " (their embeddings are merged in, not re-embedded)."
}

// ClearIndex removes a tracked codebase from daemon state.
func (server *GRPCServer) ClearIndex(ctx context.Context, request *pb.ClearIndexRequest) (resp *pb.ClearIndexResponse, err error) {
	ctx, done := beginRPC(ctx, "ClearIndex")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetPath(), "absolutePath", true); argErr != nil {
		return nil, argErr
	}
	codebase, callErr := server.manager.ClearIndex(ctx, request.GetPath(), pbClient(request.GetClient()))
	if callErr != nil {
		return nil, status.Error(adapterr.Respond(ctx, classifyManagerError(request.GetPath(), callErr)))
	}
	return &pb.ClearIndexResponse{
		CodebaseId:  codebase.ID,
		Cleared:     true,
		DisplayText: appendCorrelationRef(renderClearIndex(codebase), ctx, "codebase_id", codebase.ID),
	}, nil
}

// CancelJob cancels a tracked daemon job.
func (server *GRPCServer) CancelJob(ctx context.Context, request *pb.CancelJobRequest) (resp *pb.CancelJobResponse, err error) {
	ctx, done := beginRPC(ctx, "CancelJob")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetJobId(), "job_id", false); argErr != nil {
		return nil, argErr
	}
	job, callErr := server.manager.CancelJob(request.GetJobId())
	if callErr != nil {
		return nil, status.Error(adapterr.Respond(ctx, adapterr.NewJobNotFound(request.GetJobId())))
	}
	return &pb.CancelJobResponse{
		JobId:       job.ID,
		Cancelled:   job.State == model.JobStateCancelled,
		DisplayText: appendCorrelationRef(renderCancelJob(job), ctx, "job_id", job.ID, "codebase_id", job.CodebaseID),
	}, nil
}

// SyncIndex registers a sync request against an existing codebase.
func (server *GRPCServer) SyncIndex(ctx context.Context, request *pb.SyncIndexRequest) (resp *pb.SyncIndexResponse, err error) {
	ctx, done := beginRPC(ctx, "SyncIndex")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetPath(), "absolutePath", true); argErr != nil {
		return nil, argErr
	}
	job, codebase, deduplicated, callErr := server.manager.SyncIndex(ctx, request.GetPath(), pbClient(request.GetClient()))
	if callErr != nil {
		return nil, status.Error(adapterr.Respond(ctx, classifyManagerError(request.GetPath(), callErr)))
	}
	operation := "sync"
	if deduplicated {
		operation = job.Operation
	}
	if operation == "sync" {
		job.Operation = operation
	}
	return &pb.SyncIndexResponse{
		JobId:       job.ID,
		CodebaseId:  codebase.ID,
		State:       string(job.State),
		DisplayText: appendCorrelationRef(renderSyncIndex(codebase, job, deduplicated), ctx, "codebase_id", codebase.ID, "job_id", job.ID),
	}, nil
}

// applyDisplayTokens sets the display status plus its glyph and label on a
// protobuf codebase, so the three stay in sync from the daemon's single
// vocabulary. pbconv cannot import the daemon vocab, so the tokens are applied
// here at the boundary.
func applyDisplayTokens(pbCodebase *pb.Codebase, display displayStatus) {
	pbCodebase.DisplayStatus = string(display)
	pbCodebase.GlyphToken = glyphForDisplay(display)
	pbCodebase.StatusLabel = labelForDisplay(display)
}

// GetIndex resolves one tracked codebase whose canonical path covers the
// queried path.
func (server *GRPCServer) GetIndex(ctx context.Context, request *pb.GetIndexRequest) (resp *pb.GetIndexResponse, err error) {
	ctx, done := beginRPC(ctx, "GetIndex")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetPath(), "absolutePath", true); argErr != nil {
		return nil, argErr
	}
	codebase, activeJob, found, classification, callErr := server.manager.GetIndex(ctx, request.GetPath())
	if callErr != nil {
		return nil, status.Error(adapterr.Respond(ctx, classifyManagerError(request.GetPath(), callErr)))
	}
	if found {
		server.manager.fillLiveChunkTotal(ctx, codebase, activeJob)
	}
	var indexedDescendants []model.Codebase
	if !found {
		indexedDescendants = server.manager.IndexedDescendants(request.GetPath())
	}
	health := server.manager.DependencyHealth()
	response := &pb.GetIndexResponse{
		Tracked:          found,
		Classification:   pbconv.ToPathClassification(classification),
		DependencyHealth: toDependencyHealth(health),
		DisplayText:      server.envelopeText(ctx, health, renderGetIndex(request.GetPath(), found, codebasePointer(found, codebase), activeJob, classification, indexedDescendants, health), "codebase_id", codebaseIDOf(found, codebase), "job_id", jobIDOf(activeJob)),
	}
	if found {
		pbCodebase := pbconv.ToCodebase(codebase)
		applyDisplayTokens(pbCodebase, computeDisplayStatus(codebase, activeJob, health.Degraded()))
		response.Codebase = pbCodebase
		response.ActiveJob = pbconv.ToJobPointer(activeJob)
	}
	return response, nil
}

// ListIndexes returns all tracked codebases.
func (server *GRPCServer) ListIndexes(ctx context.Context, request *pb.ListIndexesRequest) (resp *pb.ListIndexesResponse, err error) {
	ctx, done := beginRPC(ctx, "ListIndexes")
	defer done(&err)
	_ = request
	views := server.manager.ListIndexesView()
	response := &pb.ListIndexesResponse{
		Indexes: make([]*pb.Codebase, 0, len(views)),
	}
	for _, view := range views {
		pbCodebase := pbconv.ToCodebase(view.Codebase)
		applyDisplayTokens(pbCodebase, view.Display)
		response.Indexes = append(response.Indexes, pbCodebase)
	}
	health := server.manager.DependencyHealth()
	response.DependencyHealth = toDependencyHealth(health)
	response.DisplayText = server.envelopeText(ctx, health, renderListIndexes(views))
	return response, nil
}

// GetJob resolves one tracked job by id.
func (server *GRPCServer) GetJob(ctx context.Context, request *pb.GetJobRequest) (resp *pb.GetJobResponse, err error) {
	ctx, done := beginRPC(ctx, "GetJob")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetJobId(), "job_id", false); argErr != nil {
		return nil, argErr
	}
	job, found := server.manager.GetJob(request.GetJobId())
	if !found {
		return nil, status.Error(adapterr.Respond(ctx, adapterr.NewJobNotFound(request.GetJobId())))
	}
	health := server.manager.DependencyHealth()
	return &pb.GetJobResponse{
		Job:              pbconv.ToJob(job),
		DependencyHealth: toDependencyHealth(health),
		DisplayText:      server.envelopeText(ctx, health, renderGetJob(&job, health.Degraded()), "job_id", job.ID, "codebase_id", job.CodebaseID),
	}, nil
}

// ListJobs returns all tracked jobs, optionally filtered by codebase id.
func (server *GRPCServer) ListJobs(ctx context.Context, request *pb.ListJobsRequest) (resp *pb.ListJobsResponse, err error) {
	ctx, done := beginRPC(ctx, "ListJobs")
	defer done(&err)
	_ = ctx
	jobs := server.manager.ListJobs(request.GetCodebaseId())
	response := &pb.ListJobsResponse{
		Jobs: make([]*pb.Job, 0, len(jobs)),
	}
	for _, job := range jobs {
		response.Jobs = append(response.Jobs, pbconv.ToJob(job))
	}
	health := server.manager.DependencyHealth()
	response.DependencyHealth = toDependencyHealth(health)
	response.DisplayText = server.envelopeText(ctx, health, renderListJobs(jobs), "codebase_id", request.GetCodebaseId())
	return response, nil
}

// WatchJobs streams the latest visible state for requested jobs.
func (server *GRPCServer) WatchJobs(request *pb.WatchJobsRequest, stream pb.SemanticSearchDaemonService_WatchJobsServer) (err error) {
	ctx, done := beginRPC(stream.Context(), "WatchJobs")
	defer done(&err)
	for _, jobID := range request.GetJobIds() {
		job, found := server.manager.GetJob(jobID)
		if !found {
			continue
		}
		if sendErr := stream.Send(&pb.WatchJobsResponse{Job: pbconv.ToJob(job)}); sendErr != nil {
			slog.ErrorContext(ctx, "send watch jobs event failed", "job_id", jobID, "err", sendErr)
			return status.Error(adapterr.Respond(ctx, adapterr.NewInternal("send watch jobs event for "+jobID, sendErr)))
		}
	}
	return nil
}

// SearchCode is the future search RPC surface for semantic lookups.
func (server *GRPCServer) SearchCode(ctx context.Context, request *pb.SearchCodeRequest) (resp *pb.SearchCodeResponse, err error) {
	ctx, done := beginRPC(ctx, "SearchCode")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetPath(), "absolutePath", true); argErr != nil {
		return nil, argErr
	}
	if argErr := requireNonEmpty(ctx, request.GetQuery(), "query", false); argErr != nil {
		return nil, argErr
	}
	outcome, callErr := server.manager.SearchCode(ctx, request.GetPath(), request.GetQuery(), request.GetLimit(), request.GetExtensionFilter())
	if callErr != nil {
		return nil, status.Error(adapterr.Respond(ctx, classifyManagerError(request.GetPath(), callErr)))
	}
	server.manager.fillLiveChunkTotal(ctx, outcome.Codebase, outcome.ActiveJob)
	health := server.manager.DependencyHealth()
	response := &pb.SearchCodeResponse{
		Results:          make([]*pb.SearchResult, 0, len(outcome.Results)),
		Codebase:         pbconv.ToCodebase(outcome.Codebase),
		ActiveJob:        pbconv.ToJobPointer(outcome.ActiveJob),
		DependencyHealth: toDependencyHealth(health),
		DisplayText: server.envelopeText(ctx, health, renderSearch(searchView{
			RequestedPath: request.GetPath(),
			Query:         request.GetQuery(),
			Codebase:      outcome.Codebase,
			ActiveJob:     outcome.ActiveJob,
			Results:       outcome.Results,
			StateNote:     outcome.StateNote,
		}), "codebase_id", outcome.Codebase.ID, "job_id", jobIDOf(outcome.ActiveJob)),
	}
	for _, result := range outcome.Results {
		response.Results = append(response.Results, &pb.SearchResult{
			RelativePath: result.RelativePath,
			StartLine:    result.StartLine,
			EndLine:      result.EndLine,
			Language:     result.Language,
			Score:        0,
			Content:      result.Content,
		})
	}
	return response, nil
}

// RegisterConversationCollection reserves the conversation collection RPC surface.
func (server *GRPCServer) RegisterConversationCollection(ctx context.Context, request *pb.RegisterConversationCollectionRequest) (resp *pb.RegisterConversationCollectionResponse, err error) {
	ctx, done := beginRPC(ctx, "RegisterConversationCollection")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetCollectionId(), "collection_id", false); argErr != nil {
		return nil, argErr
	}
	codebase, callErr := server.manager.RegisterConversationCollection(ctx, request.GetCollectionId())
	if callErr != nil {
		return nil, status.Error(adapterr.Respond(ctx, callErr))
	}
	return &pb.RegisterConversationCollectionResponse{
		CodebaseId:     codebase.ID,
		CollectionName: codebase.CollectionName,
		DisplayText: appendCorrelationRef(
			renderRegisterConversationCollection(request.GetCollectionId(), codebase),
			ctx,
			"codebase_id",
			codebase.ID,
		),
	}, nil
}

// UpsertConversationDocuments reserves the conversation document upsert RPC surface.
func (server *GRPCServer) UpsertConversationDocuments(ctx context.Context, request *pb.UpsertConversationDocumentsRequest) (resp *pb.UpsertConversationDocumentsResponse, err error) {
	ctx, done := beginRPC(ctx, "UpsertConversationDocuments")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetCollectionId(), "collection_id", false); argErr != nil {
		return nil, argErr
	}
	job, callErr := server.manager.upsertConversationDocuments(
		ctx,
		request.GetCollectionId(),
		pbConversationDocuments(request.GetDocuments()),
		pbClient(request.GetClient()),
	)
	if callErr != nil {
		return nil, status.Error(adapterr.Respond(ctx, classifyManagerError(request.GetCollectionId(), callErr)))
	}
	return &pb.UpsertConversationDocumentsResponse{
		JobId: job.ID,
		DisplayText: appendCorrelationRef(
			renderUpsertConversationDocuments(request.GetCollectionId(), job, len(request.GetDocuments())),
			ctx,
			"codebase_id",
			job.CodebaseID,
			"job_id",
			job.ID,
		),
	}, nil
}

// DeleteConversation reserves the conversation deletion RPC surface.
func (server *GRPCServer) DeleteConversation(ctx context.Context, request *pb.DeleteConversationRequest) (resp *pb.DeleteConversationResponse, err error) {
	ctx, done := beginRPC(ctx, "DeleteConversation")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetCollectionId(), "collection_id", false); argErr != nil {
		return nil, argErr
	}
	if argErr := requireNonEmpty(ctx, request.GetConversationId(), "conversation_id", false); argErr != nil {
		return nil, argErr
	}
	job, callErr := server.manager.deleteConversation(
		ctx,
		request.GetCollectionId(),
		request.GetConversationId(),
		pbClient(request.GetClient()),
	)
	if callErr != nil {
		return nil, status.Error(adapterr.Respond(ctx, classifyManagerError(request.GetCollectionId(), callErr)))
	}
	return &pb.DeleteConversationResponse{
		JobId: job.ID,
		DisplayText: appendCorrelationRef(
			renderDeleteConversation(request.GetCollectionId(), request.GetConversationId(), job),
			ctx,
			"codebase_id",
			job.CodebaseID,
			"job_id",
			job.ID,
		),
	}, nil
}

// SearchConversations reserves the conversation search RPC surface.
func (server *GRPCServer) SearchConversations(ctx context.Context, request *pb.SearchConversationsRequest) (resp *pb.SearchConversationsResponse, err error) {
	ctx, done := beginRPC(ctx, "SearchConversations")
	defer done(&err)
	if argErr := requireNonEmpty(ctx, request.GetCollectionId(), "collection_id", false); argErr != nil {
		return nil, argErr
	}
	if argErr := requireNonEmpty(ctx, request.GetQuery(), "query", false); argErr != nil {
		return nil, argErr
	}
	results, callErr := server.manager.SearchConversations(ctx, request.GetCollectionId(), request.GetQuery(), request.GetLimit())
	if callErr != nil {
		return nil, status.Error(adapterr.Respond(ctx, classifyManagerError(request.GetCollectionId(), callErr)))
	}
	health := server.manager.DependencyHealth()
	response := &pb.SearchConversationsResponse{
		Results:          make([]*pb.ConversationSearchResult, 0, len(results)),
		DependencyHealth: toDependencyHealth(health),
		DisplayText: server.envelopeText(ctx, health, renderConversationSearch(conversationSearchView{
			CollectionID: request.GetCollectionId(),
			Query:        request.GetQuery(),
			Results:      results,
		})),
	}
	for _, result := range results {
		response.Results = append(response.Results, &pb.ConversationSearchResult{
			ConversationId: result.ConversationID,
			MessageIndex:   result.MessageIndex,
			Role:           result.Role,
			TimestampUnix:  result.TimestampUnix,
			Score:          0,
			Content:        result.Content,
		})
	}
	return response, nil
}

// Doctor reports daemon-local diagnostics.
func (server *GRPCServer) Doctor(ctx context.Context, request *pb.DoctorRequest) (resp *pb.DoctorResponse, err error) {
	ctx, done := beginRPC(ctx, "Doctor")
	defer done(&err)
	_ = ctx
	_ = request
	diagnostics := server.manager.Doctor()
	response := &pb.DoctorResponse{
		Diagnostics: make([]*pb.Diagnostic, 0, len(diagnostics)),
	}
	for _, diagnostic := range diagnostics {
		response.Diagnostics = append(response.Diagnostics, &pb.Diagnostic{
			Severity: "warning",
			Code:     "path_check",
			Summary:  diagnostic,
			Detail:   diagnostic,
		})
	}
	health := server.manager.DependencyHealth()
	body := renderDoctor(diagnostics) + "\n\n" + renderDroppedSection(server.manager.DroppedCodebases())
	response.DisplayText = server.envelopeText(ctx, health, body)
	return response, nil
}

// Shutdown requests a graceful daemon shutdown.
func (server *GRPCServer) Shutdown(ctx context.Context, request *pb.ShutdownRequest) (resp *pb.ShutdownResponse, err error) {
	ctx, done := beginRPC(ctx, "Shutdown")
	defer done(&err)
	_ = request
	peerInfo, found := peer.FromContext(ctx)
	if found && peerInfo.Addr != nil {
		slog.InfoContext(ctx, "shutdown requested", "peer", peerInfo.Addr.String())
	} else {
		slog.InfoContext(ctx, "shutdown requested")
	}
	if server.shutdown != nil {
		go func() {
			defer func() {
				if recovered := recover(); recovered != nil {
					slog.ErrorContext(ctx, "shutdown callback panic", "err", fmt.Errorf("panic: %v", recovered))
				}
			}()
			server.shutdown()
		}()
	}
	return &pb.ShutdownResponse{Accepted: true}, nil
}

func pbClient(client *pb.ClientInfo) model.ClientInfo {
	if client == nil {
		return model.ClientInfo{Name: "", PID: 0}
	}
	return model.ClientInfo{
		Name: client.GetName(),
		PID:  client.GetPid(),
	}
}

func pbConversationDocuments(documents []*pb.ConversationDocument) []model.ConversationDocument {
	result := make([]model.ConversationDocument, 0, len(documents))
	for _, document := range documents {
		if document == nil {
			continue
		}
		result = append(result, model.ConversationDocument{
			ConversationID: document.GetConversationId(),
			MessageIndex:   document.GetMessageIndex(),
			Role:           document.GetRole(),
			TimestampUnix:  document.GetTimestampUnix(),
			Text:           document.GetText(),
		})
	}
	return result
}

func codebasePointer(found bool, codebase model.Codebase) *model.Codebase {
	if !found {
		return nil
	}
	return &codebase
}
