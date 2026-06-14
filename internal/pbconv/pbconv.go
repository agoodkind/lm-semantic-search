// Package pbconv converts daemon model types to generated protobuf types.
package pbconv

import (
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/view"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// FromStartIndexConfig maps a gRPC start-index request into daemon config.
func FromStartIndexConfig(request *pb.StartIndexRequest) model.IndexConfig {
	config := model.IndexConfig{
		SplitterType:       "ast",
		SplitterChunkSize:  2500,
		SplitterOverlap:    300,
		Extensions:         append([]string{}, request.GetCustomExtensions()...),
		IgnorePatterns:     append([]string{}, request.GetIgnorePatterns()...),
		IgnoreDigest:       "",
		EmbeddingProvider:  "",
		EmbeddingModel:     "",
		EmbeddingDimension: 0,
		VectorBackend:      "milvus",
		Hybrid:             true,
	}
	if request.GetSplitter() != nil {
		if request.GetSplitter().GetType() != "" {
			config.SplitterType = request.GetSplitter().GetType()
		}
		if request.GetSplitter().GetChunkSize() > 0 {
			config.SplitterChunkSize = request.GetSplitter().GetChunkSize()
		}
		if request.GetSplitter().GetOverlap() > 0 {
			config.SplitterOverlap = request.GetSplitter().GetOverlap()
		}
	}
	return config
}

// ToCodebase converts one daemon codebase record into its protobuf form.
func ToCodebase(codebase model.Codebase) *pb.Codebase {
	result := &pb.Codebase{
		Id:                    codebase.ID,
		CanonicalPath:         codebase.CanonicalPath,
		Status:                string(codebase.Status),
		ActiveJobId:           codebase.ActiveJobID,
		EffectiveConfig:       toIndexConfig(codebase.EffectiveConfig),
		CollectionName:        codebase.CollectionName,
		LegacyCollectionNames: append([]string{}, codebase.LegacyCollectionNames...),
		MerkleSnapshotPath:    codebase.MerkleSnapshotPath,
		InodeTrackingDisabled: codebase.InodeTrackingDisabled,
		UpdatedAt:             ts(codebase.UpdatedAt),
	}
	if codebase.LastSuccessfulRun != nil {
		result.LastSuccessfulRun = &pb.IndexRunSummary{
			IndexedFiles: codebase.LastSuccessfulRun.IndexedFiles,
			TotalChunks:  codebase.LastSuccessfulRun.TotalChunks,
			Status:       codebase.LastSuccessfulRun.Status,
			CompletedAt:  ts(codebase.LastSuccessfulRun.CompletedAt),
			SkippedFiles: append([]string{}, codebase.LastSuccessfulRun.SkippedFiles...),
		}
	}
	if codebase.LastFailedRun != nil {
		result.LastFailedRun = &pb.IndexRunFailure{
			Message:                 codebase.LastFailedRun.Message,
			LastAttemptedPercentage: codebase.LastFailedRun.LastAttemptedPercentage,
			FailedAt:                ts(codebase.LastFailedRun.FailedAt),
		}
	}
	return result
}

// ToJob converts one daemon job record into its protobuf form.
func ToJob(job model.Job) *pb.Job {
	result := &pb.Job{
		Id:            job.ID,
		CodebaseId:    job.CodebaseID,
		RequestedPath: job.RequestedPath,
		CanonicalPath: job.CanonicalPath,
		Client: &pb.ClientInfo{
			Name: job.Client.Name,
			Pid:  job.Client.PID,
		},
		Operation: job.Operation,
		State:     string(job.State),
		Forced:    job.Forced,
		Trigger:   jobTrigger(job),
		Progress: &pb.Progress{
			Phase:                     job.Progress.Phase,
			PhasePercent:              job.Progress.PhasePercent,
			OverallPercent:            job.Progress.OverallPercent,
			Unit:                      job.Progress.Unit,
			FilesTotal:                job.Progress.FilesTotal,
			FilesProcessed:            job.Progress.FilesProcessed,
			ChunksReused:              job.Progress.ChunksReused,
			ChunksGenerated:           job.Progress.ChunksGenerated,
			EmbeddingBatchesTotal:     job.Progress.EmbeddingBatchesTotal,
			EmbeddingBatchesCompleted: job.Progress.EmbeddingBatchesCompleted,
			CollectionRowsWritten:     job.Progress.CollectionRowsWritten,
			LastEventAt:               ts(job.Progress.LastEventAt),
			HeartbeatAt:               ts(job.Progress.HeartbeatAt),
			Breakdown:                 BreakdownProto(job.Progress),
		},
		Config:      toIndexConfig(job.Config),
		StartedAt:   ts(job.StartedAt),
		UpdatedAt:   ts(job.UpdatedAt),
		CompletedAt: tsp(job.CompletedAt),
		Outcome:     jobOutcome(job.State),
	}
	if job.Error != nil {
		result.Error = &pb.JobError{
			Message:   job.Error.Message,
			Retryable: job.Error.Retryable,
		}
	}
	return result
}

// ProgressCounts maps a job's model.Progress into the resolver input. It is the
// one place the raw counters become view.ProgressCounts, shared by the daemon's
// human adapter and the wire breakdown, so the breakdown input never diverges.
func ProgressCounts(p model.Progress) view.ProgressCounts {
	return view.ProgressCounts{
		RunMode:                p.RunMode,
		Unit:                   p.Unit,
		FilesTotal:             p.FilesTotal,
		FilesProcessed:         p.FilesProcessed,
		FilesAdded:             p.FilesAdded,
		FilesModified:          p.FilesModified,
		FilesRemoved:           p.FilesRemoved,
		FilesEmbedded:          p.FilesEmbedded,
		FilesSkippedOversize:   p.FilesSkippedOversize,
		FilesSkippedUnreadable: p.FilesSkippedUnreadable,
		FilesPending:           p.FilesPending,
		ChunksTotal:            p.ChunksTotal,
		ChunksReused:           p.ChunksReused,
		ChunksGenerated:        p.ChunksGenerated,
	}
}

// BreakdownProto resolves a job's progress into the proto outcome breakdown, the
// wire mirror of view.ResolveBreakdown, so JSON consumers read the same tree the
// human surfaces render.
func BreakdownProto(p model.Progress) *pb.OutcomeBreakdown {
	breakdown := view.ResolveBreakdown(ProgressCounts(p))
	return &pb.OutcomeBreakdown{
		ScopeLabel:  breakdown.ScopeLabel,
		Processed:   breakdown.Processed,
		ScopeTotal:  breakdown.ScopeTotal,
		FileRows:    outcomeRowsToProto(breakdown.FileRows),
		ChunksTotal: breakdown.ChunksTotal,
		ChunkRows:   outcomeRowsToProto(breakdown.ChunkRows),
	}
}

func outcomeRowsToProto(rows []view.OutcomeRow) []*pb.OutcomeRow {
	out := make([]*pb.OutcomeRow, 0, len(rows))
	for _, row := range rows {
		out = append(out, &pb.OutcomeRow{Kind: outcomeKindToProto(row.Kind), Count: row.Count})
	}
	return out
}

func outcomeKindToProto(kind view.OutcomeKind) pb.OutcomeKind {
	switch kind {
	case view.KindEmbedded:
		return pb.OutcomeKind_OUTCOME_KIND_EMBEDDED
	case view.KindUnchanged:
		return pb.OutcomeKind_OUTCOME_KIND_UNCHANGED
	case view.KindRemoved:
		return pb.OutcomeKind_OUTCOME_KIND_REMOVED
	case view.KindPending:
		return pb.OutcomeKind_OUTCOME_KIND_PENDING
	case view.KindOversize:
		return pb.OutcomeKind_OUTCOME_KIND_OVERSIZE
	case view.KindUnreadable:
		return pb.OutcomeKind_OUTCOME_KIND_UNREADABLE
	case view.KindAdded:
		return pb.OutcomeKind_OUTCOME_KIND_ADDED
	case view.KindReused:
		return pb.OutcomeKind_OUTCOME_KIND_REUSED
	default:
		return pb.OutcomeKind_OUTCOME_KIND_UNSPECIFIED
	}
}

// jobOutcome resolves the terminal result token for the wire: "succeeded",
// "failed", or "canceled", and "" while the job is live. It exists so machine
// consumers (the CLI's --wait follower in particular) never derive
// terminality from the raw state field.
func jobOutcome(state model.JobState) string {
	switch state {
	case model.JobStateCompleted:
		return "succeeded"
	case model.JobStateFailed:
		return "failed"
	case model.JobStateCancelled:
		return "canceled"
	case model.JobStateQueued, model.JobStateRunning, model.JobStateCancelling:
		return ""
	default:
		return ""
	}
}

// Operation values a daemon job can carry, mirrored here so the wire trigger
// token can be derived without importing the daemon package.
const (
	jobOperationIndex    = "index"
	triggerInitialBuild  = "initial_build"
	triggerForcedReindex = "forced_reindex"
	triggerChangedFiles  = "changed_files"
)

// jobTrigger derives the wire trigger token from the job's operation and force
// flag: a full build is an initial build unless forced, and any other operation
// is a changed-files run.
func jobTrigger(job model.Job) string {
	if job.Operation == jobOperationIndex {
		if job.Forced {
			return triggerForcedReindex
		}
		return triggerInitialBuild
	}
	return triggerChangedFiles
}

func toIndexConfig(config model.IndexConfig) *pb.IndexConfig {
	return &pb.IndexConfig{
		SplitterType:       config.SplitterType,
		SplitterChunkSize:  config.SplitterChunkSize,
		SplitterOverlap:    config.SplitterOverlap,
		Extensions:         append([]string{}, config.Extensions...),
		IgnorePatterns:     append([]string{}, config.IgnorePatterns...),
		IgnoreDigest:       config.IgnoreDigest,
		EmbeddingProvider:  config.EmbeddingProvider,
		EmbeddingModel:     config.EmbeddingModel,
		EmbeddingDimension: config.EmbeddingDimension,
		VectorBackend:      config.VectorBackend,
		Hybrid:             config.Hybrid,
	}
}

// ToPathClassification converts a daemon classification verdict into its
// protobuf form. A nil verdict returns nil so the response message stays
// canonical without an unspecified placeholder.
func ToPathClassification(classification *model.PathClassification) *pb.PathClassification {
	if classification == nil {
		return nil
	}
	return &pb.PathClassification{
		Kind:                pathClassificationKindToProto(classification.Kind),
		ExcludedByPattern:   classification.ExcludedByPattern,
		ExcludedByGitignore: classification.ExcludedByGitignore,
		CoveringCodebaseId:  classification.CoveringCodebaseID,
	}
}

func pathClassificationKindToProto(kind model.PathClassificationKind) pb.PathClassification_Kind {
	switch kind {
	case model.PathClassificationInScopeIndexed:
		return pb.PathClassification_KIND_IN_SCOPE_INDEXED
	case model.PathClassificationInScopeExcluded:
		return pb.PathClassification_KIND_IN_SCOPE_EXCLUDED
	case model.PathClassificationInScopeUnindexed:
		return pb.PathClassification_KIND_IN_SCOPE_UNINDEXED
	case model.PathClassificationOutOfScope:
		return pb.PathClassification_KIND_OUT_OF_SCOPE
	case model.PathClassificationUnspecified:
		return pb.PathClassification_KIND_UNSPECIFIED
	default:
		return pb.PathClassification_KIND_UNSPECIFIED
	}
}

func ts(value time.Time) *timestamppb.Timestamp {
	if value.IsZero() {
		return nil
	}
	return timestamppb.New(value)
}

func tsp(value *time.Time) *timestamppb.Timestamp {
	if value == nil {
		return nil
	}
	return ts(*value)
}
