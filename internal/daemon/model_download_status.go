package daemon

import (
	"context"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/networkcost"
)

const (
	statusGroupModelDownload = "model_download"
	percentScale             = 100

	jobPhaseModelDownload jobPhase = "model_download"
)

func modelDownloadMetrics(snapshot modelDownloadSnapshot) []*pb.Metric {
	list := []*pb.Metric{
		stringMetric(statusGroupModelDownload, "model_download.state", string(snapshot.State)),
	}
	if snapshot.ProgressKnown {
		list = append(list,
			stringMetric(statusGroupModelDownload, "model_download.artifact", snapshot.Artifact),
			intMetric(statusGroupModelDownload, "model_download.downloaded_bytes", snapshot.DownloadedBytes, unitBytes),
		)
	} else {
		list = append(list,
			absentMetric(statusGroupModelDownload, "model_download.artifact"),
			absentMetric(statusGroupModelDownload, "model_download.downloaded_bytes"),
		)
	}
	if snapshot.ProgressKnown && snapshot.TotalBytes > 0 {
		percent := float64(snapshot.DownloadedBytes) / float64(snapshot.TotalBytes) * percentScale
		list = append(list,
			intMetric(statusGroupModelDownload, "model_download.total_bytes", snapshot.TotalBytes, unitBytes),
			doubleMetric(statusGroupModelDownload, "model_download.percent", percent, unitPercent),
		)
	} else {
		list = append(list,
			absentMetric(statusGroupModelDownload, "model_download.total_bytes"),
			absentMetric(statusGroupModelDownload, "model_download.percent"),
		)
	}
	if snapshot.DecisionKnown {
		list = append(list,
			stringMetric(statusGroupModelDownload, "model_download.network_classification", string(snapshot.Classification)),
			stringMetric(statusGroupModelDownload, "model_download.network_policy", string(snapshot.Policy)),
			boolMetric(statusGroupModelDownload, "model_download.network_override", snapshot.Override),
			stringMetric(statusGroupModelDownload, "model_download.decision", string(snapshot.Decision)),
			boolMetric(
				statusGroupModelDownload,
				"model_download.warning",
				snapshot.Decision == networkcost.DecisionDownloadWithWarning,
			),
		)
	} else {
		list = append(list,
			absentMetric(statusGroupModelDownload, "model_download.network_classification"),
			absentMetric(statusGroupModelDownload, "model_download.network_policy"),
			absentMetric(statusGroupModelDownload, "model_download.network_override"),
			absentMetric(statusGroupModelDownload, "model_download.decision"),
			absentMetric(statusGroupModelDownload, "model_download.warning"),
		)
	}
	if snapshot.State == modelDownloadFailed && snapshot.LastError != "" {
		list = append(list, stringMetric(statusGroupModelDownload, "model_download.last_error", snapshot.LastError))
	} else {
		list = append(list, absentMetric(statusGroupModelDownload, "model_download.last_error"))
	}
	return list
}

func (manager *Manager) awaitEmbeddingModel(ctx context.Context, jobID string) bool {
	if manager.modelDownload == nil {
		return true
	}
	manager.modelDownload.requestRecheck()
	for {
		snapshot, changed := manager.modelDownload.observe()
		if snapshot.State == modelDownloadComplete {
			return true
		}
		manager.noteJobModelDownload(jobID, snapshot)
		select {
		case <-ctx.Done():
			return false
		case <-changed:
		}
	}
}

func (manager *Manager) noteJobModelDownload(jobID string, snapshot modelDownloadSnapshot) {
	manager.transitionMutex.Lock()
	defer manager.transitionMutex.Unlock()
	manager.mu.Lock()
	defer manager.mu.Unlock()

	job, found := manager.jobs[jobID]
	if !found || job.State != model.JobStateQueued {
		return
	}
	now := clock.Now()
	job.UpdatedAt = now
	job.Progress.Phase = string(jobPhaseModelDownload)
	job.Progress.LastEventAt = now
	job.Progress.HeartbeatAt = now
	job.Progress.ModelDownload = nil
	if snapshot.ProgressKnown {
		job.Progress.ModelDownload = &model.DownloadProgress{
			Artifact:        snapshot.Artifact,
			DownloadedBytes: snapshot.DownloadedBytes,
			TotalBytes:      snapshot.TotalBytes,
		}
	}
	manager.jobs[jobID] = job
}
