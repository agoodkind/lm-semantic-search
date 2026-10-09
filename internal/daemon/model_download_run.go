package daemon

import (
	"context"
	"log/slog"
	"time"

	"goodkind.io/lm-semantic-search/internal/embedding/onnx"
	"goodkind.io/lm-semantic-search/internal/modeldownload"
)

func (supervisor *modelDownloadSupervisor) run(ctx context.Context) {
	for {
		wait, finished := supervisor.attempt(ctx)
		if finished {
			return
		}
		timer := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		case <-supervisor.wake:
			timer.Stop()
		}
	}
}

func (supervisor *modelDownloadSupervisor) attempt(ctx context.Context) (time.Duration, bool) {
	select {
	case <-supervisor.wake:
	default:
	}
	installed, err := onnx.PresetFilesInstalled(ctx, supervisor.cfg.ModelCacheRoot, supervisor.preset)
	if ctx.Err() != nil {
		return 0, true
	}
	if err != nil {
		slog.ErrorContext(ctx, "model_download.verify_failed", "model", supervisor.preset.Name, "err", err)
		supervisor.recordFailure(err)
		return supervisor.dependencies.RetryInterval, false
	}
	if installed {
		supervisor.recordComplete(ctx)
		return 0, true
	}
	if !supervisor.decide(ctx) {
		return supervisor.dependencies.RecheckInterval, false
	}
	err = onnx.InstallPresetFilesWithProgress(
		ctx,
		supervisor.dependencies.HTTPClient,
		supervisor.cfg.ModelCacheRoot,
		supervisor.preset,
		supervisor.recordProgress,
		supervisor.attemptSleep(),
	)
	if ctx.Err() != nil {
		return 0, true
	}
	if err != nil {
		slog.ErrorContext(ctx, "model_download.failed", "model", supervisor.preset.Name, "err", err)
		supervisor.recordFailure(err)
		return supervisor.dependencies.RetryInterval, false
	}
	supervisor.recordComplete(ctx)
	return 0, true
}

func (supervisor *modelDownloadSupervisor) attemptSleep() modeldownload.SleepFunc {
	backoff := supervisor.dependencies.AttemptBackoff
	if backoff <= 0 {
		return nil
	}
	return func(ctx context.Context, _ time.Duration) {
		timer := time.NewTimer(backoff)
		defer timer.Stop()
		select {
		case <-ctx.Done():
		case <-timer.C:
		}
	}
}

func (supervisor *modelDownloadSupervisor) recordProgress(
	artifact string,
	downloadedBytes int64,
	totalBytes int64,
) {
	supervisor.update(func(snapshot *modelDownloadSnapshot) {
		snapshot.Artifact = artifact
		snapshot.DownloadedBytes = downloadedBytes
		snapshot.TotalBytes = totalBytes
		snapshot.ProgressKnown = true
	})
}

func (supervisor *modelDownloadSupervisor) recordFailure(err error) {
	supervisor.update(func(snapshot *modelDownloadSnapshot) {
		snapshot.State = modelDownloadFailed
		snapshot.LastError = err.Error()
	})
}

func (supervisor *modelDownloadSupervisor) recordComplete(ctx context.Context) {
	slog.InfoContext(ctx, "model_download.completed", "model", supervisor.preset.Name)
	supervisor.update(func(snapshot *modelDownloadSnapshot) {
		snapshot.State = modelDownloadComplete
		snapshot.LastError = ""
	})
}
