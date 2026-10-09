package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding/onnx"
	"goodkind.io/lm-semantic-search/internal/modeldownload"
	"goodkind.io/lm-semantic-search/internal/networkcost"
	"goodkind.io/lm-semantic-search/internal/networkcost/platform"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

const (
	modelDownloadRecheckInterval = 5 * time.Minute
	modelDownloadRetryInterval   = time.Minute
)

type modelDownloadState string

const (
	modelDownloadNotNeeded   modelDownloadState = "not_needed"
	modelDownloadPending     modelDownloadState = "pending"
	modelDownloadDownloading modelDownloadState = "downloading"
	modelDownloadDeferred    modelDownloadState = "deferred"
	modelDownloadFailed      modelDownloadState = "failed"
	modelDownloadComplete    modelDownloadState = "complete"
)

func (options ModelDownloadOptions) withDefaults() ModelDownloadOptions {
	if options.NetworkSource == nil {
		options.NetworkSource = platform.New()
	}
	if options.HTTPClient == nil {
		options.HTTPClient = http.DefaultClient
	}
	if options.Presets == nil {
		options.Presets = offlinemodel.PinnedRegistry()
	}
	if options.RecheckInterval <= 0 {
		options.RecheckInterval = modelDownloadRecheckInterval
	}
	if options.RetryInterval <= 0 {
		options.RetryInterval = modelDownloadRetryInterval
	}
	return options
}

type modelDownloadSnapshot struct {
	State           modelDownloadState
	Artifact        string
	DownloadedBytes int64
	TotalBytes      int64
	ProgressKnown   bool
	Classification  networkcost.Classification
	Decision        networkcost.Decision
	Policy          networkcost.Preference
	Override        bool
	DecisionKnown   bool
	LastError       string
}

type modelDownloadSupervisor struct {
	cfg          config.Config
	preset       offlinemodel.Preset
	dependencies ModelDownloadOptions

	mutex    sync.Mutex
	snapshot modelDownloadSnapshot
	changed  chan struct{}
	cancel   context.CancelFunc
	done     chan struct{}

	wake chan struct{}
}

func modelDownloadRequired(cfg config.Config) bool {
	return cfg.IndexBackend == config.IndexBackendLocal &&
		cfg.EmbeddingProvider == config.EmbeddingProviderONNX
}

func newModelDownloadSupervisor(
	ctx context.Context,
	cfg config.Config,
	dependencies ModelDownloadOptions,
) (*modelDownloadSupervisor, error) {
	dependencies = dependencies.withDefaults()
	preset, err := dependencies.Presets.Resolve(cfg.OfflineEmbeddingModel)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"resolve offline embedding model failed",
			"model", cfg.OfflineEmbeddingModel,
			"err", err,
		)
		return nil, fmt.Errorf("resolve offline embedding model: %w", err)
	}
	return &modelDownloadSupervisor{
		cfg:          cfg,
		preset:       preset,
		dependencies: dependencies,
		mutex:        sync.Mutex{},
		snapshot:     pendingModelDownloadSnapshot(),
		changed:      make(chan struct{}),
		cancel:       nil,
		done:         nil,
		wake:         make(chan struct{}, 1),
	}, nil
}

func pendingModelDownloadSnapshot() modelDownloadSnapshot {
	return modelDownloadSnapshot{
		State:           modelDownloadPending,
		Artifact:        "",
		DownloadedBytes: 0,
		TotalBytes:      0,
		ProgressKnown:   false,
		Classification:  "",
		Decision:        "",
		Policy:          "",
		Override:        false,
		DecisionKnown:   false,
		LastError:       "",
	}
}

func (supervisor *modelDownloadSupervisor) observe() (modelDownloadSnapshot, <-chan struct{}) {
	if supervisor == nil {
		snapshot := pendingModelDownloadSnapshot()
		snapshot.State = modelDownloadNotNeeded
		return snapshot, nil
	}
	supervisor.mutex.Lock()
	defer supervisor.mutex.Unlock()
	return supervisor.snapshot, supervisor.changed
}

func (supervisor *modelDownloadSupervisor) update(mutate func(*modelDownloadSnapshot)) {
	supervisor.mutex.Lock()
	defer supervisor.mutex.Unlock()
	mutate(&supervisor.snapshot)
	close(supervisor.changed)
	supervisor.changed = make(chan struct{})
}

func (supervisor *modelDownloadSupervisor) requestRecheck() {
	if supervisor == nil {
		return
	}
	select {
	case supervisor.wake <- struct{}{}:
	default:
	}
}

func (supervisor *modelDownloadSupervisor) start(ctx context.Context) {
	if supervisor == nil {
		return
	}
	runContext, cancel := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
	supervisor.mutex.Lock()
	supervisor.cancel = cancel
	supervisor.done = done
	supervisor.mutex.Unlock()
	go func() {
		defer close(done)
		defer func() {
			if recovered := recover(); recovered != nil {
				panicErr := fmt.Errorf("panic: %v", recovered)
				slog.ErrorContext(runContext, "model_download.panicked", "model", supervisor.preset.Name, "err", panicErr)
				supervisor.recordFailure(panicErr)
			}
		}()
		supervisor.run(runContext)
	}()
}

func (supervisor *modelDownloadSupervisor) stop() {
	if supervisor == nil {
		return
	}
	supervisor.mutex.Lock()
	cancel := supervisor.cancel
	done := supervisor.done
	supervisor.mutex.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	<-done
}

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

func (supervisor *modelDownloadSupervisor) networkSettings(
	ctx context.Context,
) config.ModelDownloadNetworkSettings {
	configured := config.ModelDownloadNetworkSettings{
		Policy:   supervisor.cfg.ModelDownloadNetworkPolicy,
		Override: supervisor.cfg.ModelDownloadNetworkOverride,
	}
	if supervisor.cfg.ConfigPath == "" {
		return configured
	}
	settings, err := config.ReadModelDownloadNetworkSettings(supervisor.cfg.ConfigPath)
	if err != nil {
		slog.WarnContext(
			ctx,
			"model_download.network_settings.read_failed",
			"path", supervisor.cfg.ConfigPath,
			"err", err,
		)
		return configured
	}
	return settings
}

func (supervisor *modelDownloadSupervisor) decide(ctx context.Context) bool {
	settings := supervisor.networkSettings(ctx)
	classification := supervisor.dependencies.NetworkSource.Classify(ctx)
	decision := networkcost.Decide(classification, settings.Policy, settings.Override)
	state := modelDownloadDownloading
	if decision == networkcost.DecisionDefer {
		state = modelDownloadDeferred
	}
	supervisor.update(func(snapshot *modelDownloadSnapshot) {
		snapshot.State = state
		snapshot.Classification = classification
		snapshot.Decision = decision
		snapshot.Policy = settings.Policy
		snapshot.Override = settings.Override
		snapshot.DecisionKnown = true
	})
	level := slog.LevelInfo
	event := "model_download.started"
	switch decision {
	case networkcost.DecisionDefer:
		level = slog.LevelWarn
		event = "model_download.deferred"
	case networkcost.DecisionDownloadWithWarning:
		level = slog.LevelWarn
		event = "model_download.started_with_warning"
	case networkcost.DecisionDownload:
	}
	slog.Default().LogAttrs(
		ctx,
		level,
		event,
		slog.String("model", supervisor.preset.Name),
		slog.String("network_classification", string(classification)),
		slog.String("network_policy", string(settings.Policy)),
		slog.Bool("network_override", settings.Override),
		slog.String("decision", string(decision)),
	)
	return decision != networkcost.DecisionDefer
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
