package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/networkcost"
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
	supervisor.mutex.Lock()
	if supervisor.done != nil {
		supervisor.mutex.Unlock()
		return
	}
	runContext, cancel := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
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
