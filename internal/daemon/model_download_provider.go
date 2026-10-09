package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"goodkind.io/lm-semantic-search/embedding"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/embedding/onnx"
	"goodkind.io/lm-semantic-search/internal/model"
)

type modelGatedProvider struct {
	supervisor *modelDownloadSupervisor
	mutex      sync.Mutex
	provider   embedding.Provider
}

func newModelGatedProvider(supervisor *modelDownloadSupervisor) *modelGatedProvider {
	return &modelGatedProvider{supervisor: supervisor, mutex: sync.Mutex{}, provider: nil}
}

func modelNotReadyError(snapshot modelDownloadSnapshot) *adapterr.AdapterError {
	message := fmt.Sprintf("embedding model files are not installed; download state is %s", snapshot.State)
	if snapshot.ProgressKnown {
		message += fmt.Sprintf(" artifact=%s downloaded_bytes=%d", snapshot.Artifact, snapshot.DownloadedBytes)
		if snapshot.TotalBytes > 0 {
			message += fmt.Sprintf(" total_bytes=%d", snapshot.TotalBytes)
		}
	}
	if snapshot.DecisionKnown {
		message += fmt.Sprintf(
			" network_classification=%s decision=%s",
			snapshot.Classification,
			snapshot.Decision,
		)
	}
	if snapshot.LastError != "" {
		message += " last_error=" + snapshot.LastError
	}
	return adapterr.NewEmbeddingModelNotReady(message, "Retry after the download completes. If the download is deferred, use the profile command to set --model-download-network-override or change --model-download-network-policy.", nil)
}

func (provider *modelGatedProvider) ProviderName() model.EmbeddingProvider {
	return model.EmbeddingProviderONNX
}

func (provider *modelGatedProvider) Provider(ctx context.Context) (embedding.Provider, error) {
	snapshot, _ := provider.supervisor.observe()
	if snapshot.State != modelDownloadComplete {
		provider.supervisor.requestRecheck()
		notReady := modelNotReadyError(snapshot)
		slog.WarnContext(ctx, "model_download.embedding_refused", "state", string(snapshot.State), "err", notReady)
		return nil, notReady
	}
	provider.mutex.Lock()
	defer provider.mutex.Unlock()
	if provider.provider != nil {
		return provider.provider, nil
	}
	built, err := onnx.NewProviderForInstalledPreset(
		provider.supervisor.cfg.ModelCacheRoot,
		provider.supervisor.preset,
	)
	if err != nil {
		slog.ErrorContext(ctx, "create ONNX embedding provider failed", "err", err)
		return nil, fmt.Errorf("create ONNX embedding provider: %w", err)
	}
	provider.provider = built
	return built, nil
}
