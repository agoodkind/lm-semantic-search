package daemon

import (
	"context"
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/networkcost"
)

func (supervisor *modelDownloadSupervisor) networkSettings(
	ctx context.Context,
) (config.ModelDownloadNetworkSettings, error) {
	if supervisor.cfg.ConfigPath == "" {
		configured := config.ModelDownloadNetworkSettings{
			Policy:   supervisor.cfg.ModelDownloadNetworkPolicy,
			Override: supervisor.cfg.ModelDownloadNetworkOverride,
		}
		return configured, nil
	}
	settings, err := config.ReadModelDownloadNetworkSettings(supervisor.cfg.ConfigPath)
	if err != nil {
		slog.WarnContext(
			ctx,
			"model_download.network_settings.read_failed",
			"model", supervisor.preset.Name,
			"path", supervisor.cfg.ConfigPath,
			"err", err,
		)
		return config.ModelDownloadNetworkSettings{Policy: "", Override: false}, fmt.Errorf(
			"load model download supervisor network settings: %w",
			err,
		)
	}
	return settings, nil
}

func (supervisor *modelDownloadSupervisor) decide(ctx context.Context) bool {
	settings, err := supervisor.networkSettings(ctx)
	if err != nil {
		supervisor.update(func(snapshot *modelDownloadSnapshot) {
			snapshot.State = modelDownloadDeferred
			snapshot.DecisionKnown = false
		})
		return false
	}
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
