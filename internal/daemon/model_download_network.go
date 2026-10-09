package daemon

import (
	"context"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/networkcost"
)

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
