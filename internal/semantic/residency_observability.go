package semantic

import (
	"context"
	"log/slog"
	"time"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/metrics"
)

func (controller *collectionResidencyController) updateStateMetricsLocked() {
	idle := 0
	loading := 0
	ready := 0
	for collectionName, entry := range controller.entries {
		if isStagingCollection(collectionName) {
			continue
		}
		switch entry.state {
		case collectionResidencyCold:
			idle++
		case collectionResidencyLoading:
			loading++
		case collectionResidencyReady:
			ready++
		case collectionResidencyUnknown:
		}
	}
	metrics.SetMilvusCollectionStates(idle, loading, ready)
}

func collectionResidencyStateName(state collectionResidencyState) string {
	switch state {
	case collectionResidencyCold:
		return "idle"
	case collectionResidencyLoading:
		return "loading"
	case collectionResidencyReady:
		return "ready"
	case collectionResidencyUnknown:
		return "unknown"
	}
	return "unknown"
}

func logCollectionResidencyEvent(
	ctx context.Context,
	level slog.Level,
	message string,
	collectionName string,
	state collectionResidencyState,
	progress int,
	elapsed time.Duration,
	leaseCount int,
	idle time.Duration,
	err error,
) {
	slog.LogAttrs(
		ctx,
		level,
		message,
		slog.String("collection", collectionName),
		slog.String("state", collectionResidencyStateName(state)),
		slog.Int("progress", progress),
		slog.Int64("elapsed_ms", elapsed.Milliseconds()),
		slog.Int("lease_count", leaseCount),
		slog.Int64("idle_ms", idle.Milliseconds()),
		slog.String("error_class", adapterr.Code(err)),
	)
}
