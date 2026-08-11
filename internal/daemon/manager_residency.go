package daemon

import "goodkind.io/lm-semantic-search/internal/semantic"

type semanticResidencySnapshotter interface {
	ResidencySnapshot() semantic.ResidencySnapshot
}

// ResidencySnapshot reports Milvus controller state for loopback diagnostics.
func (manager *Manager) ResidencySnapshot() semantic.ResidencySnapshot {
	if manager == nil {
		return emptyManagerResidencySnapshot()
	}
	snapshotter, ok := manager.semantic.(semanticResidencySnapshotter)
	if !ok {
		return emptyManagerResidencySnapshot()
	}
	return snapshotter.ResidencySnapshot()
}

func emptyManagerResidencySnapshot() semantic.ResidencySnapshot {
	return semantic.ResidencySnapshot{
		IdleTimeoutMS:            0,
		ReconciliationGeneration: 0,
		Closed:                   false,
		Collections:              nil,
	}
}
