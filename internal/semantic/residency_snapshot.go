package semantic

import "sort"

// ResidencyCollectionSnapshot reports controller state without changing it.
type ResidencyCollectionSnapshot struct {
	Collection         string `json:"collection"`
	State              string `json:"state"`
	Leases             int    `json:"leases"`
	Observations       int    `json:"observations"`
	Pins               int    `json:"pins"`
	TimerArmed         bool   `json:"timer_armed"`
	IdleDeadlineUnixMS int64  `json:"idle_deadline_unix_ms"`
	IdleGeneration     uint64 `json:"idle_generation"`
	Loading            bool   `json:"loading"`
	Transitioning      bool   `json:"transitioning"`
	Maintenance        bool   `json:"maintenance"`
	Reconciliation     uint64 `json:"reconciliation"`
}

// ResidencySnapshot reports the controller and every tracked collection.
type ResidencySnapshot struct {
	IdleTimeoutMS            int64                         `json:"idle_timeout_ms"`
	ReconciliationGeneration uint64                        `json:"reconciliation_generation"`
	Closed                   bool                          `json:"closed"`
	Collections              []ResidencyCollectionSnapshot `json:"collections"`
}

// ResidencySnapshot returns one point-in-time copy under the controller lock.
func (controller *collectionResidencyController) ResidencySnapshot() ResidencySnapshot {
	if controller == nil {
		return emptyResidencySnapshot()
	}
	controller.mutex.Lock()
	defer controller.mutex.Unlock()

	collections := make([]ResidencyCollectionSnapshot, 0, len(controller.entries))
	for _, entry := range controller.entries {
		deadlineUnixMS := int64(0)
		if !entry.idleDeadline.IsZero() {
			deadlineUnixMS = entry.idleDeadline.UnixMilli()
		}
		collections = append(collections, ResidencyCollectionSnapshot{
			Collection:         entry.name,
			State:              collectionResidencyStateName(entry.state),
			Leases:             entry.leases,
			Observations:       entry.observations,
			Pins:               entry.pins,
			TimerArmed:         entry.idleTimer != nil,
			IdleDeadlineUnixMS: deadlineUnixMS,
			IdleGeneration:     entry.idleGeneration,
			Loading:            entry.load != nil,
			Transitioning:      entry.activeTransition != nil,
			Maintenance:        entry.maintenance,
			Reconciliation:     entry.reconciliation,
		})
	}
	sort.Slice(collections, func(left int, right int) bool {
		return collections[left].Collection < collections[right].Collection
	})
	return ResidencySnapshot{
		IdleTimeoutMS:            controller.config.idleTimeout.Milliseconds(),
		ReconciliationGeneration: controller.reconciliationGeneration,
		Closed:                   controller.closed,
		Collections:              collections,
	}
}

// ResidencySnapshot reports Milvus residency state for loopback diagnostics.
func (service *Service) ResidencySnapshot() ResidencySnapshot {
	if service == nil {
		return emptyResidencySnapshot()
	}
	return service.residency.ResidencySnapshot()
}

func emptyResidencySnapshot() ResidencySnapshot {
	return ResidencySnapshot{
		IdleTimeoutMS:            0,
		ReconciliationGeneration: 0,
		Closed:                   false,
		Collections:              nil,
	}
}
