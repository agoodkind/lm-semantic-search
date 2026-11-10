package semantic

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
)

// collectionLoadGates owns the daemon-wide checks that run before a
// collection load sends its request to Milvus: the operator's maintenance mode,
// the concurrency limiter, and the memory-exhaustion backoff. The limiter and
// backoff are built on first use from the service config, so a Service
// assembled without NewService still loads under them.
type collectionLoadGates struct {
	maintenance atomic.Bool
	limitOnce   sync.Once
	limit       *collectionLoadLimiter
	backoffOnce sync.Once
	backoff     *collectionLoadBackoff
}

func newCollectionLoadGates() collectionLoadGates {
	return collectionLoadGates{
		maintenance: atomic.Bool{},
		limitOnce:   sync.Once{},
		limit:       nil,
		backoffOnce: sync.Once{},
		backoff:     nil,
	}
}

// SetMaintenance turns the maintenance gate on or off. While it is on every
// new collection load is refused with ErrMaintenance before any request
// reaches Milvus; loads already in flight finish on their own, and loaded
// collections keep serving the callers that already hold them.
func (service *Service) SetMaintenance(enabled bool) {
	service.loadGates.maintenance.Store(enabled)
}

// refuseLoadDuringMaintenance returns the maintenance refusal for
// collectionName while the gate is on, and nil otherwise.
func (service *Service) refuseLoadDuringMaintenance(ctx context.Context, collectionName string) error {
	if !service.loadGates.maintenance.Load() {
		return nil
	}
	err := fmt.Errorf("load of collection %s refused: %w", collectionName, ErrMaintenance)
	slog.WarnContext(ctx, "semantic.collection_load_refused_maintenance",
		"component", "semantic",
		"subcomponent", "load",
		"collection", collectionName,
		"err", err,
	)
	return err
}

// collectionLoadSlots returns the daemon-wide limiter, built on first use from
// the configured cap.
func (service *Service) collectionLoadSlots() *collectionLoadLimiter {
	service.loadGates.limitOnce.Do(func() {
		service.loadGates.limit = newCollectionLoadLimiter(
			service.cfg.MilvusMaxConcurrentCollectionLoads,
		)
	})
	return service.loadGates.limit
}

// loadBackoff returns the daemon-wide backoff, built on first use.
func (service *Service) loadBackoff() *collectionLoadBackoff {
	service.loadGates.backoffOnce.Do(func() {
		service.loadGates.backoff = newCollectionLoadBackoff()
	})
	return service.loadGates.backoff
}
