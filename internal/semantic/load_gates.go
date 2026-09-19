package semantic

import "sync"

// collectionLoadGates holds the daemon-wide checks every collection load
// passes through before it reaches Milvus: the concurrency limiter and the
// memory-exhaustion backoff. Both are built on first use from the service
// config, so a Service assembled without NewService still loads under them.
type collectionLoadGates struct {
	limitOnce   sync.Once
	limit       *collectionLoadLimiter
	backoffOnce sync.Once
	backoff     *collectionLoadBackoff
}

func newCollectionLoadGates() collectionLoadGates {
	return collectionLoadGates{
		limitOnce:   sync.Once{},
		limit:       nil,
		backoffOnce: sync.Once{},
		backoff:     nil,
	}
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
