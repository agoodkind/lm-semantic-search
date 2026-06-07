package daemon

import (
	"errors"
	"time"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/clock"
)

// dependencyMode names a degraded shared-dependency condition. The empty mode is
// healthy. Each non-empty mode selects one status-banner variant. A degraded
// mode is recorded only when a job actually fails on that condition, so a brief
// rate-limit absorbed by the in-process retry never reaches the banner; a busy
// mode appears only when the endpoint stays at capacity long enough to fail a
// job, which is a real outage worth surfacing. A cancellation is transient and
// never degrades the banner.
type dependencyMode string

const (
	dependencyHealthy             dependencyMode = ""
	dependencyEmbedderUnreachable dependencyMode = "embedder_unreachable"
	dependencyEmbedderRejected    dependencyMode = "embedder_rejected"
	dependencyEmbedderBusy        dependencyMode = "embedder_busy"
	dependencyStoreUnavailable    dependencyMode = "store_unavailable"
)

// dependencyHealth is the daemon's view of shared-infrastructure health: the
// embedding pipeline and the vector store. It is one global fact about the whole
// daemon, never a property of a single codebase, and it drives the status banner.
// It is guarded by Manager.mu and observed from job outcomes rather than probed
// synchronously, so a status call never blocks on a live dependency check.
type dependencyHealth struct {
	// Mode is the current degraded condition, or dependencyHealthy when the
	// shared dependencies last looked reachable.
	Mode dependencyMode
	// Since is when the current degraded mode began. Zero when healthy.
	Since time.Time
	// LastHealthyAt is the last time a dependency interaction succeeded. Zero
	// until the first success.
	LastHealthyAt time.Time
}

// Degraded reports whether any dependency outage is in effect, which is when the
// banner shows. This includes a busy endpoint, so a sustained at-capacity outage
// surfaces a banner. The waiting fold keys off this too, but only for a codebase
// with no live job, so a brief rate-limit during an active job still reads
// "indexing" rather than "waiting".
func (health dependencyHealth) Degraded() bool {
	return health.Mode != dependencyHealthy
}

// degradeModeFor maps a job-failure error to the banner mode it implies, or
// dependencyHealthy for anything that is not a shared-infrastructure outage. It
// runs only on a job failure, so a busy class here means the endpoint stayed at
// capacity past the in-process retry and failed the job, which is a real outage.
// A cancellation is transient and never degrades the banner.
func degradeModeFor(err error) dependencyMode {
	if err == nil {
		return dependencyHealthy
	}
	var adapterErr *adapterr.AdapterError
	if !errors.As(err, &adapterErr) {
		return dependencyHealthy
	}
	switch adapterErr.Class {
	case adapterr.ClassEmbedderUnreachable:
		return dependencyEmbedderUnreachable
	case adapterr.ClassEmbedderRejected:
		return dependencyEmbedderRejected
	case adapterr.ClassEmbedderBusy:
		return dependencyEmbedderBusy
	case adapterr.ClassMilvusUnavailable:
		return dependencyStoreUnavailable
	case adapterr.ClassEmbedCancelled, adapterr.ClassNotIndexed,
		adapterr.ClassUnknownCodebaseID, adapterr.ClassCollectionMissing, adapterr.ClassCollectionNotReady,
		adapterr.ClassSearchResultIncomplete, adapterr.ClassInvalidPath, adapterr.ClassInvalidArgument,
		adapterr.ClassConflictingJob, adapterr.ClassJobNotFound, adapterr.ClassInternal:
		return dependencyHealthy
	default:
		return dependencyHealthy
	}
}

// noteDependencyFailureLocked records a hard shared-infrastructure outage on the
// health record. A busy, cancelled, or non-infra error leaves the record
// unchanged. The caller holds manager.mu.
func (manager *Manager) noteDependencyFailureLocked(err error) {
	mode := degradeModeFor(err)
	if mode == dependencyHealthy {
		return
	}
	if manager.health.Mode != mode {
		manager.health.Mode = mode
		manager.health.Since = clock.Now()
	}
}

// noteDependencyHealthyLocked clears the health record after a dependency
// interaction succeeds, so the banner stops showing within the cycle that first
// reaches a recovered dependency. The caller holds manager.mu.
func (manager *Manager) noteDependencyHealthyLocked() {
	manager.health.Mode = dependencyHealthy
	manager.health.Since = time.Time{}
	manager.health.LastHealthyAt = clock.Now()
}

// DependencyHealth returns a snapshot of the current shared-dependency health
// for the render layer. It reads the cached record and never probes.
func (manager *Manager) DependencyHealth() dependencyHealth {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	return manager.health
}
