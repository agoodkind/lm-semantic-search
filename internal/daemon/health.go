package daemon

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/status"
)

const (
	// dependencyProbeInterval debounces the active backend probe so status reads
	// poll the store and embedder at most once per interval. Between probes a
	// status read observes the last probe outcome, which bounds staleness to the
	// interval rather than waiting for the next real job or search to fail.
	dependencyProbeInterval = 5 * time.Second
	// dependencyProbeTimeout bounds one probe so a hung backend cannot stall a
	// status read waiting on a dependency.
	dependencyProbeTimeout = 2 * time.Second
)

// dependencyMode names a degraded shared-dependency condition. The empty mode is
// healthy. Each non-empty mode selects one status-banner variant. A degraded
// mode is recorded only when a job actually fails on that condition, so a brief
// rate-limit absorbed by the in-process retry never reaches the banner; a busy
// mode appears only when the endpoint stays at capacity long enough to fail a
// job, which is a real outage worth surfacing. A cancellation is transient and
// never degrades the banner.
//
// The type and its values alias the status package so the daemon keeps its short
// names while the canonical definitions live in the single status source of
// truth.
type dependencyMode = status.DependencyMode

const (
	dependencyHealthy             = status.Healthy
	dependencyEmbedderUnreachable = status.EmbedderUnreachable
	dependencyEmbedderRejected    = status.EmbedderRejected
	dependencyEmbedderPaused      = status.EmbedderPaused
	dependencyEmbedderBusy        = status.EmbedderBusy
	dependencyStoreUnavailable    = status.StoreUnavailable
)

// dependencyHealth is the daemon's view of shared-infrastructure health: the
// embedding pipeline and the vector store. It is one global fact about the whole
// daemon, never a property of a single codebase, and it drives the status banner.
// It is guarded by Manager.mu and holds only what something actually observed:
// job and search outcomes, plus the bounded store probe a readiness surface runs
// through refreshDependencyHealth. Rendering it is a pure read.
type dependencyHealth struct {
	// Mode is the current degraded condition, or dependencyHealthy when the
	// shared dependencies last looked reachable.
	Mode dependencyMode
	// Since is when the current degraded mode began. Zero when healthy.
	Since time.Time
	// StoreReachableAt is the last time the vector store itself answered: a store
	// probe that completed, a search that read a collection, or an index job that
	// wrote rows. Zero until the store has answered once.
	StoreReachableAt time.Time
	// EmbedderReachableAt is the last time the embedding endpoint itself
	// answered: a search that embedded its query, or an index job that embedded a
	// file. A store probe never sets it, because the probe does not reach the
	// embedder, so an embedder outage keeps its true last-reachable time however
	// often the store is probed. Zero until the embedder has answered once.
	EmbedderReachableAt time.Time
}

// lastReachableAt returns the last time the dependency the current mode names
// answered. A surface shows this beside that dependency's name, so the time and
// the name have to describe the same thing: a store outage reads the store's
// time and an embedder outage reads the embedder's. A healthy record names no
// single dependency, so it reads whichever answered most recently.
func (health dependencyHealth) lastReachableAt() time.Time {
	switch health.Mode {
	case dependencyStoreUnavailable:
		return health.StoreReachableAt
	case dependencyEmbedderUnreachable, dependencyEmbedderRejected, dependencyEmbedderPaused, dependencyEmbedderBusy:
		return health.EmbedderReachableAt
	case dependencyHealthy:
		return laterTime(health.StoreReachableAt, health.EmbedderReachableAt)
	default:
		return laterTime(health.StoreReachableAt, health.EmbedderReachableAt)
	}
}

// laterTime returns the more recent of two timestamps. A zero timestamp means
// the dependency has never answered, and it loses to any real one.
func laterTime(first time.Time, second time.Time) time.Time {
	if first.After(second) {
		return first
	}
	return second
}

// healthObservation is one piece of dependency evidence, numbered in the order
// the evidence was gathered rather than the order it came back. A probe takes
// its number under Manager.mu before its round trip starts; an observation that
// gathers and records inside one critical section takes its number there. The
// numbers are what let a slow probe recognize that something newer has already
// answered for the dependency it covers.
type healthObservation uint64

// dependencySet names which shared dependencies one observation carries evidence
// about. A store probe covers the store alone; a search or an index job that
// embedded and wrote covers both. Evidence is ordered per dependency, so store
// evidence never ages out embedder evidence or the reverse.
type dependencySet struct {
	// Store reports that the observation looked at the vector store.
	Store bool
	// Embedder reports that the observation looked at the embedding endpoint.
	Embedder bool
}

// dependenciesFor names which dependency a degraded mode is a fact about, so a
// recorded outage is ordered only against evidence about that same dependency.
func dependenciesFor(mode dependencyMode) dependencySet {
	switch mode {
	case dependencyStoreUnavailable:
		return dependencySet{Store: true, Embedder: false}
	case dependencyEmbedderUnreachable, dependencyEmbedderRejected, dependencyEmbedderPaused, dependencyEmbedderBusy:
		return dependencySet{Store: false, Embedder: true}
	case dependencyHealthy:
		return dependencySet{Store: false, Embedder: false}
	default:
		return dependencySet{Store: false, Embedder: false}
	}
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
	case adapterr.ClassEmbedderPaused:
		return dependencyEmbedderPaused
	case adapterr.ClassEmbedderBusy:
		return dependencyEmbedderBusy
	case adapterr.ClassMilvusUnavailable:
		return dependencyStoreUnavailable
	case adapterr.ClassEmbedCancelled, adapterr.ClassNotIndexed,
		adapterr.ClassUnknownCodebaseID, adapterr.ClassCollectionMissing, adapterr.ClassCollectionNotReady,
		adapterr.ClassSearchResultIncomplete, adapterr.ClassInvalidPath, adapterr.ClassInvalidArgument,
		adapterr.ClassConflictingJob, adapterr.ClassJobNotFound, adapterr.ClassIndexBudgetExceeded,
		adapterr.ClassInternal:
		return dependencyHealthy
	default:
		return dependencyHealthy
	}
}

// beginHealthObservationLocked numbers a piece of evidence at the moment it
// starts being gathered. The caller holds manager.mu, so the numbers are unique
// and totally ordered across every caller.
func (manager *Manager) beginHealthObservationLocked() healthObservation {
	manager.healthObservations++
	return manager.healthObservations
}

// supersededHealthObservationLocked reports whether newer evidence has already
// been applied for some dependency this observation covers, which makes this
// observation's verdict stale. The numbering and this comparison both happen
// under manager.mu, so two callers racing cannot interleave in a way that lets
// older evidence land on top of newer: whichever one holds the lock reads a
// number that already accounts for every observation applied before it.
func (manager *Manager) supersededHealthObservationLocked(observation healthObservation, covers dependencySet) bool {
	if covers.Store && observation < manager.appliedStoreObservation {
		return true
	}
	if covers.Embedder && observation < manager.appliedEmbedderObservation {
		return true
	}
	return false
}

// applyHealthObservationLocked marks this observation as the newest evidence for
// every dependency it covers. The caller holds manager.mu and has already found
// the observation is not superseded.
func (manager *Manager) applyHealthObservationLocked(observation healthObservation, covers dependencySet) {
	if covers.Store {
		manager.appliedStoreObservation = observation
	}
	if covers.Embedder {
		manager.appliedEmbedderObservation = observation
	}
}

// noteDependencyFailureLocked records a hard shared-infrastructure outage
// observed right now. A busy, cancelled, or non-infra error leaves the record
// unchanged. The caller holds manager.mu.
func (manager *Manager) noteDependencyFailureLocked(err error) {
	manager.recordDependencyFailureLocked(err, manager.beginHealthObservationLocked())
}

// recordDependencyFailureLocked records the outage the numbered observation saw.
// An observation numbered below evidence already applied for the same dependency
// lost the race and is dropped, so a slow failing probe cannot reinstate an
// outage that a newer success has since disproved. A repeat of the mode already
// on the record still counts as newer evidence and still advances the applied
// number, so an outage that keeps being observed keeps winning against anything
// that started earlier. The caller holds manager.mu.
func (manager *Manager) recordDependencyFailureLocked(err error, observation healthObservation) {
	mode := degradeModeFor(err)
	if mode == dependencyHealthy {
		return
	}
	covers := dependenciesFor(mode)
	if manager.supersededHealthObservationLocked(observation, covers) {
		slog.Debug("dependency.health.observation_superseded", "component", "daemon", "subcomponent", "health", "observation", uint64(observation), "verdict", string(mode))
		return
	}
	manager.applyHealthObservationLocked(observation, covers)
	// The failure generation advances only for a failure that reached the record.
	// A superseded observation returned above without changing anything, so
	// counting it would tell the boot self-check that newer evidence landed while
	// the record still holds what it held before.
	manager.dependencyFailureGeneration++
	if manager.health.Mode != mode {
		slog.Warn("dependency.health.degraded", "component", "daemon", "subcomponent", "health", "from", string(manager.health.Mode), "to", string(mode))
		manager.health.Mode = mode
		manager.health.Since = clock.Now()
	}
}

// noteDependencyHealthyLocked clears the health record after an interaction that
// reached both shared dependencies, so the banner stops showing within the cycle
// that first reaches a recovered dependency. It clears any mode and stamps both
// reachability times, so callers must use it only when the success proves the
// whole pipeline (a real embed plus store write, or a search that embedded its
// query and read the store), never for a store-only probe. The caller holds
// manager.mu.
//
// It gathers and records inside one critical section, so its evidence is the
// newest that exists and can never be superseded.
func (manager *Manager) noteDependencyHealthyLocked() {
	covers := dependencySet{Store: true, Embedder: true}
	manager.applyHealthObservationLocked(manager.beginHealthObservationLocked(), covers)
	if manager.health.Degraded() {
		slog.Info("dependency.health.recovered", "component", "daemon", "subcomponent", "health", "from", string(manager.health.Mode))
	}
	reachableAt := clock.Now()
	manager.health.Mode = dependencyHealthy
	manager.health.Since = time.Time{}
	manager.health.StoreReachableAt = reachableAt
	manager.health.EmbedderReachableAt = reachableAt
}

// recordStoreReachableLocked records that the store answered the numbered
// observation. It stamps the store's own reachability time and never the
// embedder's, and it clears a store-unavailable mode only, because the probe
// does not reach the embedder: an embedder outage is observed from real embed
// outcomes, and both that outage and the embedder's true last-reachable time
// must survive a clean store probe. The caller holds manager.mu.
func (manager *Manager) recordStoreReachableLocked(observation healthObservation) {
	covers := dependencySet{Store: true, Embedder: false}
	if manager.supersededHealthObservationLocked(observation, covers) {
		slog.Debug("dependency.health.observation_superseded", "component", "daemon", "subcomponent", "health", "observation", uint64(observation), "verdict", "store_reachable")
		return
	}
	manager.applyHealthObservationLocked(observation, covers)
	manager.health.StoreReachableAt = clock.Now()
	if manager.health.Mode == dependencyStoreUnavailable {
		slog.Info("dependency.health.recovered", "component", "daemon", "subcomponent", "health", "from", string(manager.health.Mode), "via", "store_probe")
		manager.health.Mode = dependencyHealthy
		manager.health.Since = time.Time{}
	}
}

// storeProbeFailure classifies a probe error as a store outage. The probe asks
// exactly one question, whether the store answers, so any error it returns is a
// store that did not. An error already carrying an outage class keeps it, and
// anything else becomes a store outage rather than being dropped by the class
// switch, which would leave a failed probe recorded as nothing at all.
func storeProbeFailure(err error) error {
	if degradeModeFor(err) != dependencyHealthy {
		return err
	}
	return adapterr.NewMilvusUnavailable(err)
}

// noteDependencyFailure records a shared-infrastructure outage on the health
// record, acquiring manager.mu. It is the lock-taking wrapper the search path
// uses, since search runs outside the job-state critical section. It no-ops for
// any error that is not a real outage.
func (manager *Manager) noteDependencyFailure(err error) {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	manager.noteDependencyFailureLocked(err)
}

// noteDependencyHealthy clears the health record after a dependency interaction
// succeeds, acquiring manager.mu. It is the lock-taking wrapper the search path
// uses when a query embed proves the embedder recovered.
func (manager *Manager) noteDependencyHealthy() {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	manager.noteDependencyHealthyLocked()
}

// dependencyFailureGenerationNow returns the current dependency-failure
// generation for an asynchronous operation to capture before it starts.
func (manager *Manager) dependencyFailureGenerationNow() uint64 {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	return manager.dependencyFailureGeneration
}

// noteDependencyHealthyIfGeneration clears health only when no dependency
// failure was recorded after the caller began its successful operation.
func (manager *Manager) noteDependencyHealthyIfGeneration(generation uint64) bool {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	if manager.dependencyFailureGeneration != generation {
		return false
	}
	manager.noteDependencyHealthyLocked()
	return true
}

// refreshDependencyHealth runs an active liveness probe of the search backend
// when the last probe is older than dependencyProbeInterval, updating the health
// record so a readiness surface reflects current reachability rather than the
// last job outcome. It is what makes a readiness answer describe now, so every
// surface that answers whether the daemon can serve calls it before reading the
// record.
//
// The probe runs without manager.mu held so backend I/O never blocks other
// reads; the record is updated under the lock afterward. The debounce timestamp
// is stamped before the probe, under the lock, so the interval is a property of
// the manager rather than of a caller: however many surfaces call this and
// however concurrently, at most one probe runs per interval and the rest read
// the record as it stands.
//
// Two guards keep a probe from recording something it did not observe. The probe
// takes an observation number before its round trip starts, and its verdict is
// dropped in either direction when newer store evidence has been applied since:
// a passing probe cannot clear a failure recorded while it was in flight, and a
// failing probe cannot erase a success recorded while it was in flight. And a
// probe failure that coincides with the caller's own context going away is
// ignored, because the failure is not attributable to the store; a probe that
// answered before the caller walked away is still positive evidence and is kept.
func (manager *Manager) refreshDependencyHealth(ctx context.Context) {
	manager.mu.Lock()
	semantic := manager.semantic
	stale := manager.lastDepProbeAt.IsZero() || clock.Now().Sub(manager.lastDepProbeAt) >= dependencyProbeInterval
	if semantic == nil || !stale {
		manager.mu.Unlock()
		return
	}
	manager.lastDepProbeAt = clock.Now()
	observation := manager.beginHealthObservationLocked()
	manager.mu.Unlock()

	probeCtx, cancel := context.WithTimeout(ctx, dependencyProbeTimeout)
	defer cancel()
	probeErr := semantic.ProbeHealth(probeCtx)

	manager.mu.Lock()
	defer manager.mu.Unlock()
	if probeErr != nil {
		if ctx.Err() != nil {
			slog.DebugContext(ctx, "dependency.health.probe_abandoned", "component", "daemon", "subcomponent", "health", "observation", uint64(observation), "err", probeErr)
			return
		}
		manager.recordDependencyFailureLocked(storeProbeFailure(probeErr), observation)
		return
	}
	// ProbeHealth checks store reachability only, so a clean probe is evidence
	// about the store alone: it clears a store-unavailable mode and stamps the
	// store's reachability time, and it leaves both an embedder outage and the
	// embedder's own last-reachable time untouched.
	manager.recordStoreReachableLocked(observation)
}

// DependencyHealth returns a snapshot of the current shared-dependency health
// for the render layer. It is a pure read: it never upgrades the record, because
// recovery requires positive evidence that a dependency answered, and a read
// surface holds none.
//
// The store client's Available flag cannot supply that evidence. It is a latch
// set when the client first dials and cleared only on shutdown, so it stays true
// after the store dies; clearing a recorded outage from it erased true failures
// on every surface that never probed. Recovery is left to the three paths that
// do hold evidence: refreshDependencyHealth, whose store probe is a real round
// trip; a search that reached the store and embedded its query; and an index job
// that actually embedded and wrote.
//
// Holding manager.mu here is why no probe belongs in this method: it is read
// from every render site, often several times per request, and a probe would put
// a bounded but real backend round trip inside that lock.
func (manager *Manager) DependencyHealth() dependencyHealth {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	return manager.dependencyHealthLocked()
}

// dependencyHealthLocked returns the health record for a caller that already
// holds manager.mu, such as one building a wider snapshot. It is the same pure
// read DependencyHealth performs, kept as its own function so every surface
// resolves health through one place rather than copying the raw record.
func (manager *Manager) dependencyHealthLocked() dependencyHealth {
	return manager.health
}

// pathCollectionReadiness maps the per-path collection facts to a
// status.CollectionReadiness, kept entirely separate from the global dependency
// mode. A non-eligible path is not-applicable. An eligible path reads absent,
// loading, or ready from CollectionState; a store that cannot answer the load
// state reads unknown. This never returns or touches a dependencyMode, so a
// per-path not-ready collection can never raise the global store banner. The
// global banner is owned by refreshDependencyHealth and ProbeHealth alone, which
// the caller consults separately for shared-dependency health.
func (manager *Manager) pathCollectionReadiness(ctx context.Context, canonicalPath string, searchableEligible bool) status.CollectionReadiness {
	if !searchableEligible || manager.semantic == nil || canonicalPath == "" {
		return status.CollectionNotApplicable
	}
	exists, loaded, err := manager.semantic.CollectionState(ctx, canonicalPath)
	switch {
	case err != nil:
		return status.CollectionUnknown
	case !exists:
		return status.CollectionAbsent
	case !loaded:
		return status.CollectionLoading
	default:
		return status.CollectionReady
	}
}
