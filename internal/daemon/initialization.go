package daemon

import (
	"context"
	"path/filepath"
	"sync"
	"time"

	"goodkind.io/lm-semantic-search/internal/cbm"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/jobscheduler"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
)

func newManagerState(
	ctx context.Context,
	cfg config.Config,
	dependencies managerDependencies,
) *Manager {
	return &Manager{
		config:                  cfg,
		mu:                      sync.Mutex{},
		policyMutationBlocked:   false,
		transitionMutex:         sync.Mutex{},
		policyMutationMutex:     sync.Mutex{},
		codebases:               map[string]model.Codebase{},
		jobs:                    map[string]model.Job{},
		pendingCodeJobs:         map[string]pendingCodeRequest{},
		interruptedConvergeJobs: map[string]model.Job{},
		cancels:                 map[string]context.CancelFunc{},
		done:                    map[string]chan struct{}{},
		failedBuildRetries:      map[string]int{},
		heldWorktreeBuilds:      map[string]struct{}{},
		lastJobJournalAt:        map[string]time.Time{},
		appendJobEvent:          store.AppendJobEvent,
		appendJobTransition:     nil,
		jobJournal:              nil,
		runner:                  indexer.NewRunner(),
		semantic:                nil,
		graphEngines:            map[string]*cbm.Engine{},
		graphLifecycle:          map[string]*graphLifecycleState{},
		graphMutex:              sync.Mutex{},
		graphIndex:              defaultGraphIndex,
		graphIndexHook:          nil,
		lifecycleHook:           nil,
		lifecycleMutex:          sync.Mutex{},
		startedAt:               clock.Now(),
		watcherActivity:         nil,
		watcherActivityMutex:    sync.Mutex{},
		jobScheduler: jobscheduler.New(
			ctx,
			max(1, cfg.MaxConcurrentIndexJobs),
			dependencies.activitySource,
		),
		jobCapacityTimings: defaultJobCapacityTimings(),
		syncLock: newSyncLock(
			filepath.Join(cfg.ContextRoot, "mcp-sync.flock"),
			cfg.ContextRoot,
		),
		health: dependencyHealth{
			Mode:                dependencyHealthy,
			Since:               time.Time{},
			StoreReachableAt:    time.Time{},
			EmbedderReachableAt: time.Time{},
		},
		dependencyFailureGeneration: 0,
		healthObservations:          0,
		appliedStoreObservation:     0,
		appliedEmbedderObservation:  0,
		lastDepProbeAt:              time.Time{},
		deferredBuildDelay:          defaultDeferredBuildDelay,
		bootSelfCheckDelay:          defaultBootSelfCheckDelay,
		indexability:                nil,
		observer:                    nil,
		maintenance: model.MaintenanceState{
			Enabled: false,
			Reason:  "",
			Since:   time.Time{},
		},
		modelDownload: nil,
	}
}
