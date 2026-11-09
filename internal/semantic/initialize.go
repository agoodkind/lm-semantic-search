package semantic

import (
	"sync"
	"sync/atomic"

	"goodkind.io/lm-semantic-search/embedding"
	"goodkind.io/lm-semantic-search/internal/config"
)

func newServiceState(cfg config.Config, embedder embedding.Provider) *Service {
	return &Service{
		cfg:                            cfg,
		embedder:                       embedder,
		milvus:                         nil,
		insertRows:                     nil,
		available:                      atomic.Bool{},
		reconnectCancel:                nil,
		reconnectDone:                  nil,
		reconciliationMutex:            sync.Mutex{},
		reconciliationCancel:           nil,
		reconciliationWork:             sync.WaitGroup{},
		closeOnce:                      sync.Once{},
		reuseCatalogReady:              sync.Map{},
		reuseCatalogMutex:              sync.Mutex{},
		reuseCatalogAppendMutex:        sync.Mutex{},
		reuseVectorDimensions:          sync.Map{},
		reuseVectorDimensionMutex:      sync.Mutex{},
		reuseVectorDimensionGeneration: make(map[string]uint64),
		collectionLoads: collectionLoadCoordinator{
			mutex:   sync.Mutex{},
			flights: nil,
		},
		loadGates:                   newCollectionLoadGates(),
		residency:                   nil,
		ensuredSplitPartColumns:     sync.Map{},
		ensuredReuseIdentityColumns: sync.Map{},
		mmapPolicyVersions:          make(map[string]int),
		mmapPolicyMutex:             sync.Mutex{},
		mmapPolicyGeneration:        make(map[string]uint64),
		mmapPolicyFailures:          make(map[string]mmapPolicyFailure),
	}
}
