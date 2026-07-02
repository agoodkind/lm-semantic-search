package daemon

import (
	"fmt"
	"math"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/gitworktree"
	"goodkind.io/lm-semantic-search/internal/model"
)

type admissionState struct {
	maxChunks      int64
	maxBytes       int64
	expectedChunks int64
	growthLimit    int64
	chunks         int64
	bytes          int64
}

var emptyAdmissionBudget = model.AdmissionBudget{MaxJobChunks: 0, MaxJobBytes: 0}

func newAdmissionState(cfg config.Config, budget model.AdmissionBudget, expectedChunks int32) *admissionState {
	maxChunks := tighterPositiveLimit(int64(cfg.MaxJobChunks), int64(budget.MaxJobChunks))
	maxBytes := tighterPositiveLimit(cfg.MaxJobBytes, budget.MaxJobBytes)

	state := &admissionState{
		maxChunks:      maxChunks,
		maxBytes:       maxBytes,
		expectedChunks: int64(expectedChunks),
		growthLimit:    0,
		chunks:         0,
		bytes:          0,
	}
	if expectedChunks > 0 && (cfg.ExpectedJobGrowthFactor > 0 || cfg.ExpectedJobGrowthFloor > 0) {
		factor := cfg.ExpectedJobGrowthFactor
		if factor <= 0 {
			factor = 1
		}
		factorLimit := int64(math.Ceil(float64(expectedChunks) * factor))
		floorLimit := int64(expectedChunks) + int64(cfg.ExpectedJobGrowthFloor)
		state.growthLimit = maxInt64(factorLimit, floorLimit)
	}
	return state
}

func (state *admissionState) Admit(chunks []model.StoredChunk) error {
	if state == nil || len(chunks) == 0 {
		return nil
	}
	nextChunks := state.chunks + int64(len(chunks))
	nextBytes := state.bytes + storedChunkBytes(chunks)
	if state.maxChunks > 0 && nextChunks > state.maxChunks {
		return adapterr.NewIndexBudgetExceeded(fmt.Sprintf("chunk cap exceeded: %d chunks would exceed the fixed cap of %d", nextChunks, state.maxChunks))
	}
	if state.maxBytes > 0 && nextBytes > state.maxBytes {
		return adapterr.NewIndexBudgetExceeded(fmt.Sprintf("byte cap exceeded: %d bytes would exceed the fixed cap of %d", nextBytes, state.maxBytes))
	}
	if state.growthLimit > 0 && nextChunks > state.growthLimit {
		return adapterr.NewIndexBudgetExceeded(fmt.Sprintf("expected-size cap exceeded: %d chunks would exceed the baseline %d limit of %d", nextChunks, state.expectedChunks, state.growthLimit))
	}
	state.chunks = nextChunks
	state.bytes = nextBytes
	return nil
}

func storedChunkBytes(chunks []model.StoredChunk) int64 {
	var total int64
	for _, chunk := range chunks {
		total += int64(len(chunk.Content))
	}
	return total
}

func maxInt64(first int64, second int64) int64 {
	if first > second {
		return first
	}
	return second
}

func tighterPositiveLimit(serverLimit int64, requestLimit int64) int64 {
	if requestLimit > 0 && (serverLimit == 0 || requestLimit < serverLimit) {
		return requestLimit
	}
	return serverLimit
}

func (manager *Manager) admissionForJob(job model.Job) *admissionState {
	expectedChunks := manager.expectedChunksForAdmission(job.CodebaseID, job.CanonicalPath, job.Config)
	return newAdmissionState(manager.config, job.Budget, expectedChunks)
}

func (manager *Manager) admissionForCodebase(codebase model.Codebase) *admissionState {
	// Each converge batch gets the same per-job chunk and byte caps as an index
	// job, so repeated watcher batches cannot restart an unbounded budget.
	expectedChunks := manager.expectedChunksForAdmission(codebase.ID, codebase.CanonicalPath, codebase.EffectiveConfig)
	return newAdmissionState(manager.config, emptyAdmissionBudget, expectedChunks)
}

func (manager *Manager) expectedChunksForAdmission(codebaseID string, canonicalPath string, indexConfig model.IndexConfig) int32 {
	manager.mu.Lock()
	codebase, found := manager.codebases[codebaseID]
	if found && codebase.LastSuccessfulRun != nil {
		totalChunks := codebase.LiveChunkTotal
		if totalChunks == 0 {
			totalChunks = codebase.LastSuccessfulRun.TotalChunks
		}
		manager.mu.Unlock()
		return totalChunks
	}
	manager.mu.Unlock()
	return manager.largestSiblingSuccessfulChunks(canonicalPath, indexConfig)
}

func (manager *Manager) largestSiblingSuccessfulChunks(canonicalPath string, indexConfig model.IndexConfig) int32 {
	info, ok := gitworktree.Resolve(canonicalPath)
	if !ok {
		return 0
	}
	siblingRoots := gitworktree.SiblingWorktreeRoots(info.CommonDir)
	siblings := make(map[string]struct{}, len(siblingRoots))
	for _, root := range siblingRoots {
		if root != info.WorktreeRoot {
			siblings[root] = struct{}{}
		}
	}
	if len(siblings) == 0 {
		return 0
	}

	var largest int32
	manager.mu.Lock()
	defer manager.mu.Unlock()
	for _, codebase := range manager.codebases {
		if _, member := siblings[codebase.CanonicalPath]; !member {
			continue
		}
		if codebase.Kind == model.CodebaseKindDocument {
			continue
		}
		if codebase.LastSuccessfulRun == nil {
			continue
		}
		if !reuseModelMatches(codebase.EffectiveConfig, indexConfig) {
			continue
		}
		if codebase.LastSuccessfulRun.TotalChunks > largest {
			largest = codebase.LastSuccessfulRun.TotalChunks
		}
	}
	return largest
}
