package daemon

import (
	"slices"
	"strings"
	"time"

	"goodkind.io/lm-semantic-search/internal/gitworktree"
	"goodkind.io/lm-semantic-search/internal/model"
)

type discoveredSourceTier string

const (
	discoveredSourceTierNone               discoveredSourceTier = "none"
	discoveredSourceTierSameHeadCommit     discoveredSourceTier = "same_head_commit"
	discoveredSourceTierMainWorktree       discoveredSourceTier = "main_worktree"
	discoveredSourceTierLatestCompletedRun discoveredSourceTier = "latest_completed_run"
	discoveredSourceTierActiveJob          discoveredSourceTier = "active_job"
)

type discoveredSourceCandidate struct {
	codebase     model.Codebase
	mainWorktree bool
}

type discoveredSourceTierCandidates struct {
	tier       discoveredSourceTier
	candidates []discoveredSourceCandidate
}

func (manager *Manager) selectDiscoveredWorktreeSourceLocked(
	info gitworktree.Info,
	indexConfig model.IndexConfig,
) (model.Codebase, discoveredSourceTier, bool) {
	candidates := manager.discoveredSourceCandidatesLocked(info, indexConfig)
	liveCandidates := make([]discoveredSourceCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		if ownsLiveCollection(candidate.codebase) {
			liveCandidates = append(liveCandidates, candidate)
		}
	}

	var tiers []discoveredSourceTierCandidates
	if len(liveCandidates) > 0 {
		tiers = []discoveredSourceTierCandidates{
			{tier: discoveredSourceTierSameHeadCommit, candidates: candidatesAtHeadCommit(info.WorktreeRoot, liveCandidates)},
			{tier: discoveredSourceTierMainWorktree, candidates: mainWorktreeCandidates(liveCandidates)},
			{tier: discoveredSourceTierLatestCompletedRun, candidates: candidatesByLatestCompletedRun(liveCandidates)},
		}
	} else {
		tiers = []discoveredSourceTierCandidates{
			{tier: discoveredSourceTierActiveJob, candidates: manager.candidatesWithActiveJobLocked(candidates)},
		}
	}
	for _, tierCandidates := range tiers {
		for _, candidate := range tierCandidates.candidates {
			if manager.worktreeSourceCycleLocked(info.WorktreeRoot, candidate.codebase.CanonicalPath) {
				continue
			}
			return candidate.codebase, tierCandidates.tier, true
		}
	}
	var noSource model.Codebase
	return noSource, discoveredSourceTierNone, false
}

func (manager *Manager) discoveredSourceCandidatesLocked(
	info gitworktree.Info,
	indexConfig model.IndexConfig,
) []discoveredSourceCandidate {
	candidates := make([]discoveredSourceCandidate, 0)
	if info.CommonDir == "" {
		return candidates
	}
	siblingRoots := gitworktree.SiblingWorktreeRoots(info.CommonDir)
	for _, codebase := range manager.codebases {
		if codebase.CanonicalPath == info.WorktreeRoot || !slices.Contains(siblingRoots, codebase.CanonicalPath) {
			continue
		}
		if codebase.Kind == model.CodebaseKindDocument || codebase.CollectionName == "" {
			continue
		}
		if !reuseModelMatches(codebase.EffectiveConfig, indexConfig) {
			continue
		}
		worktree, resolved := gitworktree.Resolve(codebase.CanonicalPath)
		candidates = append(candidates, discoveredSourceCandidate{
			codebase:     codebase,
			mainWorktree: resolved && !worktree.Linked && worktree.WorktreeRoot == codebase.CanonicalPath,
		})
	}
	slices.SortFunc(candidates, compareDiscoveredSourceCandidates)
	return candidates
}

func compareDiscoveredSourceCandidates(first discoveredSourceCandidate, second discoveredSourceCandidate) int {
	if first.mainWorktree != second.mainWorktree {
		if first.mainWorktree {
			return -1
		}
		return 1
	}
	return strings.Compare(first.codebase.CanonicalPath, second.codebase.CanonicalPath)
}

func candidatesAtHeadCommit(worktreeRoot string, candidates []discoveredSourceCandidate) []discoveredSourceCandidate {
	matching := make([]discoveredSourceCandidate, 0, len(candidates))
	targetCommit, targetResolved := gitworktree.HeadCommit(worktreeRoot)
	if !targetResolved {
		return matching
	}
	for _, candidate := range candidates {
		candidateCommit, candidateResolved := gitworktree.HeadCommit(candidate.codebase.CanonicalPath)
		if candidateResolved && candidateCommit == targetCommit {
			matching = append(matching, candidate)
		}
	}
	return matching
}

func mainWorktreeCandidates(candidates []discoveredSourceCandidate) []discoveredSourceCandidate {
	mainWorktrees := make([]discoveredSourceCandidate, 0, 1)
	for _, candidate := range candidates {
		if candidate.mainWorktree {
			mainWorktrees = append(mainWorktrees, candidate)
		}
	}
	return mainWorktrees
}

func candidatesByLatestCompletedRun(candidates []discoveredSourceCandidate) []discoveredSourceCandidate {
	ordered := slices.Clone(candidates)
	slices.SortStableFunc(ordered, func(first discoveredSourceCandidate, second discoveredSourceCandidate) int {
		return lastCompletedRunAt(second.codebase).Compare(lastCompletedRunAt(first.codebase))
	})
	return ordered
}

func lastCompletedRunAt(codebase model.Codebase) time.Time {
	if codebase.LastSuccessfulRun == nil {
		var never time.Time
		return never
	}
	return codebase.LastSuccessfulRun.CompletedAt
}

func (manager *Manager) candidatesWithActiveJobLocked(candidates []discoveredSourceCandidate) []discoveredSourceCandidate {
	active := make([]discoveredSourceCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		if manager.activeJobSnapshotLocked(candidate.codebase) != nil {
			active = append(active, candidate)
		}
	}
	return active
}
