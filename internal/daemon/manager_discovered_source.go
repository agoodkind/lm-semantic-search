package daemon

import (
	"cmp"
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
	discoveredSourceTierOverlap            discoveredSourceTier = "content_overlap"
)

type discoveredSourceCandidate struct {
	codebase     model.Codebase
	mainWorktree bool
}

const (
	discoveredSourceTierRankSameHeadCommit = iota
	discoveredSourceTierRankMainWorktree
	discoveredSourceTierRankLatestCompletedRun
)

type rankedDiscoveredSource struct {
	candidate discoveredSourceCandidate
	overlap   discoveredSourceOverlap
	tier      discoveredSourceTier
	tierRank  int
}

type discoveredSourceSelection struct {
	codebase model.Codebase
	tier     discoveredSourceTier
	overlap  discoveredSourceOverlap
}

func (manager *Manager) selectDiscoveredWorktreeSourceLocked(
	info gitworktree.Info,
	indexConfig model.IndexConfig,
	overlaps map[string]discoveredSourceOverlap,
) (discoveredSourceSelection, bool) {
	candidates := manager.discoveredSourceCandidatesLocked(info, indexConfig)
	liveCandidates := make([]discoveredSourceCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		if ownsLiveCollection(candidate.codebase) {
			liveCandidates = append(liveCandidates, candidate)
		}
	}

	var noOverlap discoveredSourceOverlap
	var noSource model.Codebase
	if len(liveCandidates) == 0 {
		for _, candidate := range manager.candidatesWithActiveJobLocked(candidates) {
			if manager.worktreeSourceCycleLocked(info.WorktreeRoot, candidate.codebase.CanonicalPath) {
				continue
			}
			return discoveredSourceSelection{codebase: candidate.codebase, tier: discoveredSourceTierActiveJob, overlap: noOverlap}, true
		}
		return discoveredSourceSelection{codebase: noSource, tier: discoveredSourceTierNone, overlap: noOverlap}, false
	}

	eligible := make([]rankedDiscoveredSource, 0, len(liveCandidates))
	for _, ranked := range rankDiscoveredSources(info.WorktreeRoot, liveCandidates, overlaps) {
		if manager.worktreeSourceCycleLocked(info.WorktreeRoot, ranked.candidate.codebase.CanonicalPath) {
			continue
		}
		eligible = append(eligible, ranked)
	}
	if len(eligible) == 0 {
		return discoveredSourceSelection{codebase: noSource, tier: discoveredSourceTierNone, overlap: noOverlap}, false
	}
	selected := eligible[0]
	tier := selected.tier
	if len(eligible) > 1 && selected.overlap.matches > eligible[1].overlap.matches {
		tier = discoveredSourceTierOverlap
	}
	return discoveredSourceSelection{codebase: selected.candidate.codebase, tier: tier, overlap: selected.overlap}, true
}

func rankDiscoveredSources(
	worktreeRoot string,
	liveCandidates []discoveredSourceCandidate,
	overlaps map[string]discoveredSourceOverlap,
) []rankedDiscoveredSource {
	atHeadCommit := make(map[string]struct{}, len(liveCandidates))
	for _, candidate := range candidatesAtHeadCommit(worktreeRoot, liveCandidates) {
		atHeadCommit[candidate.codebase.ID] = struct{}{}
	}
	ranked := make([]rankedDiscoveredSource, 0, len(liveCandidates))
	for _, candidate := range liveCandidates {
		tier := discoveredSourceTierLatestCompletedRun
		tierRank := discoveredSourceTierRankLatestCompletedRun
		if _, sameHeadCommit := atHeadCommit[candidate.codebase.ID]; sameHeadCommit {
			tier = discoveredSourceTierSameHeadCommit
			tierRank = discoveredSourceTierRankSameHeadCommit
		} else if candidate.mainWorktree {
			tier = discoveredSourceTierMainWorktree
			tierRank = discoveredSourceTierRankMainWorktree
		}
		ranked = append(ranked, rankedDiscoveredSource{
			candidate: candidate,
			overlap:   overlaps[candidate.codebase.ID],
			tier:      tier,
			tierRank:  tierRank,
		})
	}
	slices.SortStableFunc(ranked, compareRankedDiscoveredSources)
	return ranked
}

func compareRankedDiscoveredSources(first rankedDiscoveredSource, second rankedDiscoveredSource) int {
	if first.overlap.matches != second.overlap.matches {
		return cmp.Compare(second.overlap.matches, first.overlap.matches)
	}
	if first.tierRank != second.tierRank {
		return cmp.Compare(first.tierRank, second.tierRank)
	}
	if first.tierRank == discoveredSourceTierRankLatestCompletedRun {
		byCompletedRun := lastCompletedRunAt(second.candidate.codebase).Compare(lastCompletedRunAt(first.candidate.codebase))
		if byCompletedRun != 0 {
			return byCompletedRun
		}
	}
	return compareDiscoveredSourceCandidates(first.candidate, second.candidate)
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
