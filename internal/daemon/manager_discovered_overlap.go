package daemon

import (
	"context"
	"os"
	"path/filepath"
	"slices"

	"goodkind.io/lm-semantic-search/internal/gitworktree"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
)

const (
	discoveredSourceOverlapSampleLimit       = 200
	discoveredSourceOverlapMinimumCandidates = 2
)

type discoveredSourceOverlap struct {
	matches int
	sampled int
}

// discoveredSourceOverlaps compares file hashes for sampled paths in the discovered
// worktree with candidates' saved Merkle snapshots. It returns match counts and
// the common sample size keyed by candidate codebase ID. It reads snapshots from
// disk without querying the vector store and performs file reads outside the
// registry lock. A candidate's checkout HEAD does not identify the file contents
// in its saved snapshot. Reuse source selection ranks content matches before using
// HEAD to break ties.
func (manager *Manager) discoveredSourceOverlaps(
	ctx context.Context,
	info gitworktree.Info,
	indexConfig model.IndexConfig,
) map[string]discoveredSourceOverlap {
	manager.mu.Lock()
	candidates := manager.discoveredSourceCandidatesLocked(info, indexConfig)
	manager.mu.Unlock()

	liveCandidates := make([]discoveredSourceCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		if ownsLiveCollection(candidate.codebase) {
			liveCandidates = append(liveCandidates, candidate)
		}
	}
	if len(liveCandidates) < discoveredSourceOverlapMinimumCandidates {
		return map[string]discoveredSourceOverlap{}
	}

	snapshots := make(map[string]merkle.Snapshot, len(liveCandidates))
	for _, candidate := range liveCandidates {
		checkpoint := manager.loadLiveCheckpoint(ctx, candidate.codebase, candidate.codebase.EffectiveConfig.IgnoreDigest)
		if !checkpoint.usable() {
			continue
		}
		snapshots[candidate.codebase.ID] = checkpoint.snapshot
	}

	sampledPaths := sampleSnapshotPaths(snapshots)
	worktreeHashes := make(map[string]string, len(sampledPaths))
	for _, relativePath := range sampledPaths {
		data, err := os.ReadFile(filepath.Join(info.WorktreeRoot, relativePath))
		if err != nil {
			continue
		}
		worktreeHashes[relativePath] = merkle.ContentHash(data)
	}

	overlaps := make(map[string]discoveredSourceOverlap, len(snapshots))
	for codebaseID, snapshot := range snapshots {
		matches := 0
		for _, relativePath := range sampledPaths {
			storedHash, stored := snapshot.Files[relativePath]
			worktreeHash, read := worktreeHashes[relativePath]
			if stored && read && storedHash == worktreeHash {
				matches++
			}
		}
		overlaps[codebaseID] = discoveredSourceOverlap{matches: matches, sampled: len(sampledPaths)}
	}
	return overlaps
}

func sampleSnapshotPaths(snapshots map[string]merkle.Snapshot) []string {
	union := make(map[string]struct{})
	for _, snapshot := range snapshots {
		for relativePath := range snapshot.Files {
			union[relativePath] = struct{}{}
		}
	}
	paths := make([]string, 0, len(union))
	for relativePath := range union {
		paths = append(paths, relativePath)
	}
	slices.Sort(paths)
	if len(paths) <= discoveredSourceOverlapSampleLimit {
		return paths
	}
	sampled := make([]string, 0, discoveredSourceOverlapSampleLimit)
	for sampleIndex := range discoveredSourceOverlapSampleLimit {
		sampled = append(sampled, paths[sampleIndex*len(paths)/discoveredSourceOverlapSampleLimit])
	}
	return sampled
}
