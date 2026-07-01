package daemon

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/model"
)

// ignoreOverridesFromRegistry returns one codebase's custom ignore patterns
// from the registry source of truth, which is the single home for per-codebase
// ignore state. It is the provider the resolver calls while building a
// codebase's matcher, and the resolver calls it only inside buildRules on a
// cache miss, which is off the hot path, so taking manager.mu here is
// acceptable.
//
// This relies on a narrower invariant than "the resolver is never called under
// manager.mu". The decision methods are the only entry points that can trigger
// buildRules and so call back into this provider: Decide, Ignored, IgnoreDetail,
// and DecideContent. None of them runs while the caller holds manager.mu. Every
// such site releases manager.mu first: converge (ConvergePaths unlocks before
// the per-path loop), the watcher dispatch (the watcher never takes manager.mu),
// status classifyTrackedPath (called after GetIndex unlocks), discovery, and
// merkle Capture. So this lock can never deadlock against an in-flight decision.
// InvalidateRules is exempt and may run under manager.mu, because it only drops
// the cached rules and never calls this provider. A codebase absent from the registry
// returns nil. The returned slice is a clone so the caller cannot mutate the
// stored config.
func (manager *Manager) ignoreOverridesFromRegistry(codebaseID string) []string {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	codebase, found := manager.codebases[codebaseID]
	if !found {
		return nil
	}
	return slices.Clone(codebase.EffectiveConfig.IgnorePatterns)
}

func (manager *Manager) submoduleAllowlistFromRegistry(codebaseID string) []string {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	codebase, found := manager.codebases[codebaseID]
	if !found {
		return nil
	}
	return slices.Clone(codebase.EffectiveConfig.IncludeSubmodules)
}

// legacyDigestForCodebase returns the canonical digest of the codebase's
// stored EffectiveConfig. The plan helpers pass this to
// merkle.LoadSnapshotForConfig so a pre-config-digest snapshot is salvaged
// when the request matches what the codebase was last indexed under.
func (manager *Manager) legacyDigestForCodebase(codebaseID string) string {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	codebase, found := manager.codebases[codebaseID]
	if !found {
		return ""
	}
	return digestIndexConfig(codebase.EffectiveConfig)
}

// digestIndexConfig hashes the indexing-relevant fields of an IndexConfig.
// The IgnoreDigest field itself is excluded from the hash input so the
// digest is stable across runs: re-hashing a stored EffectiveConfig
// produces the same value, which the merkle snapshot's ConfigDigest match
// relies on for resume.
func digestIndexConfig(indexConfig model.IndexConfig) string {
	hashable := indexConfig
	hashable.IgnoreDigest = ""
	digestBytes, err := json.Marshal(hashable)
	if err != nil {
		digest := sha256.Sum256([]byte(hashable.SplitterType))
		return "sha256:" + hex.EncodeToString(digest[:])
	}
	digest := sha256.Sum256(digestBytes)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func (manager *Manager) enrichIndexConfig(indexConfig model.IndexConfig) model.IndexConfig {
	if strings.TrimSpace(indexConfig.SplitterType) == "" {
		indexConfig.SplitterType = "ast"
	}
	if indexConfig.SplitterChunkSize == 0 {
		indexConfig.SplitterChunkSize = 2500
	}
	if indexConfig.SplitterOverlap == 0 {
		indexConfig.SplitterOverlap = 300
	}
	indexConfig.EmbeddingProvider = manager.config.EmbeddingProvider
	indexConfig.EmbeddingModel = manager.config.EmbeddingModel
	if manager.config.EmbeddingDimension > 0 {
		indexConfig.EmbeddingDimension = manager.config.EmbeddingDimension
	}
	indexConfig.VectorBackend = "milvus"
	indexConfig.Hybrid = manager.config.HybridMode
	indexConfig.IgnorePatterns = mergeDistinct(indexConfig.IgnorePatterns, manager.config.CustomIgnorePatterns)
	indexConfig.IncludeSubmodules = mergeNormalizedSubmodules(indexConfig.IncludeSubmodules, manager.config.IncludeSubmodules)
	return indexConfig
}

// mergeDistinct returns base + extras with duplicates removed and original
// ordering preserved.
func mergeDistinct(base []string, extras []string) []string {
	if len(extras) == 0 {
		return base
	}
	seen := make(map[string]struct{}, len(base)+len(extras))
	out := make([]string, 0, len(base)+len(extras))
	for _, value := range base {
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	for _, value := range extras {
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func mergeNormalizedSubmodules(base []string, extras []string) []string {
	return mergeDistinct(normalizeSubmoduleList(base), normalizeSubmoduleList(extras))
}

func normalizeSubmoduleList(values []string) []string {
	normalized := make([]string, 0, len(values))
	for _, value := range values {
		trimmed := strings.TrimSpace(strings.ReplaceAll(value, "\\", "/"))
		if trimmed == "" || trimmed == "." {
			continue
		}
		cleanedNative := filepath.Clean(filepath.FromSlash(trimmed))
		cleaned := filepath.ToSlash(cleanedNative)
		if cleaned == "." || cleaned == ".." || strings.HasPrefix(cleaned, "../") || filepath.IsAbs(cleanedNative) {
			continue
		}
		normalized = append(normalized, cleaned)
	}
	return normalized
}

func (manager *Manager) chunkPath(codebaseID string) string {
	return filepath.Join(manager.config.ChunksDir, codebaseID+".json")
}

func (manager *Manager) merklePath(codebaseID string) string {
	return filepath.Join(manager.config.MerkleDir, codebaseID+".json")
}

func (manager *Manager) stagingMerklePath(codebaseID string) string {
	return filepath.Join(manager.config.MerkleDir, codebaseID+".staging.json")
}

func newID(prefix string) string {
	randomBytes := make([]byte, 6)
	if _, err := rand.Read(randomBytes); err != nil {
		return fmt.Sprintf("%s_%d", prefix, clock.Now().UnixNano())
	}
	return fmt.Sprintf("%s_%d_%s", prefix, clock.Now().Unix(), hex.EncodeToString(randomBytes))
}
