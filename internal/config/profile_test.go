package config

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

func TestApplyProfileOfflineExpandsBackendAndEmbedder(t *testing.T) {
	t.Parallel()
	got := ApplyProfile(Config{Profile: ProfileOffline})
	if got.IndexBackend != IndexBackendLocal {
		t.Fatalf("IndexBackend = %q, want %q", got.IndexBackend, IndexBackendLocal)
	}
	if got.EmbeddingProvider != EmbeddingProviderONNX {
		t.Fatalf("EmbeddingProvider = %q, want %q", got.EmbeddingProvider, EmbeddingProviderONNX)
	}
	if got.OfflineEmbeddingModel != offlinemodel.EmbeddingGemma {
		t.Fatalf(
			"OfflineEmbeddingModel = %q, want %q",
			got.OfflineEmbeddingModel,
			offlinemodel.EmbeddingGemma,
		)
	}
	if got.EmbeddingModel != offlinemodel.EmbeddingGemma {
		t.Fatalf("EmbeddingModel = %q, want %q", got.EmbeddingModel, offlinemodel.EmbeddingGemma)
	}
	if got.EmbeddingDimension != 768 {
		t.Fatalf("EmbeddingDimension = %d, want 768", got.EmbeddingDimension)
	}
	if got.QueryInstructionPrefix != "task: code retrieval | query: " {
		t.Fatalf("QueryInstructionPrefix = %q", got.QueryInstructionPrefix)
	}
}

func TestApplyProfileOfflineDerivesSelectedModel(t *testing.T) {
	t.Parallel()

	got := ApplyProfile(Config{
		Profile:               ProfileOffline,
		OfflineEmbeddingModel: offlinemodel.BGESmall,
	})
	if got.EmbeddingModel != offlinemodel.BGESmall {
		t.Fatalf("EmbeddingModel = %q, want %q", got.EmbeddingModel, offlinemodel.BGESmall)
	}
	if got.EmbeddingDimension != 384 {
		t.Fatalf("EmbeddingDimension = %d, want 384", got.EmbeddingDimension)
	}
	if got.QueryInstructionPrefix != "" {
		t.Fatalf("QueryInstructionPrefix = %q, want empty", got.QueryInstructionPrefix)
	}
}

func TestApplyProfileStandardIsDefaultUnchanged(t *testing.T) {
	t.Parallel()
	base := Config{
		Profile:               ProfileStandard,
		EmbeddingProvider:     "OpenAI",
		EmbeddingModel:        "text-embedding-3-small",
		EmbeddingDimension:    1536,
		OfflineEmbeddingModel: offlinemodel.BGESmall,
		IndexBackend:          IndexBackendMilvus,
		MilvusAddress:         "127.0.0.1:19530",
		MilvusToken:           "tok",
	}
	got := ApplyProfile(base)
	if got.IndexBackend != IndexBackendMilvus || got.EmbeddingProvider != "OpenAI" {
		t.Fatalf("standard profile mutated backend or embedder: %+v", got)
	}
	if got.EmbeddingModel != "text-embedding-3-small" || got.EmbeddingDimension != 1536 {
		t.Fatalf("standard profile mutated embedding model: %+v", got)
	}
	if got.MilvusAddress != "127.0.0.1:19530" || got.MilvusToken != "tok" {
		t.Fatalf("standard profile cleared Milvus config: addr=%q token=%q", got.MilvusAddress, got.MilvusToken)
	}
}

func TestApplyProfileEmptyProfileFillsDefaultBackend(t *testing.T) {
	t.Parallel()
	got := ApplyProfile(Config{EmbeddingProvider: "OpenAI"})
	if got.IndexBackend != IndexBackendMilvus {
		t.Fatalf("empty profile IndexBackend = %q, want %q", got.IndexBackend, IndexBackendMilvus)
	}
}

func TestApplyProfileOfflineClearsPrepopulatedMilvus(t *testing.T) {
	t.Parallel()
	base := Config{Profile: ProfileOffline, MilvusAddress: "127.0.0.1:19530", MilvusToken: "tok"}
	got := ApplyProfile(base)
	if got.MilvusAddress != "" {
		t.Fatalf("offline profile left MilvusAddress = %q, want empty", got.MilvusAddress)
	}
	if got.MilvusToken != "" {
		t.Fatalf("offline profile left MilvusToken = %q, want empty", got.MilvusToken)
	}
}
