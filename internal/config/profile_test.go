package config

import "testing"

func TestApplyProfileOfflineExpandsBackendAndEmbedder(t *testing.T) {
	t.Parallel()
	got := ApplyProfile(Config{Profile: ProfileOffline})
	if got.IndexBackend != IndexBackendLocal {
		t.Fatalf("IndexBackend = %q, want %q", got.IndexBackend, IndexBackendLocal)
	}
	if got.EmbeddingProvider != EmbeddingProviderONNX {
		t.Fatalf("EmbeddingProvider = %q, want %q", got.EmbeddingProvider, EmbeddingProviderONNX)
	}
	if got.EmbeddingDimension != 384 {
		t.Fatalf("EmbeddingDimension = %d, want 384", got.EmbeddingDimension)
	}
}

func TestApplyProfileStandardIsDefaultUnchanged(t *testing.T) {
	t.Parallel()
	base := Config{Profile: ProfileStandard, EmbeddingProvider: "OpenAI", IndexBackend: IndexBackendMilvus, MilvusAddress: "127.0.0.1:19530", MilvusToken: "tok"}
	got := ApplyProfile(base)
	if got.IndexBackend != IndexBackendMilvus || got.EmbeddingProvider != "OpenAI" {
		t.Fatalf("standard profile mutated backend or embedder: %+v", got)
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
