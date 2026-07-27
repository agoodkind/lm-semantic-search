package config

import (
	"math"
	"strconv"
	"testing"
	"time"
)

// The collection-load bound is the operator's tuning path for a store whose
// collections legitimately take longer to materialize than the built-in bound
// allows. An omitted field keeps the built-in bound, config.json sets it, and
// the environment variable wins over both.
func TestDefaultResolvesMilvusCollectionLoadTimeout(t *testing.T) {
	t.Run("omitted keeps the built-in bound", func(t *testing.T) {
		t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_TIMEOUT_MS", "")
		cfg := defaultWithPersistedConfig(t, persistedConfig{})
		if cfg.MilvusCollectionLoadTimeoutMS != 0 {
			t.Errorf(
				"MilvusCollectionLoadTimeoutMS = %d want 0 so the store keeps its built-in bound",
				cfg.MilvusCollectionLoadTimeoutMS,
			)
		}
	})

	t.Run("config.json value is used", func(t *testing.T) {
		t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_TIMEOUT_MS", "")
		const fileTimeoutMS = 240000
		cfg := defaultWithPersistedConfig(t, persistedConfig{MilvusCollectionLoadTimeoutMS: fileTimeoutMS})
		if cfg.MilvusCollectionLoadTimeoutMS != fileTimeoutMS {
			t.Errorf("MilvusCollectionLoadTimeoutMS = %d want %d", cfg.MilvusCollectionLoadTimeoutMS, fileTimeoutMS)
		}
	})

	t.Run("environment overrides config.json", func(t *testing.T) {
		const fileTimeoutMS = 240000
		const environmentTimeoutMS = 30000
		t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_TIMEOUT_MS", strconv.Itoa(environmentTimeoutMS))
		cfg := defaultWithPersistedConfig(t, persistedConfig{MilvusCollectionLoadTimeoutMS: fileTimeoutMS})
		if cfg.MilvusCollectionLoadTimeoutMS != environmentTimeoutMS {
			t.Errorf("MilvusCollectionLoadTimeoutMS = %d want %d", cfg.MilvusCollectionLoadTimeoutMS, environmentTimeoutMS)
		}
	})
}

// The range check makes the millisecond-to-duration conversion total. Without
// it, a negative or oversized count wraps into either a multi-century duration
// that removes the bound or a negative duration that fails every wait instantly.
func TestMilvusCollectionLoadTimeoutRejectsUnconvertibleCounts(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		milliseconds int
		want         time.Duration
	}{
		{name: "zero", milliseconds: 0, want: 0},
		{name: "negative", milliseconds: -1, want: 0},
		{name: "most negative", milliseconds: math.MinInt, want: 0},
		{name: "ordinary", milliseconds: 90000, want: 90 * time.Second},
		{name: "largest convertible", milliseconds: int(MaxMilvusCollectionLoadTimeoutMS), want: time.Duration(MaxMilvusCollectionLoadTimeoutMS) * time.Millisecond},
		{name: "one past the largest", milliseconds: int(MaxMilvusCollectionLoadTimeoutMS) + 1, want: 0},
	}
	for _, testCase := range cases {
		got := MilvusCollectionLoadTimeout(testCase.milliseconds)
		if got != testCase.want {
			t.Errorf("%s: MilvusCollectionLoadTimeout(%d) = %s want %s", testCase.name, testCase.milliseconds, got, testCase.want)
		}
		if got < 0 {
			t.Errorf("%s: MilvusCollectionLoadTimeout(%d) = %s, a negative duration means the multiplication wrapped", testCase.name, testCase.milliseconds, got)
		}
	}
}

// An unconvertible count from either source resolves to zero, so the store
// keeps its built-in bound instead of inheriting a wrapped duration.
func TestDefaultRejectsOutOfRangeMilvusCollectionLoadTimeout(t *testing.T) {
	t.Run("file value", func(t *testing.T) {
		t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_TIMEOUT_MS", "")
		cfg := defaultWithPersistedConfig(t, persistedConfig{MilvusCollectionLoadTimeoutMS: -1})
		if cfg.MilvusCollectionLoadTimeoutMS != 0 {
			t.Errorf("MilvusCollectionLoadTimeoutMS = %d want 0 for an unconvertible file value", cfg.MilvusCollectionLoadTimeoutMS)
		}
	})

	t.Run("environment value", func(t *testing.T) {
		t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_TIMEOUT_MS", strconv.FormatInt(MaxMilvusCollectionLoadTimeoutMS+1, 10))
		cfg := defaultWithPersistedConfig(t, persistedConfig{MilvusCollectionLoadTimeoutMS: 90000})
		if cfg.MilvusCollectionLoadTimeoutMS != 0 {
			t.Errorf("MilvusCollectionLoadTimeoutMS = %d want 0 for an unconvertible environment value", cfg.MilvusCollectionLoadTimeoutMS)
		}
	})
}
