package config

import (
	"bytes"
	"log/slog"
	"strconv"
	"strings"
	"testing"
)

// The concurrent load cap is the operator's lever against a Milvus that runs
// out of memory when the daemon asks for too many collections at once. An
// omitted field keeps the default, config.json sets it, and the environment
// variable wins over both.
func TestDefaultResolvesMilvusMaxConcurrentCollectionLoads(t *testing.T) {
	t.Run("omitted keeps the default", func(t *testing.T) {
		t.Setenv("CLAUDE_CONTEXT_MILVUS_MAX_CONCURRENT_COLLECTION_LOADS", "")
		cfg := defaultWithPersistedConfig(t, persistedConfig{})
		if cfg.MilvusMaxConcurrentCollectionLoads != 2 {
			t.Errorf("MilvusMaxConcurrentCollectionLoads = %d want 2", cfg.MilvusMaxConcurrentCollectionLoads)
		}
	})

	t.Run("config.json value is used", func(t *testing.T) {
		t.Setenv("CLAUDE_CONTEXT_MILVUS_MAX_CONCURRENT_COLLECTION_LOADS", "")
		cfg := defaultWithPersistedConfig(t, persistedConfig{MilvusMaxConcurrentCollectionLoads: 4})
		if cfg.MilvusMaxConcurrentCollectionLoads != 4 {
			t.Errorf("MilvusMaxConcurrentCollectionLoads = %d want 4", cfg.MilvusMaxConcurrentCollectionLoads)
		}
	})

	t.Run("environment overrides config.json", func(t *testing.T) {
		t.Setenv("CLAUDE_CONTEXT_MILVUS_MAX_CONCURRENT_COLLECTION_LOADS", "1")
		cfg := defaultWithPersistedConfig(t, persistedConfig{MilvusMaxConcurrentCollectionLoads: 4})
		if cfg.MilvusMaxConcurrentCollectionLoads != 1 {
			t.Errorf("MilvusMaxConcurrentCollectionLoads = %d want 1", cfg.MilvusMaxConcurrentCollectionLoads)
		}
	})
}

// A cap below one would park every load forever, so a negative or unparsable
// value from either source keeps the default and the warning names the knob.
func TestDefaultRejectsUnusableMilvusMaxConcurrentCollectionLoads(t *testing.T) {
	testCases := []struct {
		name        string
		environment string
		fileValue   int
	}{
		{name: "negative file value", environment: "", fileValue: -1},
		{name: "negative environment value", environment: strconv.Itoa(-3), fileValue: 4},
		{name: "unparsable environment value", environment: "two", fileValue: 4},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Setenv("CLAUDE_CONTEXT_MILVUS_MAX_CONCURRENT_COLLECTION_LOADS", testCase.environment)

			var logs bytes.Buffer
			previousLogger := slog.Default()
			slog.SetDefault(slog.New(slog.NewTextHandler(&logs, nil)))
			t.Cleanup(func() {
				slog.SetDefault(previousLogger)
			})

			cfg := defaultWithPersistedConfig(t, persistedConfig{
				MilvusMaxConcurrentCollectionLoads: testCase.fileValue,
			})
			if cfg.MilvusMaxConcurrentCollectionLoads != 2 {
				t.Errorf("MilvusMaxConcurrentCollectionLoads = %d want 2", cfg.MilvusMaxConcurrentCollectionLoads)
			}
			if !strings.Contains(logs.String(), "config_field=milvusMaxConcurrentCollectionLoads") {
				t.Errorf("warning does not name config field: %s", logs.String())
			}
			if !strings.Contains(logs.String(), "env_var=CLAUDE_CONTEXT_MILVUS_MAX_CONCURRENT_COLLECTION_LOADS") {
				t.Errorf("warning does not name environment variable: %s", logs.String())
			}
		})
	}
}
