package config

import (
	"bytes"
	"log/slog"
	"math"
	"strconv"
	"strings"
	"testing"
)

func TestDefaultResolvesMilvusCollectionResidencyDefaults(t *testing.T) {
	t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_WAIT_TIMEOUT_MS", "")
	t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_IDLE_TIMEOUT_MS", "")

	omitted := defaultWithPersistedConfig(t, persistedConfig{})
	if omitted.MilvusCollectionLoadWaitTimeoutMS != 15000 {
		t.Errorf(
			"MilvusCollectionLoadWaitTimeoutMS = %d want 15000",
			omitted.MilvusCollectionLoadWaitTimeoutMS,
		)
	}
	if omitted.MilvusCollectionIdleTimeoutMS != 900000 {
		t.Errorf(
			"MilvusCollectionIdleTimeoutMS = %d want 900000",
			omitted.MilvusCollectionIdleTimeoutMS,
		)
	}

	zero := 0
	disabled := defaultWithPersistedConfig(t, persistedConfig{
		MilvusCollectionIdleTimeoutMS: &zero,
	})
	if disabled.MilvusCollectionIdleTimeoutMS != 0 {
		t.Errorf(
			"MilvusCollectionIdleTimeoutMS = %d want 0",
			disabled.MilvusCollectionIdleTimeoutMS,
		)
	}
}

func TestDefaultResolvesMilvusCollectionResidencyPrecedence(t *testing.T) {
	idleFileValue := 120000
	t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_WAIT_TIMEOUT_MS", "30000")
	t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_IDLE_TIMEOUT_MS", "0")

	cfg := defaultWithPersistedConfig(t, persistedConfig{
		MilvusCollectionLoadWaitTimeoutMS: 20000,
		MilvusCollectionIdleTimeoutMS:     &idleFileValue,
	})
	if cfg.MilvusCollectionLoadWaitTimeoutMS != 30000 {
		t.Errorf(
			"MilvusCollectionLoadWaitTimeoutMS = %d want 30000",
			cfg.MilvusCollectionLoadWaitTimeoutMS,
		)
	}
	if cfg.MilvusCollectionIdleTimeoutMS != 0 {
		t.Errorf(
			"MilvusCollectionIdleTimeoutMS = %d want 0",
			cfg.MilvusCollectionIdleTimeoutMS,
		)
	}
}

func TestDefaultRejectsInvalidMilvusCollectionResidencyTimeouts(t *testing.T) {
	invalidValues := []int{-1, math.MaxInt}
	for _, invalidValue := range invalidValues {
		t.Run(strconv.Itoa(invalidValue), func(t *testing.T) {
			t.Setenv(
				"CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_WAIT_TIMEOUT_MS",
				strconv.Itoa(invalidValue),
			)
			t.Setenv(
				"CLAUDE_CONTEXT_MILVUS_COLLECTION_IDLE_TIMEOUT_MS",
				strconv.Itoa(invalidValue),
			)
			cfg := defaultWithPersistedConfig(t, persistedConfig{})
			if cfg.MilvusCollectionLoadWaitTimeoutMS != 15000 {
				t.Errorf(
					"MilvusCollectionLoadWaitTimeoutMS = %d want 15000",
					cfg.MilvusCollectionLoadWaitTimeoutMS,
				)
			}
			if cfg.MilvusCollectionIdleTimeoutMS != 900000 {
				t.Errorf(
					"MilvusCollectionIdleTimeoutMS = %d want 900000",
					cfg.MilvusCollectionIdleTimeoutMS,
				)
			}
		})
	}
}

func TestDefaultWarnsForMalformedMilvusResidencyEnvironmentValues(t *testing.T) {
	testCases := []struct {
		name        string
		environment string
		value       string
		configField string
	}{
		{
			name:        "load wait overflow",
			environment: "CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_WAIT_TIMEOUT_MS",
			value:       strconv.FormatUint(math.MaxUint64, 10),
			configField: "milvusCollectionLoadWaitTimeoutMs",
		},
		{
			name:        "idle parse failure",
			environment: "CLAUDE_CONTEXT_MILVUS_COLLECTION_IDLE_TIMEOUT_MS",
			value:       "not-a-millisecond-count",
			configField: "milvusCollectionIdleTimeoutMs",
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_WAIT_TIMEOUT_MS", "")
			t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_IDLE_TIMEOUT_MS", "")
			t.Setenv(testCase.environment, testCase.value)

			var logs bytes.Buffer
			previousLogger := slog.Default()
			slog.SetDefault(slog.New(slog.NewTextHandler(&logs, nil)))
			t.Cleanup(func() {
				slog.SetDefault(previousLogger)
			})

			cfg := defaultWithPersistedConfig(t, persistedConfig{})
			if cfg.MilvusCollectionLoadWaitTimeoutMS != 15000 {
				t.Errorf(
					"MilvusCollectionLoadWaitTimeoutMS = %d want 15000",
					cfg.MilvusCollectionLoadWaitTimeoutMS,
				)
			}
			if cfg.MilvusCollectionIdleTimeoutMS != 900000 {
				t.Errorf(
					"MilvusCollectionIdleTimeoutMS = %d want 900000",
					cfg.MilvusCollectionIdleTimeoutMS,
				)
			}
			if !strings.Contains(logs.String(), "config_field="+testCase.configField) {
				t.Errorf("warning does not name config field: %s", logs.String())
			}
			if !strings.Contains(logs.String(), "env_var="+testCase.environment) {
				t.Errorf("warning does not name environment variable: %s", logs.String())
			}
		})
	}
}
