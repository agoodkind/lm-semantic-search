package config

import (
	"math"
	"strconv"
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
