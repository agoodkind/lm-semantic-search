//go:build milvusintegration

package milvusintegration

import (
	"context"
	"os"
	"strconv"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
	"goodkind.io/lm-semantic-search/internal/sandbox"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/store"
)

// integrationConfig resolves a daemon configuration through the same
// config.Default the installed daemon uses, rooted in a throwaway directory by
// the sandbox defaults and pointed at the throwaway Milvus. The embedder is the
// real in-process ONNX model from the machine's model cache, so no part of the
// pipeline is stood in for. settings override or add environment values on
// top of that base.
func integrationConfig(t *testing.T, stack throwawayStack, settings map[string]string) config.Config {
	t.Helper()
	preset, err := offlinemodel.Resolve(offlinemodel.DefaultName)
	if err != nil {
		t.Fatalf("resolve offline embedding preset: %v", err)
	}
	// The unix socket path must fit the platform's sun_path limit, and
	// t.TempDir lives under a long path, so the socket gets a short directory.
	socketDir, err := os.MkdirTemp("/tmp", "lms-itest-")
	if err != nil {
		t.Fatalf("mkdir short socket dir: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(socketDir) })

	chosen := map[string]string{
		"CLAUDE_CONTEXT_PROFILE":                     config.ProfileStandard,
		"MILVUS_ADDRESS":                             stack.milvusAddress(),
		"MILVUS_TOKEN":                               "",
		"MILVUS_DATABASE":                            "",
		"EMBEDDING_PROVIDER":                         string(config.EmbeddingProviderONNX),
		"OFFLINE_EMBEDDING_MODEL":                    preset.Name,
		"EMBEDDING_MODEL":                            preset.Name,
		"EMBEDDING_DIMENSION":                        strconv.FormatInt(int64(preset.Dimension), 10),
		"EMBEDDING_BATCH_SIZE":                       "8",
		"CLAUDE_CONTEXTD_SOCKET_PATH":                socketDir + "/daemon.sock",
		"CLAUDE_CONTEXT_BACKGROUND_SYNC":             "false",
		"CLAUDE_CONTEXT_TRIGGER_WATCHER":             "false",
		"CLAUDE_CONTEXT_FILE_WATCHER":                "false",
		"CLAUDE_CONTEXT_DEBUG_LISTENER":              "false",
		"CLAUDE_CONTEXT_PERF_COUNTERS_INTERVAL_MS":   "0",
		"CLAUDE_CONTEXT_MAX_CONCURRENT_INDEX_JOBS":   "1",
		"CLAUDE_CONTEXT_RESUME_ON_BOOT":              "false",
		"CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_WAIT_TIMEOUT_MS": "120000",
	}
	for name, value := range settings {
		chosen[name] = value
	}
	for name, value := range chosen {
		t.Setenv(name, value)
	}
	for _, variable := range sandbox.Env(t.TempDir()) {
		if _, alreadySet := os.LookupEnv(variable.Name); alreadySet {
			continue
		}
		t.Setenv(variable.Name, variable.Value)
	}
	resolved, err := config.Default()
	if err != nil {
		t.Fatalf("resolve integration config through config.Default: %v", err)
	}
	if resolved.MilvusAddress != stack.milvusAddress() {
		t.Fatalf("resolved MilvusAddress = %q, want the throwaway stack %q", resolved.MilvusAddress, stack.milvusAddress())
	}
	if resolved.EmbeddingProvider != config.EmbeddingProviderONNX {
		t.Fatalf("resolved EmbeddingProvider = %q, want the in-process ONNX embedder", resolved.EmbeddingProvider)
	}
	for _, dir := range sandbox.Directories(resolved) {
		if err := store.EnsureDir(dir); err != nil {
			t.Fatalf("EnsureDir(%s): %v", dir, err)
		}
	}
	return resolved
}

// newService builds the production semantic service against the throwaway
// stack, with every Milvus call it makes reported to recorder, and waits until
// it has connected.
func newService(t *testing.T, cfg config.Config, recorder *milvusCallRecorder) *semantic.Service {
	t.Helper()
	service, err := semantic.NewService(observedContext(recorder), cfg)
	if err != nil {
		t.Fatalf("semantic.NewService: %v", err)
	}
	t.Cleanup(func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := service.Close(closeCtx); err != nil {
			t.Errorf("close semantic service: %v", err)
		}
	})
	deadline := time.Now().Add(stackReadyTimeout)
	for !service.Available() {
		if time.Now().After(deadline) {
			t.Fatalf("semantic service did not connect to %s within %s", cfg.MilvusAddress, stackReadyTimeout)
		}
		time.Sleep(milvusReadyPoll)
	}
	return service
}
