package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

func TestParseCommaSeparated(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		input  string
		expect []string
	}{
		{"empty", "", nil},
		{"whitespace only", "  \t  ", nil},
		{"single", "alpha", []string{"alpha"}},
		{"multi", "alpha,bravo,charlie", []string{"alpha", "bravo", "charlie"}},
		{"trimmed", "  alpha , bravo ,charlie  ", []string{"alpha", "bravo", "charlie"}},
		{"skips blanks", "alpha,,bravo,", []string{"alpha", "bravo"}},
	}
	for _, tc := range cases {
		got := parseCommaSeparated(tc.input)
		if !reflect.DeepEqual(got, tc.expect) {
			t.Errorf("parseCommaSeparated(%q) = %#v want %#v", tc.input, got, tc.expect)
		}
	}
}

func TestLoadContextEnvFileSetsMissingKeys(t *testing.T) {
	tempDir := t.TempDir()
	envPath := filepath.Join(tempDir, ".env")
	contents := `
# comment
EMBEDDING_PROVIDER=OpenAI
EMBEDDING_MODEL="text-embedding-3-small"
EMPTYLINE=

OPENAI_API_KEY='sk-fromfile'
`
	if err := os.WriteFile(envPath, []byte(contents), 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	t.Setenv("OPENAI_API_KEY", "sk-from-process")
	t.Setenv("EMBEDDING_PROVIDER", "")
	if err := os.Unsetenv("EMBEDDING_PROVIDER"); err != nil {
		t.Fatalf("Unsetenv returned error: %v", err)
	}
	if err := os.Unsetenv("EMBEDDING_MODEL"); err != nil {
		t.Fatalf("Unsetenv returned error: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Unsetenv("EMBEDDING_PROVIDER")
		_ = os.Unsetenv("EMBEDDING_MODEL")
		_ = os.Unsetenv("EMPTYLINE")
	})

	loadContextEnvFile(envPath)

	if got := os.Getenv("EMBEDDING_PROVIDER"); got != "OpenAI" {
		t.Errorf("EMBEDDING_PROVIDER = %q want OpenAI", got)
	}
	if got := os.Getenv("EMBEDDING_MODEL"); got != "text-embedding-3-small" {
		t.Errorf("EMBEDDING_MODEL = %q want text-embedding-3-small (quotes stripped)", got)
	}
	if got := os.Getenv("OPENAI_API_KEY"); got != "sk-from-process" {
		t.Errorf("OPENAI_API_KEY = %q want sk-from-process (process env wins over .env file)", got)
	}
}

func TestDefaultReadsCustomIgnoreEnvVar(t *testing.T) {
	tempState := t.TempDir()
	tempHome := t.TempDir()
	t.Setenv("HOME", tempHome)
	t.Setenv("CLAUDE_CONTEXTD_STATE_ROOT", tempState)
	t.Setenv("CUSTOM_IGNORE_PATTERNS", "vendor/**, third_party/**")

	cfg, err := Default()
	if err != nil {
		t.Fatalf("Default returned error: %v", err)
	}
	if !reflect.DeepEqual(cfg.CustomIgnorePatterns, []string{"vendor/**", "third_party/**"}) {
		t.Errorf("CustomIgnorePatterns = %#v", cfg.CustomIgnorePatterns)
	}
}

// isolateState points HOME and the state root at temp dirs so Default() never
// reads the real machine's ~/.context or ~/.lm-semantic-search state during the test.
func isolateState(t *testing.T) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("CLAUDE_CONTEXTD_STATE_ROOT", t.TempDir())
}

func defaultWithPersistedConfig(t *testing.T, fileConfig persistedConfig) Config {
	t.Helper()
	isolateState(t)
	t.Setenv("EMBEDDING_MODEL", "")
	configRoot := t.TempDir()
	t.Setenv("CLAUDE_CONTEXTD_CONFIG_ROOT", configRoot)

	data, err := json.Marshal(fileConfig)
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}
	configPath := filepath.Join(configRoot, "config.json")
	if err := os.WriteFile(configPath, data, 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	cfg, err := Default()
	if err != nil {
		t.Fatalf("Default returned error: %v", err)
	}
	return cfg
}

func TestDefaultDebugAndJobControlDefaults(t *testing.T) {
	isolateState(t)

	cfg, err := Default()
	if err != nil {
		t.Fatalf("Default returned error: %v", err)
	}

	if !cfg.DebugListenerEnabled {
		t.Errorf("DebugListenerEnabled = false want true")
	}
	if cfg.DebugListenAddr != defaultDebugListenAddr {
		t.Errorf("DebugListenAddr = %q want %q", cfg.DebugListenAddr, defaultDebugListenAddr)
	}
	if cfg.PerfCountersIntervalMS != defaultPerfCountersIntervalMS {
		t.Errorf("PerfCountersIntervalMS = %d want %d", cfg.PerfCountersIntervalMS, defaultPerfCountersIntervalMS)
	}
	if cfg.MaxConcurrentIndexJobs != defaultMaxConcurrentIndexJobs {
		t.Errorf("MaxConcurrentIndexJobs = %d want %d", cfg.MaxConcurrentIndexJobs, defaultMaxConcurrentIndexJobs)
	}
	if !cfg.ResumeIndexingOnBoot {
		t.Errorf("ResumeIndexingOnBoot = false want true")
	}
}

func TestDefaultEmbeddingBatchTokenBudgetDefaultsTo6000(t *testing.T) {
	cfg := defaultWithPersistedConfig(t, persistedConfig{})

	if cfg.EmbeddingBatchTokenBudget != 6000 {
		t.Errorf("EmbeddingBatchTokenBudget = %d want 6000", cfg.EmbeddingBatchTokenBudget)
	}
}

func TestDefaultQueryInstructionPrefixForNVEmbedCodeModel(t *testing.T) {
	cfg := defaultWithPersistedConfig(t, persistedConfig{
		EmbeddingModel: "nvidia/NV-EmbedCode-7b-v1",
	})
	wantPrefix := "Instruct: Retrieve code or text relevant to the query.\nQuery: "

	if cfg.QueryInstructionPrefix != wantPrefix {
		t.Errorf("QueryInstructionPrefix = %q want %q", cfg.QueryInstructionPrefix, wantPrefix)
	}
}

func TestDefaultQueryInstructionPrefixEmptyForOtherModels(t *testing.T) {
	cfg := defaultWithPersistedConfig(t, persistedConfig{
		EmbeddingModel: "text-embedding-3-small",
	})

	if cfg.QueryInstructionPrefix != "" {
		t.Errorf("QueryInstructionPrefix = %q want empty", cfg.QueryInstructionPrefix)
	}
}

func intPtr(value int) *int {
	return &value
}

func TestDefaultEmbeddingRequestTimeoutDefaultsAndPersists(t *testing.T) {
	defaulted := defaultWithPersistedConfig(t, persistedConfig{})
	if defaulted.EmbeddingRequestTimeoutMS != defaultEmbeddingRequestTimeoutMS {
		t.Errorf("omitted EmbeddingRequestTimeoutMS = %d want default %d", defaulted.EmbeddingRequestTimeoutMS, defaultEmbeddingRequestTimeoutMS)
	}

	persisted := defaultWithPersistedConfig(t, persistedConfig{
		EmbeddingRequestTimeoutMS: intPtr(45000),
	})
	if persisted.EmbeddingRequestTimeoutMS != 45000 {
		t.Errorf("persisted EmbeddingRequestTimeoutMS = %d want 45000", persisted.EmbeddingRequestTimeoutMS)
	}

	// An explicit 0 in config.json disables the bound and must survive as 0,
	// distinct from an omitted field that falls back to the default.
	disabled := defaultWithPersistedConfig(t, persistedConfig{
		EmbeddingRequestTimeoutMS: intPtr(0),
	})
	if disabled.EmbeddingRequestTimeoutMS != 0 {
		t.Errorf("persisted zero EmbeddingRequestTimeoutMS = %d want 0 (disabled)", disabled.EmbeddingRequestTimeoutMS)
	}
}

func TestDefaultEmbeddingRequestTimeoutEnvOverridesPersisted(t *testing.T) {
	t.Setenv("CLAUDE_CONTEXT_EMBEDDING_REQUEST_TIMEOUT_MS", "12000")
	cfg := defaultWithPersistedConfig(t, persistedConfig{
		EmbeddingRequestTimeoutMS: intPtr(45000),
	})
	if cfg.EmbeddingRequestTimeoutMS != 12000 {
		t.Errorf("EmbeddingRequestTimeoutMS = %d want env override 12000", cfg.EmbeddingRequestTimeoutMS)
	}
}

func TestDefaultEmbeddingBatchConfigUsesPersistedValues(t *testing.T) {
	cfg := defaultWithPersistedConfig(t, persistedConfig{
		EmbeddingModel:            "nvidia/NV-EmbedCode-7b-v1",
		EmbeddingBatchTokenBudget: 4096,
		QueryInstructionPrefix:    "custom query prefix: ",
	})

	if cfg.EmbeddingBatchTokenBudget != 4096 {
		t.Errorf("EmbeddingBatchTokenBudget = %d want 4096", cfg.EmbeddingBatchTokenBudget)
	}
	if cfg.QueryInstructionPrefix != "custom query prefix: " {
		t.Errorf("QueryInstructionPrefix = %q want custom query prefix", cfg.QueryInstructionPrefix)
	}
}

func TestDefaultOfflineEmbeddingModelDefaultsAndReadsConfig(t *testing.T) {
	defaulted := defaultWithPersistedConfig(t, persistedConfig{})
	if defaulted.OfflineEmbeddingModel != offlinemodel.EmbeddingGemma {
		t.Fatalf(
			"default OfflineEmbeddingModel = %q, want %q",
			defaulted.OfflineEmbeddingModel,
			offlinemodel.EmbeddingGemma,
		)
	}

	persisted := defaultWithPersistedConfig(t, persistedConfig{
		OfflineEmbeddingModel: offlinemodel.BGESmall,
	})
	if persisted.OfflineEmbeddingModel != offlinemodel.BGESmall {
		t.Fatalf(
			"persisted OfflineEmbeddingModel = %q, want %q",
			persisted.OfflineEmbeddingModel,
			offlinemodel.BGESmall,
		)
	}
}

func TestDefaultOfflineEmbeddingModelEnvOverridesConfig(t *testing.T) {
	t.Setenv("OFFLINE_EMBEDDING_MODEL", offlinemodel.EmbeddingGemma)
	cfg := defaultWithPersistedConfig(t, persistedConfig{
		OfflineEmbeddingModel: offlinemodel.BGESmall,
	})
	if cfg.OfflineEmbeddingModel != offlinemodel.EmbeddingGemma {
		t.Fatalf(
			"OfflineEmbeddingModel = %q, want env override %q",
			cfg.OfflineEmbeddingModel,
			offlinemodel.EmbeddingGemma,
		)
	}
}

func TestDefaultKeepsDaemonStateAndCompatRootsSplit(t *testing.T) {
	tempHome := t.TempDir()
	t.Setenv("HOME", tempHome)
	// Neutralize the XDG overrides so this test exercises the home-directory
	// fallback it asserts. CI runners set XDG_CONFIG_HOME/XDG_STATE_HOME, which
	// resolveXDGConfigHome/resolveXDGStateHome honor, so without this the roots
	// resolve to the runner's real home instead of tempHome.
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("XDG_STATE_HOME", "")
	if err := os.Unsetenv("CLAUDE_CONTEXTD_STATE_ROOT"); err != nil {
		t.Fatalf("Unsetenv returned error: %v", err)
	}
	if err := os.Unsetenv("CLAUDE_CONTEXTD_CONFIG_ROOT"); err != nil {
		t.Fatalf("Unsetenv returned error: %v", err)
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("UserHomeDir returned error: %v", err)
	}

	cfg, err := Default()
	if err != nil {
		t.Fatalf("Default returned error: %v", err)
	}

	wantConfigRoot := filepath.Join(homeDir, ".config", "lm-semantic-search")
	if cfg.ConfigRoot != wantConfigRoot {
		t.Fatalf("ConfigRoot = %q, want %q", cfg.ConfigRoot, wantConfigRoot)
	}
	if cfg.ConfigPath != filepath.Join(wantConfigRoot, "config.json") {
		t.Fatalf("ConfigPath = %q", cfg.ConfigPath)
	}
	wantStateRoot := filepath.Join(homeDir, ".local", "state", "lm-semantic-search")
	if cfg.StateRoot != wantStateRoot {
		t.Fatalf("StateRoot = %q, want %q", cfg.StateRoot, wantStateRoot)
	}
	wantContextRoot := filepath.Join(homeDir, ".context")
	if cfg.ContextRoot != wantContextRoot {
		t.Fatalf("ContextRoot = %q, want %q", cfg.ContextRoot, wantContextRoot)
	}
	if cfg.RegistryPath != filepath.Join(wantStateRoot, "registry.json") {
		t.Fatalf("RegistryPath = %q", cfg.RegistryPath)
	}
	if cfg.MerkleDir != filepath.Join(wantStateRoot, "merkle") {
		t.Fatalf("MerkleDir = %q", cfg.MerkleDir)
	}
	if cfg.ChunksDir != filepath.Join(wantStateRoot, "chunks") {
		t.Fatalf("ChunksDir = %q", cfg.ChunksDir)
	}
	if cfg.GraphDir != filepath.Join(wantStateRoot, "graph") {
		t.Fatalf("GraphDir = %q", cfg.GraphDir)
	}
	if cfg.SocketPath != filepath.Join(wantStateRoot, "sockets", "lm-semantic-search-daemon.sock") {
		t.Fatalf("SocketPath = %q", cfg.SocketPath)
	}
}

func TestDefaultUsesXDGRootsWhenSet(t *testing.T) {
	tempHome := t.TempDir()
	t.Setenv("HOME", tempHome)
	xdgConfig := filepath.Join(tempHome, "xdg-config")
	xdgState := filepath.Join(tempHome, "xdg-state")
	t.Setenv("XDG_CONFIG_HOME", xdgConfig)
	t.Setenv("XDG_STATE_HOME", xdgState)
	if err := os.Unsetenv("CLAUDE_CONTEXTD_STATE_ROOT"); err != nil {
		t.Fatalf("Unsetenv returned error: %v", err)
	}
	if err := os.Unsetenv("CLAUDE_CONTEXTD_CONFIG_ROOT"); err != nil {
		t.Fatalf("Unsetenv returned error: %v", err)
	}

	cfg, err := Default()
	if err != nil {
		t.Fatalf("Default returned error: %v", err)
	}

	if cfg.ConfigRoot != filepath.Join(xdgConfig, "lm-semantic-search") {
		t.Fatalf("ConfigRoot = %q", cfg.ConfigRoot)
	}
	if cfg.StateRoot != filepath.Join(xdgState, "lm-semantic-search") {
		t.Fatalf("StateRoot = %q", cfg.StateRoot)
	}
}

func TestDefaultDebugAndJobControlEnvOverrides(t *testing.T) {
	isolateState(t)

	t.Setenv("CLAUDE_CONTEXT_DEBUG_LISTENER", "false")
	t.Setenv("CLAUDE_CONTEXT_DEBUG_LISTEN_ADDR", "127.0.0.1:7000")
	t.Setenv("CLAUDE_CONTEXT_PERF_COUNTERS_INTERVAL_MS", "0")
	t.Setenv("CLAUDE_CONTEXT_MAX_CONCURRENT_INDEX_JOBS", "8")
	t.Setenv("CLAUDE_CONTEXT_RESUME_ON_BOOT", "false")

	cfg, err := Default()
	if err != nil {
		t.Fatalf("Default returned error: %v", err)
	}

	if cfg.DebugListenerEnabled {
		t.Errorf("DebugListenerEnabled = true want false")
	}
	if cfg.DebugListenAddr != "127.0.0.1:7000" {
		t.Errorf("DebugListenAddr = %q want 127.0.0.1:7000", cfg.DebugListenAddr)
	}
	if cfg.PerfCountersIntervalMS != 0 {
		t.Errorf("PerfCountersIntervalMS = %d want 0", cfg.PerfCountersIntervalMS)
	}
	if cfg.MaxConcurrentIndexJobs != 8 {
		t.Errorf("MaxConcurrentIndexJobs = %d want 8", cfg.MaxConcurrentIndexJobs)
	}
	if cfg.ResumeIndexingOnBoot {
		t.Errorf("ResumeIndexingOnBoot = true want false")
	}
}

func TestDefaultLogRotationAndCleanupDefaults(t *testing.T) {
	isolateState(t)

	cfg, err := Default()
	if err != nil {
		t.Fatalf("Default returned error: %v", err)
	}

	if cfg.LogRotationMaxBytes != defaultLogRotationMaxBytes {
		t.Errorf("LogRotationMaxBytes = %d want %d", cfg.LogRotationMaxBytes, defaultLogRotationMaxBytes)
	}
	if cfg.LogRetentionBytes != defaultLogRetentionBytes {
		t.Errorf("LogRetentionBytes = %d want %d", cfg.LogRetentionBytes, defaultLogRetentionBytes)
	}
	if !cfg.LogCleanupEnabled {
		t.Errorf("LogCleanupEnabled = false want true")
	}
	if cfg.LogCleanupIntervalMS != defaultLogCleanupIntervalMS {
		t.Errorf("LogCleanupIntervalMS = %d want %d", cfg.LogCleanupIntervalMS, defaultLogCleanupIntervalMS)
	}
}

func TestDefaultLogRotationAndCleanupEnvOverrides(t *testing.T) {
	isolateState(t)

	t.Setenv("CLAUDE_CONTEXT_LOG_ROTATION_MAX_BYTES", "1048576")
	t.Setenv("CLAUDE_CONTEXT_LOG_RETENTION_BYTES", "10485760")
	t.Setenv("CLAUDE_CONTEXT_LOG_CLEANUP_ENABLED", "false")
	t.Setenv("CLAUDE_CONTEXT_LOG_CLEANUP_INTERVAL_MS", "60000")

	cfg, err := Default()
	if err != nil {
		t.Fatalf("Default returned error: %v", err)
	}

	if cfg.LogRotationMaxBytes != 1048576 {
		t.Errorf("LogRotationMaxBytes = %d want 1048576", cfg.LogRotationMaxBytes)
	}
	if cfg.LogRetentionBytes != 10485760 {
		t.Errorf("LogRetentionBytes = %d want 10485760", cfg.LogRetentionBytes)
	}
	if cfg.LogCleanupEnabled {
		t.Errorf("LogCleanupEnabled = true want false")
	}
	if cfg.LogCleanupIntervalMS != 60000 {
		t.Errorf("LogCleanupIntervalMS = %d want 60000", cfg.LogCleanupIntervalMS)
	}
}
