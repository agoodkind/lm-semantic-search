package config

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
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

func TestDefaultReadsCustomExtensionAndIgnoreEnvVars(t *testing.T) {
	tempState := t.TempDir()
	tempHome := t.TempDir()
	t.Setenv("HOME", tempHome)
	t.Setenv("CLAUDE_CONTEXTD_STATE_ROOT", tempState)
	t.Setenv("CUSTOM_EXTENSIONS", ".toml,.yaml")
	t.Setenv("CUSTOM_IGNORE_PATTERNS", "vendor/**, third_party/**")

	cfg, err := Default()
	if err != nil {
		t.Fatalf("Default returned error: %v", err)
	}
	if !reflect.DeepEqual(cfg.CustomExtensions, []string{".toml", ".yaml"}) {
		t.Errorf("CustomExtensions = %#v", cfg.CustomExtensions)
	}
	if !reflect.DeepEqual(cfg.CustomIgnorePatterns, []string{"vendor/**", "third_party/**"}) {
		t.Errorf("CustomIgnorePatterns = %#v", cfg.CustomIgnorePatterns)
	}
}

// isolateState points HOME and the state root at temp dirs so Default() never
// reads the real machine's ~/.context or ~/.contextd state during the test.
func isolateState(t *testing.T) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("CLAUDE_CONTEXTD_STATE_ROOT", t.TempDir())
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
