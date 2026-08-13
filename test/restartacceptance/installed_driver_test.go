//go:build restartacceptance

package restartacceptance

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
)

func TestInstalledLMSForcesProductionMetadataTimeoutAcrossProcessBoundary(t *testing.T) {
	configHome := t.TempDir()
	configRoot := filepath.Join(configHome, "lm-semantic-search")
	if err := os.MkdirAll(configRoot, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(configRoot, "config.json"),
		[]byte(`{"milvusMetadataCallTimeoutMs":2}`),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	t.Setenv("XDG_CONFIG_HOME", configHome)
	t.Setenv("CLAUDE_CONTEXT_MILVUS_METADATA_CALL_TIMEOUT_MS", "1")

	run := acceptanceRun{Paths: pathsForRun(t.TempDir())}
	process := installedLMSProcess(run)
	probePath := filepath.Join(t.TempDir(), "metadata-timeout")
	process.Path = os.Args[0]
	process.Args = []string{"-test.run=^TestRestartAcceptanceConfigProbe$"}
	process.Environment["LMS_RESTART_CONFIG_PROBE"] = probePath
	command, err := startInstalledProcess(process)
	if err != nil {
		t.Fatal(err)
	}
	if err := command.Wait(); err != nil {
		t.Fatalf("wait for config probe: %v", err)
	}
	body, err := os.ReadFile(probePath)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(body)); got != "60000" {
		t.Fatalf("effective metadata call timeout = %q, want 60000", got)
	}
}

func TestInstalledLMSForcesProductionCollectionLoadTimeoutAcrossProcessBoundary(t *testing.T) {
	configHome := t.TempDir()
	configRoot := filepath.Join(configHome, "lm-semantic-search")
	if err := os.MkdirAll(configRoot, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(configRoot, "config.json"),
		[]byte(`{"milvusCollectionLoadTimeoutMs":2}`),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	t.Setenv("XDG_CONFIG_HOME", configHome)
	t.Setenv("CLAUDE_CONTEXT_MILVUS_COLLECTION_LOAD_TIMEOUT_MS", "1")

	run := acceptanceRun{Paths: pathsForRun(t.TempDir())}
	process := installedLMSProcess(run)
	probePath := filepath.Join(t.TempDir(), "collection-load-timeout")
	process.Path = os.Args[0]
	process.Args = []string{"-test.run=^TestRestartAcceptanceCollectionLoadConfigProbe$"}
	process.Environment["LMS_RESTART_CONFIG_PROBE"] = probePath
	command, err := startInstalledProcess(process)
	if err != nil {
		t.Fatal(err)
	}
	if err := command.Wait(); err != nil {
		t.Fatalf("wait for config probe: %v", err)
	}
	body, err := os.ReadFile(probePath)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(body)); got != "90000" {
		t.Fatalf("effective collection load timeout = %q, want 90000", got)
	}
}

func TestRestartAcceptanceConfigProbe(t *testing.T) {
	probePath := os.Getenv("LMS_RESTART_CONFIG_PROBE")
	if probePath == "" {
		return
	}
	cfg, err := config.Default()
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(strconv.Itoa(cfg.MilvusMetadataCallTimeoutMS) + "\n")
	if err := os.WriteFile(probePath, body, 0o600); err != nil {
		t.Fatal(fmt.Errorf("write config probe: %w", err))
	}
}

func TestRestartAcceptanceCollectionLoadConfigProbe(t *testing.T) {
	probePath := os.Getenv("LMS_RESTART_CONFIG_PROBE")
	if probePath == "" {
		return
	}
	cfg, err := config.Default()
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(strconv.Itoa(cfg.MilvusCollectionLoadTimeoutMS) + "\n")
	if err := os.WriteFile(probePath, body, 0o600); err != nil {
		t.Fatal(fmt.Errorf("write config probe: %w", err))
	}
}

func TestPreserveCaseDiagnosticsCopiesTheDaemonLogBeforeCleanup(t *testing.T) {
	paths := pathsForRun(t.TempDir())
	logPath := installedLMSLogPath(paths)
	if err := os.MkdirAll(filepath.Dir(logPath), 0o700); err != nil {
		t.Fatal(err)
	}
	const body = `{"msg":"hybrid search failed","err":"raw Milvus failure"}`
	if err := os.WriteFile(logPath, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}

	if err := preserveCaseDiagnostics(paths, "b"); err != nil {
		t.Fatalf("preserve diagnostics: %v", err)
	}
	preserved, err := os.ReadFile(filepath.Join(paths.Artifacts, "scenario-b-lms.log"))
	if err != nil {
		t.Fatalf("read preserved log: %v", err)
	}
	if string(preserved) != body {
		t.Fatalf("preserved log = %q, want %q", preserved, body)
	}
}

func TestPreserveCaseDiagnosticsRejectsAnUnsafeScenarioName(t *testing.T) {
	paths := pathsForRun(t.TempDir())
	if err := preserveCaseDiagnostics(paths, "../escape"); err == nil {
		t.Fatal("preserve diagnostics accepted an unsafe scenario name")
	}
}
