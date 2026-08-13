//go:build restartacceptance

package restartacceptance

import (
	"os"
	"path/filepath"
	"testing"
)

func TestInstalledLMSKeepsMetadataCallsAtTheProductionDefault(t *testing.T) {
	run := acceptanceRun{Paths: runPaths{}}
	process := installedLMSProcess(run)
	if got := process.Environment["CLAUDE_CONTEXT_MILVUS_METADATA_CALL_TIMEOUT_MS"]; got != "" {
		t.Fatalf("metadata call timeout = %q, want no acceptance override", got)
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
