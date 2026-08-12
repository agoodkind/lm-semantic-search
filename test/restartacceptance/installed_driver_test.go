//go:build restartacceptance

package restartacceptance

import "testing"

func TestInstalledLMSUsesShorterMetadataBoundThanScenarioFailureBound(t *testing.T) {
	run := acceptanceRun{Paths: runPaths{}}
	process := installedLMSProcess(run)
	if got := process.Environment["CLAUDE_CONTEXT_MILVUS_METADATA_CALL_TIMEOUT_MS"]; got != "10000" {
		t.Fatalf("metadata call timeout = %q, want 10000", got)
	}
}
