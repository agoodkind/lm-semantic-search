//go:build installlive

package installlive

import (
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/test/operatorstate"
)

const operatorStateRoot = ".local/state/lm-semantic-search"

// runGuardedWrite sets HOME to a new directory with an operator state root,
// then runs operatorstate.RunWithGuard around a run that writes relativePath
// under that root. It returns the guard exit code.
func runGuardedWrite(t *testing.T, relativePath string) int {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	root := filepath.Join(home, operatorStateRoot)
	if err := os.MkdirAll(filepath.Join(root, "logs"), 0o755); err != nil {
		t.Fatalf("create operator state root: %v", err)
	}
	return operatorstate.RunWithGuard(func() int {
		if err := os.WriteFile(filepath.Join(root, relativePath), []byte("written during the run\n"), 0o600); err != nil {
			t.Fatalf("write %s: %v", relativePath, err)
		}
		return 0
	})
}

// TestOperatorStateGuardFailsOnUpdateLockWrite writes update.lock under the
// state root during a guarded run. The guard must fail the run.
func TestOperatorStateGuardFailsOnUpdateLockWrite(t *testing.T) {
	if exitCode := runGuardedWrite(t, "update.lock"); exitCode != 1 {
		t.Fatalf("guard exit code after an update.lock write = %d, want 1", exitCode)
	}
}

// TestOperatorStateGuardIgnoresDaemonLogWrite writes a daemon log file under
// the state root during a guarded run, as a running LMS daemon does. The
// guard must pass the run.
func TestOperatorStateGuardIgnoresDaemonLogWrite(t *testing.T) {
	if exitCode := runGuardedWrite(t, filepath.Join("logs", "daemon.log")); exitCode != 0 {
		t.Fatalf("guard exit code after a log write = %d, want 0", exitCode)
	}
}
