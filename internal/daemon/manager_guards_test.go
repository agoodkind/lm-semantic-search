package daemon

import "testing"

func TestGuardFilesystemRootRejectsRoot(t *testing.T) {
	t.Parallel()

	manager := &Manager{}
	if err := manager.guardFilesystemRoot("/"); err == nil {
		t.Fatal("guardFilesystemRoot accepted the filesystem root")
	}
	if err := manager.guardFilesystemRoot("/Users/example/repo"); err != nil {
		t.Fatalf("guardFilesystemRoot rejected a normal path: %v", err)
	}
}
