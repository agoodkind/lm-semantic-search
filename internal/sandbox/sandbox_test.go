package sandbox

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/config"
)

// isolateAmbient clears variables a developer machine may already carry, so a
// test observes the sandbox defaults rather than the operator's environment.
// HOME moves too, because the context root falls back to it.
func isolateAmbient(t *testing.T) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	for _, variable := range Env(t.TempDir()) {
		t.Setenv(variable.Name, "")
		if err := os.Unsetenv(variable.Name); err != nil {
			t.Fatalf("unset %s: %v", variable.Name, err)
		}
	}
}

// Nothing a sandbox daemon reads or writes lands outside the directory it was
// given. The model cache is the one exception, asserted separately below.
func TestResolveKeepsEveryDaemonPathInsideTheRoot(t *testing.T) {
	isolateAmbient(t)
	root := t.TempDir()

	resolved, err := Resolve(root)
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}

	paths := map[string]string{
		"StateRoot":    resolved.StateRoot,
		"ConfigRoot":   resolved.ConfigRoot,
		"ConfigPath":   resolved.ConfigPath,
		"ContextRoot":  resolved.ContextRoot,
		"SocketPath":   resolved.SocketPath,
		"LogPath":      resolved.LogPath,
		"LogsDir":      resolved.LogsDir,
		"RegistryPath": resolved.RegistryPath,
		"JobsPath":     resolved.JobsPath,
		"MerkleDir":    resolved.MerkleDir,
		"ChunksDir":    resolved.ChunksDir,
		"GraphDir":     resolved.GraphDir,
	}
	for name, path := range paths {
		if !strings.HasPrefix(filepath.Clean(path), filepath.Clean(root)) {
			t.Errorf("%s = %q, want a path under the sandbox root %q", name, path, root)
		}
	}
}

// The model cache must not move with the root. A checksum-verified model is
// identical for every daemon, so rooting the cache in a throwaway directory
// would re-download it every run and need network on an otherwise local path.
func TestResolveKeepsTheModelCacheOutsideTheRoot(t *testing.T) {
	isolateAmbient(t)
	root := t.TempDir()

	resolved, err := Resolve(root)
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}

	if resolved.ModelCacheRoot == "" {
		t.Fatal("ModelCacheRoot is empty; a sandbox would have nowhere to read a cached model from")
	}
	if strings.HasPrefix(filepath.Clean(resolved.ModelCacheRoot), filepath.Clean(root)) {
		t.Errorf(
			"ModelCacheRoot = %q, want a path outside the sandbox root %q so the model survives the run",
			resolved.ModelCacheRoot,
			root,
		)
	}
}

// A sandbox reaches neither shared backend, which is what let a throwaway
// daemon compete with the operator's for the GPU.
func TestResolveDefaultsToTheLocalBackends(t *testing.T) {
	isolateAmbient(t)

	resolved, err := Resolve(t.TempDir())
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}

	if resolved.Profile != config.ProfileOffline {
		t.Errorf("Profile = %q, want %q", resolved.Profile, config.ProfileOffline)
	}
	if resolved.IndexBackend != config.IndexBackendLocal {
		t.Errorf("IndexBackend = %q, want %q", resolved.IndexBackend, config.IndexBackendLocal)
	}
	if resolved.EmbeddingProvider != config.EmbeddingProviderONNX {
		t.Errorf(
			"EmbeddingProvider = %q, want %q",
			resolved.EmbeddingProvider,
			config.EmbeddingProviderONNX,
		)
	}
	if resolved.MilvusAddress != "" {
		t.Errorf("MilvusAddress = %q, want empty so no sandbox dials the shared store", resolved.MilvusAddress)
	}
}

// Each default is an override point rather than a forced setting. The live
// suite depends on this: it keeps the isolation while asking for the real store.
func TestResolveLeavesACallerSuppliedValueAlone(t *testing.T) {
	isolateAmbient(t)
	root := t.TempDir()
	chosenContextRoot := t.TempDir()

	t.Setenv("CLAUDE_CONTEXT_PROFILE", config.ProfileStandard)
	t.Setenv("CLAUDE_CONTEXTD_CONTEXT_ROOT", chosenContextRoot)

	resolved, err := Resolve(root)
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}

	if resolved.Profile != config.ProfileStandard {
		t.Errorf("Profile = %q, want the caller's %q", resolved.Profile, config.ProfileStandard)
	}
	if resolved.ContextRoot != chosenContextRoot {
		t.Errorf("ContextRoot = %q, want the caller's %q", resolved.ContextRoot, chosenContextRoot)
	}
	// The values the caller did not name are still isolated.
	if !strings.HasPrefix(filepath.Clean(resolved.StateRoot), filepath.Clean(root)) {
		t.Errorf("StateRoot = %q, want a path under %q", resolved.StateRoot, root)
	}
}

// A sandbox must not compete for the loopback port the installed daemon binds,
// which made the first working sandbox fail to start while it was running.
func TestResolveKeepsTheDebugListenerOffTheFixedPort(t *testing.T) {
	isolateAmbient(t)

	resolved, err := Resolve(t.TempDir())
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}

	if !strings.HasSuffix(resolved.DebugListenAddr, ":0") {
		t.Errorf(
			"DebugListenAddr = %q, want a port of 0 so the kernel assigns a free one",
			resolved.DebugListenAddr,
		)
	}
}
