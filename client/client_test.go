package client

import "testing"

func TestResolveSocketPathUsesDaemonSocketEnv(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("CLAUDE_CONTEXTD_SOCKET_PATH", "/tmp/lm-semantic-search-test.sock")

	socketPath := ResolveSocketPath()
	if socketPath != "/tmp/lm-semantic-search-test.sock" {
		t.Fatalf("ResolveSocketPath() = %q", socketPath)
	}
}
