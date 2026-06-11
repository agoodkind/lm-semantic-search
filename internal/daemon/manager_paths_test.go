package daemon

import (
	"path/filepath"
	"testing"
)

func TestResolveRequestPathJoinsRelativeAgainstCallerCwd(t *testing.T) {
	t.Parallel()

	got, err := resolveRequestPath("sub/dir", "/Users/example/repo")
	if err != nil {
		t.Fatalf("resolveRequestPath returned error: %v", err)
	}
	want := filepath.Join("/Users/example/repo", "sub/dir")
	if got != want {
		t.Fatalf("resolveRequestPath = %q, want %q", got, want)
	}
}

func TestResolveRequestPathResolvesDotToCallerCwd(t *testing.T) {
	t.Parallel()

	got, err := resolveRequestPath(".", "/Users/example/repo")
	if err != nil {
		t.Fatalf("resolveRequestPath returned error: %v", err)
	}
	if got != "/Users/example/repo" {
		t.Fatalf("resolveRequestPath = %q, want /Users/example/repo", got)
	}
}

func TestResolveRequestPathRejectsRelativeWithoutCallerCwd(t *testing.T) {
	t.Parallel()

	for _, callerCwd := range []string{"", "   ", "not/absolute"} {
		if _, err := resolveRequestPath(".", callerCwd); err == nil {
			t.Fatalf("resolveRequestPath(%q, %q) returned nil error", ".", callerCwd)
		}
	}
}

func TestResolveRequestPathPassesThroughAbsoluteIDAndURI(t *testing.T) {
	t.Parallel()

	cases := []string{"/abs/path", "cb_123_abc", "chat:///clyde-conversations"}
	for _, requested := range cases {
		got, err := resolveRequestPath(requested, "/Users/example/repo")
		if err != nil {
			t.Fatalf("resolveRequestPath(%q) returned error: %v", requested, err)
		}
		if got != requested {
			t.Fatalf("resolveRequestPath(%q) = %q, want pass-through", requested, got)
		}
	}
}
