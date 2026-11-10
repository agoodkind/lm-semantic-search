package daemon

import (
	"path/filepath"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
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

// A URI-shaped argument must be rejected, not resolved as a filesystem path.
// filepath.Abs on "chat:///x" with cwd "/" produced the ghost path
// "/chat:/x" that broke the boot resume pass.
func TestCanonicalizePathRejectsURISchemes(t *testing.T) {
	t.Parallel()
	for _, arg := range []string{"chat:///clyde-conversations", "https://example.com/repo", "file:///tmp/x"} {
		_, err := canonicalizePath(arg)
		if err == nil {
			t.Fatalf("canonicalizePath(%q) succeeded, want a rejection", arg)
		}
		if !strings.Contains(err.Error(), "URI") {
			t.Fatalf("canonicalizePath(%q) error %q does not name the URI cause", arg, err)
		}
	}
}

func TestDropGhostURICodebases(t *testing.T) {
	t.Parallel()
	ghost := newCodebaseRecord("/chat:/clyde-conversations")
	real := newCodebaseRecord("/Users/x/repo")
	conversation := newCodebaseRecord("chat:///clyde-conversations")
	conversation.Kind = model.CodebaseKindDocument
	codebases := map[string]model.Codebase{ghost.ID: ghost, real.ID: real, conversation.ID: conversation}

	dropGhostURICodebases(codebases)

	if _, ok := codebases[ghost.ID]; ok {
		t.Fatal("ghost URI record survived the repair pass")
	}
	if _, ok := codebases[real.ID]; !ok {
		t.Fatal("real filesystem record was dropped")
	}
	if _, ok := codebases[conversation.ID]; !ok {
		t.Fatal("document codebase was dropped")
	}
}
