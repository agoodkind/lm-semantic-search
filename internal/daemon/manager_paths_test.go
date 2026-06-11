package daemon

import (
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

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
