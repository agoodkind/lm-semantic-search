package daemon

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/adapterr"
)

func TestLooksLikeCodebaseID(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"cb_1780608524_a47875a9f695", true},
		{"cb_bogus", true},
		{"/Users/x/repo", false},
		{"cb_with/slash", false},
		{"relative/path", false},
		{"repo", false},
		{"", false},
	}
	for _, testCase := range cases {
		got := looksLikeCodebaseID(testCase.in)
		if got != testCase.want {
			t.Errorf("looksLikeCodebaseID(%q) = %v, want %v", testCase.in, got, testCase.want)
		}
	}
}

func TestResolveCanonicalPathAcceptsIDPathAndSymlink(t *testing.T) {
	manager, _, repoPath := newTestManager(t)
	canonical, err := filepath.EvalSymlinks(repoPath)
	if err != nil {
		t.Fatalf("EvalSymlinks returned error: %v", err)
	}
	codebase := newCodebaseRecord(canonical)
	manager.mu.Lock()
	manager.codebases[codebase.ID] = codebase
	manager.mu.Unlock()

	byID, err := manager.resolveCanonicalPath(codebase.ID)
	if err != nil {
		t.Fatalf("resolveCanonicalPath(id) returned error: %v", err)
	}
	if byID != canonical {
		t.Errorf("resolveCanonicalPath(id) = %q, want %q", byID, canonical)
	}

	byPath, err := manager.resolveCanonicalPath(repoPath)
	if err != nil {
		t.Fatalf("resolveCanonicalPath(path) returned error: %v", err)
	}
	if byPath != canonical {
		t.Errorf("resolveCanonicalPath(path) = %q, want %q", byPath, canonical)
	}

	linkPath := filepath.Join(t.TempDir(), "link")
	if err := os.Symlink(repoPath, linkPath); err != nil {
		t.Fatalf("Symlink returned error: %v", err)
	}
	bySymlink, err := manager.resolveCanonicalPath(linkPath)
	if err != nil {
		t.Fatalf("resolveCanonicalPath(symlink) returned error: %v", err)
	}
	if bySymlink != canonical {
		t.Errorf("resolveCanonicalPath(symlink) = %q, want %q", bySymlink, canonical)
	}
}

func TestResolveCanonicalPathUnknownIDErrorsClearly(t *testing.T) {
	manager, _, _ := newTestManager(t)

	_, err := manager.resolveCanonicalPath("cb_does_not_exist")
	if err == nil {
		t.Fatal("resolveCanonicalPath(unknown id) returned nil error")
	}

	var adapterErr *adapterr.AdapterError
	if !errors.As(err, &adapterErr) {
		t.Fatalf("error is not an AdapterError: %v", err)
	}
	if adapterErr.Class != adapterr.ClassUnknownCodebaseID {
		t.Errorf("class = %q, want %q", adapterErr.Class, adapterr.ClassUnknownCodebaseID)
	}
	if !strings.Contains(adapterr.SafeMessage(err), `no codebase with id "cb_does_not_exist"`) {
		t.Errorf("SafeMessage = %q, want it to name the id", adapterr.SafeMessage(err))
	}
}
