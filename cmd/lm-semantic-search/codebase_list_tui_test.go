package main

import (
	"strings"
	"testing"
	"unicode/utf8"

	tea "github.com/charmbracelet/bubbletea"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

func TestListModelViewShowsRecords(t *testing.T) {
	codebases := []*pb.Codebase{
		{Id: "cb_1_aaaa", CanonicalPath: "/tmp/alpha", Status: "indexed", LastSuccessfulRun: &pb.IndexRunSummary{IndexedFiles: 12}},
		{Id: "cb_2_bbbb", CanonicalPath: "/tmp/beta", Status: "stale"},
	}
	model := newListModel(cliOptions{}, codebases)

	view := model.View()
	for _, want := range []string{"cb_1_aaaa", "cb_2_bbbb", "alpha", "beta", "indexed", "stale", "12", "NAME", "STATUS", "PATH"} {
		if !strings.Contains(view, want) {
			t.Errorf("View() missing %q\n%s", want, view)
		}
	}
}

// TestListRowRendersDaemonTokens proves a row renders the daemon-provided glyph
// and label tokens rather than a client-side map, so the CLI shares the daemon's
// status vocabulary including "waiting".
func TestListRowRendersDaemonTokens(t *testing.T) {
	codebases := []*pb.Codebase{
		{Id: "cb_1_aaaa", CanonicalPath: "/tmp/alpha", Status: "indexing", DisplayStatus: "waiting", GlyphToken: "⋯", StatusLabel: "waiting"},
	}
	model := newListModel(cliOptions{}, codebases)

	view := model.View()
	for _, want := range []string{"⋯", "waiting"} {
		if !strings.Contains(view, want) {
			t.Errorf("View() missing daemon token %q\n%s", want, view)
		}
	}
}

// TestListRowFallsBackToRawStatusWithoutTokens proves a row from an older daemon
// that omits the glyph and label tokens still renders the raw status word.
func TestListRowFallsBackToRawStatusWithoutTokens(t *testing.T) {
	codebases := []*pb.Codebase{
		{Id: "cb_1_aaaa", CanonicalPath: "/tmp/alpha", Status: "indexed"},
	}
	model := newListModel(cliOptions{}, codebases)

	view := model.View()
	if !strings.Contains(view, "indexed") {
		t.Errorf("View() missing fallback raw status:\n%s", view)
	}
}

func TestFitHeadKeepsTail(t *testing.T) {
	got := fitHead("/Users/agoodkind/Sites/lmd", 8)
	if !strings.HasSuffix(got, "lmd") {
		t.Errorf("fitHead kept the wrong end: %q", got)
	}
	if !strings.HasPrefix(got, "…") {
		t.Errorf("fitHead missing leading ellipsis: %q", got)
	}
	if utf8.RuneCountInString(got) != 8 {
		t.Errorf("fitHead width = %d runes, want 8: %q", utf8.RuneCountInString(got), got)
	}
}

func TestListModelNavigationMovesSelection(t *testing.T) {
	codebases := []*pb.Codebase{
		{Id: "cb_1_aaaa", CanonicalPath: "/tmp/alpha", Status: "indexed"},
		{Id: "cb_2_bbbb", CanonicalPath: "/tmp/beta", Status: "stale"},
	}
	model := newListModel(cliOptions{}, codebases)
	if model.cursor != 0 {
		t.Fatalf("initial cursor = %d, want 0", model.cursor)
	}

	downModel, _ := model.Update(tea.KeyMsg{Type: tea.KeyDown})
	moved, ok := downModel.(listModel)
	if !ok {
		t.Fatalf("Update did not return a listModel")
	}
	if moved.cursor != 1 {
		t.Errorf("cursor after down = %d, want 1", moved.cursor)
	}

	upModel, _ := moved.Update(tea.KeyMsg{Type: tea.KeyUp})
	back, ok := upModel.(listModel)
	if !ok {
		t.Fatalf("Update did not return a listModel")
	}
	if back.cursor != 0 {
		t.Errorf("cursor after up = %d, want 0", back.cursor)
	}
}

func TestListModelRefreshPreservesSelectionByID(t *testing.T) {
	codebases := []*pb.Codebase{
		{Id: "cb_1_aaaa", CanonicalPath: "/tmp/alpha", Status: "indexed"},
		{Id: "cb_2_bbbb", CanonicalPath: "/tmp/beta", Status: "stale"},
	}
	model := newListModel(cliOptions{}, codebases)
	model.cursor = 1

	// A refresh reorders the records; the cursor should follow cb_2_bbbb.
	reordered := []*pb.Codebase{
		{Id: "cb_3_cccc", CanonicalPath: "/tmp/gamma", Status: "indexed"},
		{Id: "cb_2_bbbb", CanonicalPath: "/tmp/beta", Status: "indexed"},
		{Id: "cb_1_aaaa", CanonicalPath: "/tmp/alpha", Status: "indexed"},
	}
	refreshed := model.applyRefresh(refreshedMsg{codebases: reordered})
	if refreshed.codebases[refreshed.cursor].GetId() != "cb_2_bbbb" {
		t.Errorf("cursor moved off cb_2_bbbb after refresh: index=%d id=%s", refreshed.cursor, refreshed.codebases[refreshed.cursor].GetId())
	}
}

func TestListModelQuitKeySignalsQuit(t *testing.T) {
	codebases := []*pb.Codebase{
		{Id: "cb_1_aaaa", CanonicalPath: "/tmp/alpha", Status: "indexed"},
	}
	model := newListModel(cliOptions{}, codebases)
	updated, cmd := model.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'q'}})
	quit, ok := updated.(listModel)
	if !ok {
		t.Fatalf("Update did not return a listModel")
	}
	if !quit.quitting {
		t.Error("q did not set quitting")
	}
	if cmd == nil {
		t.Error("q did not return a quit command")
	}
}
