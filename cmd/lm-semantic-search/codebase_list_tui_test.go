package main

import (
	"strings"
	"testing"

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
	for _, want := range []string{"cb_1_aaaa", "cb_2_bbbb", "/tmp/alpha", "/tmp/beta", "indexed", "stale", "12"} {
		if !strings.Contains(view, want) {
			t.Errorf("View() missing %q\n%s", want, view)
		}
	}
}

func TestListModelNavigationMovesSelection(t *testing.T) {
	codebases := []*pb.Codebase{
		{Id: "cb_1_aaaa", CanonicalPath: "/tmp/alpha", Status: "indexed"},
		{Id: "cb_2_bbbb", CanonicalPath: "/tmp/beta", Status: "stale"},
	}
	model := newListModel(cliOptions{}, codebases)
	if model.table.Cursor() != 0 {
		t.Fatalf("initial cursor = %d, want 0", model.table.Cursor())
	}

	downModel, _ := model.Update(tea.KeyMsg{Type: tea.KeyDown})
	moved, ok := downModel.(listModel)
	if !ok {
		t.Fatalf("Update did not return a listModel")
	}
	if moved.table.Cursor() != 1 {
		t.Errorf("cursor after down = %d, want 1", moved.table.Cursor())
	}

	upModel, _ := moved.Update(tea.KeyMsg{Type: tea.KeyUp})
	back, ok := upModel.(listModel)
	if !ok {
		t.Fatalf("Update did not return a listModel")
	}
	if back.table.Cursor() != 0 {
		t.Errorf("cursor after up = %d, want 0", back.table.Cursor())
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
