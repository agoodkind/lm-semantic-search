package daemon

import (
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
	render "goodkind.io/lm-semantic-search/internal/render"
	"goodkind.io/lm-semantic-search/internal/view"
)

func TestResolveStatusViewFallsBackToLiveChunkTotal(t *testing.T) {
	t.Parallel()
	codebase := model.Codebase{CanonicalPath: "/repo", LiveChunkTotal: 33240}
	job := model.Job{
		State:     model.JobStateRunning,
		Operation: "sync",
		Progress:  model.Progress{FilesInCodebase: 100, FilesModified: 2},
	}
	statusView, templateName := resolveStatusView(codebase, &job, displayIndexing, "")
	if statusView.Breakdown.ChunksTotal != 33240 {
		t.Fatalf("ChunksTotal = %d, want the live total 33240", statusView.Breakdown.ChunksTotal)
	}
	if templateName != "incremental.md.tmpl" {
		t.Fatalf("template = %q, want incremental", templateName)
	}
}

// TestResolveStatusViewDiscoveredSelectsTemplate proves a discovered codebase
// resolves to the discovered template, and that rendering a discovered status
// view through render.GetIndex carries the reuse-forecast line and the
// not-yet-indexed body.
func TestResolveStatusViewDiscoveredSelectsTemplate(t *testing.T) {
	t.Parallel()
	codebase := model.Codebase{Status: model.CodebaseStatusDiscovered, CanonicalPath: "/x"}
	statusView, templateName := resolveStatusView(codebase, nil, displayDiscovered, "")
	if templateName != "discovered.md.tmpl" {
		t.Fatalf("template = %q, want discovered.md.tmpl", templateName)
	}

	statusView.ReuseForecastLine = "♻️ reuses embeddings from 2 indexed sibling worktrees"
	out := render.GetIndex(view.GetIndexView{
		Tracked:       true,
		RequestedPath: "/x",
		CanonicalPath: "/x",
		Display:       view.Display(displayDiscovered),
		TemplateName:  templateName,
		Status:        statusView,
	})
	if !strings.Contains(out, "discovered, not yet indexed") {
		t.Fatalf("discovered render missing the not-yet-indexed body; got %q", out)
	}
	if !strings.Contains(out, statusView.ReuseForecastLine) {
		t.Fatalf("discovered render missing the reuse forecast line; got %q", out)
	}
}

func TestRenderMutationAckManifest(t *testing.T) {
	t.Parallel()
	out := render.MutationAck(view.MutationAckView{
		Kind:            view.AckManifest,
		Path:            "",
		JobID:           "",
		StateLabel:      "",
		AlreadyTerminal: false,
		Deduplicated:    false,
		CollectionID:    "clyde-conversations",
		CollectionName:  "",
		CodebaseID:      "",
		ConversationID:  "",
		DocumentCount:   0,
		NeededCount:     11,
		TotalCount:      1011,
	})
	if !strings.Contains(out, "needs 11 of 1011") {
		t.Fatalf("manifest ack = %q, want the needed-of-total counts", out)
	}
}
