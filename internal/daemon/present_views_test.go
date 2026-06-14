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
