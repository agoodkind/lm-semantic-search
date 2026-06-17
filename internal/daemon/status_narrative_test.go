package daemon

import (
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/view"
)

func narrativeText(narrative view.StatusNarrative) string {
	return strings.Join(narrative.Lines, "\n")
}

func TestResolveStatusNarrativeMissing(t *testing.T) {
	t.Parallel()
	out := narrativeText(resolveStatusNarrative(displayMissing, "/repo", view.FailureSurface{}, view.QuarantineSurface{}, view.StatusView{}))
	for _, want := range []string{"source directory is missing", "Re-create the directory to resume indexing"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing narrative lacks %q in:\n%s", want, out)
		}
	}
}

func TestResolveStatusNarrativeFailedIncludesCorrelationIds(t *testing.T) {
	t.Parallel()
	failure := view.FailureSurface{HasFailure: true, Message: "boom", JobID: "job-xyz", TraceID: "trace-abc"}
	out := narrativeText(resolveStatusNarrative(displayFailed, "/repo", failure, view.QuarantineSurface{}, view.StatusView{}))
	for _, want := range []string{"could not be indexed", "🚧 boom", "Failed job job-xyz", "trace_id=trace-abc"} {
		if !strings.Contains(out, want) {
			t.Fatalf("failed narrative lacks %q in:\n%s", want, out)
		}
	}
}

func TestResolveStatusNarrativeFailedFallsBackWithoutDetail(t *testing.T) {
	t.Parallel()
	out := narrativeText(resolveStatusNarrative(displayFailed, "/repo", view.FailureSurface{}, view.QuarantineSurface{}, view.StatusView{}))
	if !strings.Contains(out, "could not be indexed. Re-run index_codebase to retry.") {
		t.Fatalf("failed narrative without detail lacks retry prompt in:\n%s", out)
	}
}

func TestResolveStatusNarrativeStaleIncludesRepairDetail(t *testing.T) {
	t.Parallel()
	failure := view.FailureSurface{
		HasFailure:    true,
		Message:       "Milvus collection missing; automatic rebuild could not start: boom",
		FailedAtLabel: "4:52 PM PDT",
		JobID:         "job-xyz",
		TraceID:       "trace-abc",
	}
	out := narrativeText(resolveStatusNarrative(displayStale, "/repo", failure, view.QuarantineSurface{}, view.StatusView{}))
	for _, want := range []string{"is stale since 4:52 PM PDT", "Repair detail: Milvus collection missing", "automatic rebuild could not start", "trace_id=trace-abc"} {
		if !strings.Contains(out, want) {
			t.Fatalf("stale narrative lacks %q in:\n%s", want, out)
		}
	}
}

func TestResolveStatusNarrativeQuarantinedFormatsCounts(t *testing.T) {
	t.Parallel()
	quarantine := view.QuarantineSurface{
		HasQuarantine:      true,
		Reason:             quarantineReasonWatcherLargeDelete,
		FirstObservedLabel: "4:52 PM PDT",
		LastObservedLabel:  "4:53 PM PDT",
		ObservationCount:   1,
		MissingCount:       400,
		TotalCount:         4292,
		Trigger:            quarantineTriggerWatcher,
	}
	statusView := view.StatusView{HasStats: true, Files: 58, Chunks: 600}
	out := narrativeText(resolveStatusNarrative(displayQuarantined, "/repo", view.FailureSurface{}, quarantine, statusView))
	for _, want := range []string{
		"is quarantined after a suspicious large disappearance",
		"Search continues to serve the last known-good index",
		"Last known good index: 58 files, 600 chunks",
		"400 of 4,292 tracked files",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("quarantined narrative lacks %q in:\n%s", want, out)
		}
	}
}

func TestResolveStatusNarrativeTemplateStateIsEmpty(t *testing.T) {
	t.Parallel()
	narrative := resolveStatusNarrative(displayIndexed, "/repo", view.FailureSurface{}, view.QuarantineSurface{}, view.StatusView{})
	if len(narrative.Lines) != 0 {
		t.Fatalf("template-state narrative should be empty, got: %#v", narrative.Lines)
	}
}
