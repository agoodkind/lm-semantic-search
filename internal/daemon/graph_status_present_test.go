package daemon

import (
	"strings"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/view"
)

func useRelativeTimeNowForTest(t *testing.T, now time.Time) {
	t.Helper()

	previousNow := relativeTimeNow
	relativeTimeNow = func() time.Time {
		return now
	}
	t.Cleanup(func() {
		relativeTimeNow = previousNow
	})
}

func formatLocalStatusDateForTest(value time.Time) string {
	const layout = "on Jan 2, 2006"
	location, err := time.LoadLocation("Local")
	if err != nil {
		return value.Format(layout)
	}
	return value.In(location).Format(layout)
}

func TestFormatRelativeTimeBuckets(t *testing.T) {
	now := time.Date(2026, 7, 3, 12, 0, 0, 0, time.UTC)
	useRelativeTimeNowForTest(t, now)

	cases := []struct {
		name  string
		value time.Time
		want  string
	}{
		{name: "zero value", value: time.Time{}, want: ""},
		{name: "0 seconds", value: now, want: "just now"},
		{name: "30 seconds", value: now.Add(-30 * time.Second), want: "just now"},
		{name: "59 seconds", value: now.Add(-59 * time.Second), want: "just now"},
		{name: "60 seconds", value: now.Add(-60 * time.Second), want: "1 minute ago"},
		{name: "120 seconds", value: now.Add(-120 * time.Second), want: "2 minutes ago"},
		{name: "59 minutes", value: now.Add(-59 * time.Minute), want: "59 minutes ago"},
		{name: "60 minutes", value: now.Add(-60 * time.Minute), want: "1 hour ago"},
		{name: "23 hours", value: now.Add(-23 * time.Hour), want: "23 hours ago"},
		{name: "24 hours", value: now.Add(-24 * time.Hour), want: "yesterday"},
		{name: "47 hours", value: now.Add(-47 * time.Hour), want: "yesterday"},
		{name: "49 hours", value: now.Add(-49 * time.Hour), want: formatLocalStatusDateForTest(now.Add(-49 * time.Hour))},
		{name: "10 days", value: now.Add(-10 * 24 * time.Hour), want: formatLocalStatusDateForTest(now.Add(-10 * 24 * time.Hour))},
		{name: "future", value: now.Add(time.Minute), want: "just now"},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			if got := formatRelativeTime(testCase.value); got != testCase.want {
				t.Fatalf("formatRelativeTime(%s) = %q, want %q", testCase.name, got, testCase.want)
			}
		})
	}
}

func TestResolveStatusNarrativeAppendsGraphLineFromViewFields(t *testing.T) {
	cases := []struct {
		name       string
		statusView view.StatusView
		want       string
	}{
		{
			name:       "updated",
			statusView: view.StatusView{GraphUpdatedAt: "6 minutes ago"},
			want:       "🕸️ Code graph updated 6 minutes ago",
		},
		{
			name:       "ready no time",
			statusView: view.StatusView{GraphReadyNoTime: true},
			want:       "🕸️ Code graph: ready",
		},
		{
			name:       "not built",
			statusView: view.StatusView{GraphNotBuilt: true},
			want:       "🕸️ Code graph: builds shortly, or run index_codebase",
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			narrative := resolveStatusNarrative(
				displayMissing,
				"/repo",
				view.FailureSurface{},
				view.QuarantineSurface{},
				testCase.statusView,
			)
			if len(narrative.Lines) == 0 {
				t.Fatal("narrative returned no lines")
			}
			got := narrative.Lines[len(narrative.Lines)-1]
			if got != testCase.want {
				t.Fatalf("graph narrative line = %q, want %q", got, testCase.want)
			}
			if strings.Contains(strings.Join(narrative.Lines, "\n"), "semantic snapshot") {
				t.Fatalf("narrative still mentions semantic snapshot:\n%s", strings.Join(narrative.Lines, "\n"))
			}
		})
	}
}

func TestResolveSearchStatusViewPopulatesGraphFields(t *testing.T) {
	manager, _, _ := newTestManager(t)
	t.Cleanup(func() {
		manager.CloseGraphEngines()
	})

	now := time.Date(2026, 7, 3, 12, 0, 0, 0, time.UTC)
	useRelativeTimeNowForTest(t, now)

	activeJob := &model.Job{
		State: model.JobStateRunning,
	}

	codebase := model.Codebase{
		ID:             "cb-code",
		CanonicalPath:  "/repo/code",
		Kind:           model.CodebaseKindCode,
		Status:         model.CodebaseStatusIndexed,
		GraphState:     model.GraphStateStale,
		GraphUpdatedAt: now.Add(-6 * time.Minute),
	}
	statusView, _, _ := resolveSearchStatusView(codebase, activeJob, dependencyHealth{})
	if statusView.GraphUpdatedAt != "6 minutes ago" {
		t.Fatalf("GraphUpdatedAt = %q, want %q", statusView.GraphUpdatedAt, "6 minutes ago")
	}

	legacyCodebase := model.Codebase{
		ID:             "cb-legacy-code",
		CanonicalPath:  "/repo/legacy-code",
		Status:         model.CodebaseStatusIndexed,
		GraphState:     model.GraphStateStale,
		GraphUpdatedAt: now.Add(-6 * time.Minute),
	}
	statusView, _, _ = resolveSearchStatusView(legacyCodebase, activeJob, dependencyHealth{})
	if statusView.GraphUpdatedAt != "6 minutes ago" {
		t.Fatalf("legacy empty-kind GraphUpdatedAt = %q, want %q", statusView.GraphUpdatedAt, "6 minutes ago")
	}

	documentCodebase := model.Codebase{
		ID:             "cb-document",
		CanonicalPath:  "chat:///thread-alpha",
		Kind:           model.CodebaseKindDocument,
		Status:         model.CodebaseStatusIndexed,
		GraphState:     model.GraphStateStale,
		GraphUpdatedAt: now.Add(-6 * time.Minute),
	}
	statusView, _, _ = resolveSearchStatusView(documentCodebase, activeJob, dependencyHealth{})
	if statusView.GraphUpdatedAt != "" || statusView.GraphReadyNoTime || statusView.GraphNotBuilt {
		t.Fatalf("non-code graph fields = %+v, want all zero", statusView)
	}
}
