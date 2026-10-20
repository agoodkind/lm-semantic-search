package statushistory_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/statushistory"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestBuildReportsUnknownTimeWhenRetainedScopesDiffer(t *testing.T) {
	for _, testCase := range []struct {
		name       string
		spanMillis int64
		jobMillis  int64
		unknown    bool
	}{
		{name: "matching scope", spanMillis: 100, jobMillis: 200, unknown: false},
		{name: "span scope exceeds retained jobs", spanMillis: 200, jobMillis: 100, unknown: true},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			root := t.TempDir()
			now := time.Date(2026, 10, 20, 18, 0, 0, 0, time.UTC)
			completed := now.Add(-time.Minute)
			logRecord := struct {
				Time       time.Time `json:"time"`
				Message    string    `json:"msg"`
				Span       string    `json:"span"`
				SpanID     string    `json:"span_id"`
				JobID      string    `json:"job_id"`
				DurationMS int64     `json:"duration_ms"`
			}{Time: completed, Message: "daemon.span.completed", Span: "daemon.runJob", SpanID: "one", JobID: "job-one", DurationMS: testCase.spanMillis}
			body, err := json.Marshal(logRecord)
			if err != nil {
				t.Fatal(err)
			}
			logs := filepath.Join(root, "logs")
			if err := os.Mkdir(logs, 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(logs, "daemon.jsonl"), append(body, '\n'), 0o600); err != nil {
				t.Fatal(err)
			}
			body, err = json.Marshal(model.JobEvent{Event: "job_completed", OccurredAt: completed, Job: model.Job{ID: "job-one", State: model.JobStateCompleted, StartedAt: completed.Add(-time.Duration(testCase.jobMillis) * time.Millisecond), CompletedAt: &completed}})
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, "jobs.jsonl"), append(body, '\n'), 0o600); err != nil {
				t.Fatal(err)
			}
			report, err := statushistory.Build(statushistory.Input{StateRoot: root, Since: time.Hour, Now: now, Status: &pb.GetStatusResponse{ReadAt: timestamppb.New(now)}})
			if err != nil {
				t.Fatal(err)
			}
			if testCase.unknown {
				if report.UnattributedDurationMS != nil || !slices.Contains(report.Warnings, "instrumented stage time exceeds measured job duration") {
					t.Fatalf("Mismatched scopes must report unknown time and a warning: %+v", report)
				}
				return
			}
			if report.UnattributedDurationMS == nil || *report.UnattributedDurationMS != 100 {
				t.Fatalf("Matching scopes must report 100 ms: %+v", report.UnattributedDurationMS)
			}
		})
	}
}
