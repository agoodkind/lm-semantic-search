package store

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/model"
)

func TestReadJobEventsLatestReturnsLastEventPerJobID(t *testing.T) {
	journalPath := filepath.Join(t.TempDir(), "jobs.jsonl")
	occurredAt := []time.Time{
		time.Date(2026, time.July, 30, 8, 0, 0, 0, time.UTC),
		time.Date(2026, time.July, 30, 8, 1, 0, 0, time.UTC),
		time.Date(2026, time.July, 30, 8, 2, 0, 0, time.UTC),
	}
	events := []model.JobEvent{
		{Event: "job_queued", OccurredAt: occurredAt[0], Job: model.Job{ID: "job-1"}},
		{Event: "job_running", OccurredAt: occurredAt[1], Job: model.Job{ID: "job-1"}},
		{Event: "job_completed", OccurredAt: occurredAt[2], Job: model.Job{ID: "job-1"}},
	}
	for _, event := range events {
		if err := AppendJobEvent(journalPath, event); err != nil {
			t.Fatalf("AppendJobEvent returned error: %v", err)
		}
	}

	latest, err := ReadJobEventsLatest(journalPath)
	if err != nil {
		t.Fatalf("ReadJobEventsLatest returned error: %v", err)
	}
	got := latest["job-1"]
	if got.Event != "job_completed" {
		t.Fatalf("latest event = %q, want %q", got.Event, "job_completed")
	}
	if !got.OccurredAt.Equal(occurredAt[2]) {
		t.Fatalf("latest OccurredAt = %v, want %v", got.OccurredAt, occurredAt[2])
	}
}

func TestWriteJobEventsWritesOneLinePerEventAndRoundTrips(t *testing.T) {
	journalPath := filepath.Join(t.TempDir(), "jobs.jsonl")
	events := []model.JobEvent{
		{
			Event:      "job_running",
			OccurredAt: time.Date(2026, time.July, 30, 8, 0, 0, 0, time.UTC),
			Job:        model.Job{ID: "job-1", State: model.JobStateRunning},
		},
		{
			Event:      "job_completed",
			OccurredAt: time.Date(2026, time.July, 30, 8, 1, 0, 0, time.UTC),
			Job:        model.Job{ID: "job-2", State: model.JobStateCompleted},
		},
	}
	if err := WriteJobEvents(journalPath, events); err != nil {
		t.Fatalf("WriteJobEvents returned error: %v", err)
	}

	data, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
	if len(lines) != len(events) {
		t.Fatalf("journal line count = %d, want %d", len(lines), len(events))
	}
	for i, line := range lines {
		var got model.JobEvent
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			t.Fatalf("Unmarshal journal line %d returned error: %v", i, err)
		}
		if !reflect.DeepEqual(got, events[i]) {
			t.Fatalf("journal event %d = %#v, want %#v", i, got, events[i])
		}
	}

	latest, err := ReadJobEventsLatest(journalPath)
	if err != nil {
		t.Fatalf("ReadJobEventsLatest returned error: %v", err)
	}
	wantLatest := map[string]model.JobEvent{
		"job-1": events[0],
		"job-2": events[1],
	}
	if !reflect.DeepEqual(latest, wantLatest) {
		t.Fatalf("ReadJobEventsLatest = %#v, want %#v", latest, wantLatest)
	}
}

func TestWriteJobEventsPreservesOriginalFileWhenWriteFails(t *testing.T) {
	journalDir := t.TempDir()
	journalPath := filepath.Join(journalDir, "jobs.jsonl")
	original := []byte("original journal\n")
	if err := os.WriteFile(journalPath, original, 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	before, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatalf("ReadFile before failure returned error: %v", err)
	}

	if err := os.Chmod(journalDir, 0o555); err != nil {
		t.Fatalf("Chmod read-only returned error: %v", err)
	}
	t.Cleanup(func() {
		if err := os.Chmod(journalDir, 0o755); err != nil {
			t.Errorf("restore directory permissions: %v", err)
		}
	})
	writeErr := WriteJobEvents(
		journalPath,
		[]model.JobEvent{{Event: "job_completed", Job: model.Job{ID: "job-1"}}},
	)
	if err := os.Chmod(journalDir, 0o755); err != nil {
		t.Fatalf("restore directory permissions: %v", err)
	}
	if writeErr == nil {
		t.Fatal("WriteJobEvents returned nil, want an error")
	}

	after, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatalf("ReadFile after failure returned error: %v", err)
	}
	if !bytes.Equal(after, before) {
		t.Fatalf("journal after failed write = %q, want %q", after, before)
	}
}

func TestWriteJobEventsCleansTempFileWhenEncodingFails(t *testing.T) {
	journalDir := t.TempDir()
	journalPath := filepath.Join(journalDir, "jobs.jsonl")
	original := []byte("original journal\n")
	if err := os.WriteFile(journalPath, original, 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	invalidTime := time.Date(10000, time.January, 1, 0, 0, 0, 0, time.UTC)
	writeErr := WriteJobEvents(journalPath, []model.JobEvent{{
		Event:      "job_running",
		OccurredAt: invalidTime,
		Job: model.Job{
			ID:        "job-1",
			State:     model.JobStateRunning,
			UpdatedAt: invalidTime,
		},
	}})
	if writeErr == nil {
		t.Fatal("WriteJobEvents returned nil, want an encoding error")
	}

	after, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatalf("ReadFile after failure returned error: %v", err)
	}
	if !bytes.Equal(after, original) {
		t.Fatalf("journal after failed write = %q, want %q", after, original)
	}
	entries, err := os.ReadDir(journalDir)
	if err != nil {
		t.Fatalf("ReadDir returned error: %v", err)
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".tmp-") {
			t.Fatalf("temporary journal %q was not removed", entry.Name())
		}
	}
}

func TestReadJobEventsProjectsLatestJobs(t *testing.T) {
	journalPath := filepath.Join(t.TempDir(), "jobs.jsonl")
	events := []model.JobEvent{
		{Event: "job_queued", Job: model.Job{ID: "job-1", State: model.JobStateQueued}},
		{Event: "job_running", Job: model.Job{ID: "job-1", State: model.JobStateRunning}},
		{Event: "job_completed", Job: model.Job{ID: "job-2", State: model.JobStateCompleted}},
	}
	for _, event := range events {
		if err := AppendJobEvent(journalPath, event); err != nil {
			t.Fatalf("AppendJobEvent returned error: %v", err)
		}
	}

	jobs, err := ReadJobEvents(journalPath)
	if err != nil {
		t.Fatalf("ReadJobEvents returned error: %v", err)
	}
	want := map[string]model.Job{
		"job-1": events[1].Job,
		"job-2": events[2].Job,
	}
	if !reflect.DeepEqual(jobs, want) {
		t.Fatalf("ReadJobEvents = %#v, want %#v", jobs, want)
	}
}
