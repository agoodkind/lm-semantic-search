//go:build restartacceptance

package sandboxharness

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestEvidenceRecorderWritesEventsAndFinalResult(t *testing.T) {
	root := t.TempDir()
	recorder, err := NewEvidenceRecorder(root, func() time.Time {
		return time.Date(2026, 8, 13, 1, 2, 3, 0, time.UTC)
	})
	if err != nil {
		t.Fatalf("new recorder: %v", err)
	}
	if err := recorder.Record("restore", "passed", map[string]string{"archive": "verified"}); err != nil {
		t.Fatalf("record evidence: %v", err)
	}
	result := Result{RunID: "run-accepted", Status: "passed"}
	if err := recorder.Finish(result); err != nil {
		t.Fatalf("finish evidence: %v", err)
	}
	eventBody, err := os.ReadFile(filepath.Join(root, "events.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	var event Event
	if err := json.Unmarshal(eventBody, &event); err != nil {
		t.Fatalf("decode event: %v", err)
	}
	if event.Phase != "restore" || event.Details["archive"] != "verified" {
		t.Fatalf("event = %#v", event)
	}
	resultBody, err := os.ReadFile(filepath.Join(root, "result.json"))
	if err != nil {
		t.Fatal(err)
	}
	var got Result
	if err := json.Unmarshal(resultBody, &got); err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if got != result {
		t.Fatalf("result = %#v, want %#v", got, result)
	}
}
