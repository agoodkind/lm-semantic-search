//go:build restartacceptance

package sandboxharness

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Event records one timestamped acceptance transition.
type Event struct {
	Timestamp time.Time         `json:"timestamp"`
	Phase     string            `json:"phase"`
	Status    string            `json:"status"`
	Details   map[string]string `json:"details,omitempty"`
}

// Result records the final outcome of one isolated run.
type Result struct {
	RunID  string `json:"run_id"`
	Status string `json:"status"`
	Error  string `json:"error,omitempty"`
}

// EvidenceRecorder writes structured events and final result artifacts.
type EvidenceRecorder struct {
	root string
	now  func() time.Time
}

// NewEvidenceRecorder creates a recorder in an existing guarded artifact root.
func NewEvidenceRecorder(root string, now func() time.Time) (*EvidenceRecorder, error) {
	if now == nil {
		return nil, fmt.Errorf("evidence clock is unavailable")
	}
	info, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("inspect evidence root: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("evidence root %q is not a directory", root)
	}
	return &EvidenceRecorder{root: root, now: now}, nil
}

// Record appends one JSON event.
func (recorder *EvidenceRecorder) Record(phase string, status string, details map[string]string) error {
	event := Event{Timestamp: recorder.now().UTC(), Phase: phase, Status: status, Details: details}
	body, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("encode evidence event: %w", err)
	}
	path := filepath.Join(recorder.root, "events.jsonl")
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return fmt.Errorf("open evidence log: %w", err)
	}
	if _, err := file.Write(append(body, '\n')); err != nil {
		_ = file.Close()
		return fmt.Errorf("write evidence log: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close evidence log: %w", err)
	}
	return nil
}

// Finish writes the machine-readable and Markdown final result.
func (recorder *EvidenceRecorder) Finish(result Result) error {
	body, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("encode acceptance result: %w", err)
	}
	if err := os.WriteFile(filepath.Join(recorder.root, "result.json"), append(body, '\n'), 0o600); err != nil {
		return fmt.Errorf("write JSON acceptance result: %w", err)
	}
	markdown := fmt.Sprintf("# Restart acceptance\n\nRun: %s\n\nStatus: %s\n", result.RunID, result.Status)
	if result.Error != "" {
		markdown += "\nError: " + result.Error + "\n"
	}
	if err := os.WriteFile(filepath.Join(recorder.root, "result.md"), []byte(markdown), 0o600); err != nil {
		return fmt.Errorf("write Markdown acceptance result: %w", err)
	}
	return nil
}
