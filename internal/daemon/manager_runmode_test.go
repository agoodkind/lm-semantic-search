package daemon

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

func TestSetJobRunMode(t *testing.T) {
	manager, _, _ := newTestManager(t)
	job := model.Job{ID: "job-rm", State: model.JobStateRunning}
	manager.mu.Lock()
	manager.jobs[job.ID] = job
	manager.mu.Unlock()

	manager.setJobRunMode(job.ID, model.RunModeResuming)

	got, _ := manager.GetJob(job.ID)
	if got.Progress.RunMode != model.RunModeResuming {
		t.Fatalf("RunMode = %q, want %q", got.Progress.RunMode, model.RunModeResuming)
	}
}
