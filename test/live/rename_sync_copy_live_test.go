//go:build live

package live

import "testing"

// TestDeltaSyncCopiesRenamedChunks checks chunk reuse after tracked file renames
// with the watcher disabled and with its converge waiting for an index slot.
// It checks that the embedder does not receive inputs for renamed files and that
// stored rows and snapshot entries use the destination paths.
func TestDeltaSyncCopiesRenamedChunks(t *testing.T) {
	scenarios := []renameQueueScenario{
		{mode: renameQueueSyncOnly, gitRepo: true},
		{mode: renameQueueSyncOnly, gitRepo: false},
		{mode: renameQueueBusy, gitRepo: true},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name(), func(t *testing.T) {
			measurement := runRenameQueueScenario(t, scenario)
			requireRenameQueueReuse(t, measurement)
			if scenario.mode != renameQueueSyncOnly {
				return
			}
			for _, job := range measurement.JobsAfterRename {
				if job.Client == renameQueueSyncClient && job.State != "completed" {
					t.Errorf("Sync job %s ended as %q. The test expected every sync job to complete.", job.ID, job.State)
				}
			}
		})
	}
}
