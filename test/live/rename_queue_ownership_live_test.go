//go:build live

package live

import "testing"

// TestAdmittedConvergeOwnsCodebaseDuringSweep renames tracked files while the
// watcher converge has an available index slot and while it waits for the only
// index slot. It checks that the converge completes and copies stored chunks for
// every renamed file. It also checks that the periodic sweep skips the codebase
// while the converge waits for an index slot.
func TestAdmittedConvergeOwnsCodebaseDuringSweep(t *testing.T) {
	scenarios := []renameQueueScenario{
		{mode: renameQueueEmpty, gitRepo: true},
		{mode: renameQueueBusy, gitRepo: true},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name(), func(t *testing.T) {
			measurement := runRenameQueueScenario(t, scenario)
			requireRenameQueueReuse(t, measurement)
			for _, job := range measurement.JobsAfterRename {
				if job.Client == renameQueueWatcherClient && job.State != "completed" {
					t.Errorf("Watcher job %s ended as %q. The test expected every watcher job to complete.", job.ID, job.State)
				}
			}
			if measurement.ConvergeCopyChunksDelta != int64(measurement.RenamedFiles) {
				t.Errorf("The watcher converge copied chunks for %d paths. The test expected %d paths.", measurement.ConvergeCopyChunksDelta, measurement.RenamedFiles)
			}
			if scenario.mode == renameQueueBusy && measurement.SyncSkippedDelta == 0 {
				t.Error("The periodic sweep did not skip the codebase while its watcher converge waited for an index slot.")
			}
		})
	}
}
