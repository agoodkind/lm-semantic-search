package daemon

import (
	"context"

	"goodkind.io/lm-semantic-search/internal/semantic"
)

type writeTotalsJobKey struct{}

func withWriteTotalsJob(ctx context.Context, jobID string) context.Context {
	return context.WithValue(ctx, writeTotalsJobKey{}, jobID)
}

func writeTotalsJobID(ctx context.Context) string {
	jobID, _ := ctx.Value(writeTotalsJobKey{}).(string)
	return jobID
}

// trackJobWrites counts each increase in one call's cumulative progress once.
func (manager *Manager) trackJobWrites(jobID string, next func(semantic.Progress)) func(semantic.Progress) {
	var reportedRows, reportedBatches int32
	return func(progress semantic.Progress) {
		rows := max(progress.CollectionRowsWritten-reportedRows, 0)
		batches := max(progress.EmbeddingBatchesCompleted-reportedBatches, 0)
		reportedRows = max(reportedRows, progress.CollectionRowsWritten)
		reportedBatches = max(reportedBatches, progress.EmbeddingBatchesCompleted)
		manager.addJobWriteTotals(jobID, int64(rows), int64(batches))
		if next != nil {
			next(progress)
		}
	}
}

func (manager *Manager) addJobWriteTotals(jobID string, rows, batches int64) {
	if jobID == "" || rows == 0 && batches == 0 {
		return
	}
	manager.transitionMutex.Lock()
	defer manager.transitionMutex.Unlock()
	manager.mu.Lock()
	defer manager.mu.Unlock()
	job, found := manager.jobs[jobID]
	if !found || isTerminalJobState(job.State) || job.Progress.WriteTotals == nil {
		return
	}
	totals := *job.Progress.WriteTotals
	totals.CollectionRowsWritten += rows
	totals.EmbeddingBatchesCompleted += batches
	job.Progress.WriteTotals = &totals
	manager.jobs[jobID] = job
	manager.journalJobProgressLocked(job)
}
