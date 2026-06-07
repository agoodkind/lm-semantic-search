package daemon

import (
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/status"
)

// displayStatus is the user-facing status a codebase presents. It aliases the
// status package's Display so the daemon keeps its short names while the values
// and the resolution rules live in the single status source of truth.
type displayStatus = status.Display

const (
	displayPreparing = status.DisplayPreparing
	displayIndexing  = status.DisplayIndexing
	displayIndexed   = status.DisplayIndexed
	displayWaiting   = status.DisplayWaiting
	displayStale     = status.DisplayStale
	displayFailed    = status.DisplayFailed
	displayMissing   = status.DisplayMissing
)

// computeDisplayStatus resolves the display status through the status package,
// the single source of truth for every surface (list, detail, MCP, CLI). It
// reduces the live job and the daemon's dependency health into the normalized
// status.Inputs and lets status.ResolveDisplay fold them onto the persisted
// status. pipelineDegraded carries only whether a shared dependency is degraded;
// ResolveDisplay reads it through Degraded(), so the specific mode does not
// matter here and the banner names the cause separately.
func computeDisplayStatus(codebase model.Codebase, activeJob *model.Job, pipelineDegraded bool) displayStatus {
	dependency := status.Healthy
	if pipelineDegraded {
		dependency = status.EmbedderBusy
	}
	return status.Resolve(status.Inputs{
		Status:                  codebase.Status,
		HasActiveJob:            activeJob != nil,
		JobScopeKnown:           activeJob != nil && jobScopeKnown(activeJob.Progress),
		BackgroundSyncReconcile: activeJob != nil && isBackgroundSyncReconcile(&codebase, activeJob),
		Dependency:              dependency,
		Search:                  status.SearchNone,
	}).Display
}
