package daemon

import (
	"strings"
	"time"

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

// resolveJobSurface reduces a raw job and the pipeline-degraded flag into the
// status package's resolved job view. It is the one seam between a model.Job and
// the SOT, the job-side mirror of computeDisplayStatus, so it lives here at the
// boundary rather than in the render layer the guard test keeps free of raw job
// reads. Every job surface formats from the JobSurface it returns instead of
// re-deriving a state label or error echo. A job stopping on a shared
// dependency is exactly a retryable error during a degraded pipeline, which
// ResolveJob folds by suppressing the per-job echo the banner already carries.
func resolveJobSurface(job model.Job, pipelineDegraded bool) status.JobSurface {
	dependency := status.Healthy
	if pipelineDegraded {
		dependency = status.EmbedderBusy
	}
	retryable := false
	errorMessage := ""
	if job.Error != nil {
		retryable = job.Error.Retryable
		errorMessage = strings.TrimSpace(job.Error.Message)
	}
	return status.ResolveJob(status.JobInputs{
		State:        job.State,
		Retryable:    retryable,
		ErrorMessage: errorMessage,
		Dependency:   dependency,
	})
}

// codebaseFailureView is the resolved failure detail a render bucket formats. It
// is built once at the boundary from the raw failure record so the render layer
// never reaches into codebase.LastFailedRun; a renderer that cannot see the raw
// failure record cannot print failure text that contradicts the bucket the SOT
// chose. HasFailure is false when the codebase carries no recorded failure.
type codebaseFailureView struct {
	HasFailure bool
	Message    string
	FailedAt   time.Time
	JobID      string
	TraceID    string
}

// resolveCodebaseFailure reduces a codebase's raw failure record into the
// render-facing failure view, the codebase-side mirror of resolveJobSurface. It
// is the only reader of codebase.LastFailedRun outside the lifecycle logic, kept
// here at the boundary rather than in the render layer the guard test holds free
// of raw failure reads.
func resolveCodebaseFailure(codebase model.Codebase) codebaseFailureView {
	if codebase.LastFailedRun == nil {
		return codebaseFailureView{HasFailure: false, Message: "", FailedAt: time.Time{}, JobID: "", TraceID: ""}
	}
	return codebaseFailureView{
		HasFailure: true,
		Message:    codebase.LastFailedRun.Message,
		FailedAt:   codebase.LastFailedRun.FailedAt,
		JobID:      codebase.LastFailedRun.JobID,
		TraceID:    codebase.LastFailedRun.TraceID,
	}
}
