package daemon

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/view"
)

type jobPhase string

const (
	jobPhaseCancelling jobPhase = "cancelling"
	jobPhaseCancelled  jobPhase = "cancelled"
)

// formatCount renders an integer with thousands separators so large corpus
// numbers stay readable, for example "33,240".
func formatCount(value int32) string {
	digits := strconv.FormatInt(int64(value), 10)
	if len(digits) <= 3 {
		return digits
	}
	var out strings.Builder
	lead := len(digits) % 3
	if lead > 0 {
		out.WriteString(digits[:lead])
		if len(digits) > lead {
			out.WriteString(",")
		}
	}
	for i := lead; i < len(digits); i += 3 {
		out.WriteString(digits[i : i+3])
		if i+3 < len(digits) {
			out.WriteString(",")
		}
	}
	return out.String()
}

// formatBoundaryTime renders a timestamp for view labels in daemon local time.
func formatBoundaryTime(value time.Time) string {
	if value.IsZero() {
		return ""
	}
	const layout = "1/2/2006, 3:04:05 PM MST"
	location, err := time.LoadLocation("Local")
	if err != nil {
		return value.Format(layout)
	}
	return value.In(location).Format(layout)
}

// progressHeading names the pass for an active job, empty otherwise.
func progressHeading(job model.Job) string {
	if job.State != model.JobStateQueued && job.State != model.JobStateRunning && job.State != model.JobStateCancelling {
		return ""
	}
	switch job.Progress.RunMode {
	case model.RunModeResuming:
		return "Resuming after restart: checking changed work, embedding only what's new"
	case model.RunModeFirstBuild:
		return "Building initial index"
	case model.RunModeForcedReindex:
		return "Forced reindex"
	case model.RunModeChanged:
		return "Indexing new changes"
	default:
		return ""
	}
}

// scopeLabelFor types the denominator from the run mode and unit.
func scopeLabelFor(runMode string, unit string, total int32) string {
	plural := unit
	if total != 1 {
		plural = unit + "s"
	}
	switch runMode {
	case model.RunModeFirstBuild:
		return plural + " (full build)"
	case model.RunModeForcedReindex:
		return plural + " (forced reindex)"
	case model.RunModeResuming, model.RunModeChanged:
		return "changed " + plural
	default:
		return plural
	}
}

// Outcome row glyphs. The lead glyph types each child line before its count so
// a reader sees normal vs transient vs error at a glance.
const (
	glyphEmbedded  = "➕"
	glyphUnchanged = "⏭️"
	glyphRemoved   = "🗑️"
	glyphPending   = "⏳"
	glyphOversize  = "📏"
	glyphError     = "⚠️"
	glyphReused    = "♻️"
)

// reuseCapableRunMode reports whether a pass can serve chunks from already
// embedded vectors, so the chunk tree shows a reused row even at zero. A first
// build has no prior vectors, so it omits the row entirely.
func reuseCapableRunMode(runMode string) bool {
	switch runMode {
	case model.RunModeChanged, model.RunModeResuming, model.RunModeForcedReindex:
		return true
	default:
		return false
	}
}

// resolveOutcomeBreakdown reduces raw progress counters into the shared outcome
// tree. It is the single source of truth for the file-and-chunk breakdown, so
// every status surface renders an identical tree. The file rows partition the
// processed set: embedded plus the clamped seed-reuse remainder (unchanged) plus
// removed plus the skip buckets (pending, oversize, unreadable) sum to Processed.
// Pending is its own counter (FilesPending) from an undelivered conversation,
// distinct from a real unreadable file, so no unit inference is needed.
func resolveOutcomeBreakdown(progress model.Progress) view.OutcomeBreakdown {
	unit := progress.Unit
	if unit == "" {
		unit = "file"
	}

	embedded := progress.FilesEmbedded
	oversize := progress.FilesSkippedOversize
	unreadable := progress.FilesSkippedUnreadable
	pending := progress.FilesPending
	removed := progress.FilesRemoved
	unchanged := max(progress.FilesProcessed-embedded-oversize-unreadable-pending, 0)

	// The changed set is known from the diff before the embed loop reports a
	// FilesTotal, so the denominator and the scope gate fold it in. This keeps
	// the file tree showing "0 of N processed" during the brief pre-embed window
	// rather than vanishing until the first per-file update.
	changedSet := progress.FilesAdded + progress.FilesModified + progress.FilesRemoved
	hasFileScope := progress.FilesTotal > 0 || progress.FilesProcessed > 0 || removed > 0 || changedSet > 0

	processed := embedded + unchanged + removed + pending + oversize + unreadable
	scopeTotal := max(progress.FilesTotal+removed, changedSet)
	chunksTotal := max(progress.ChunksTotal, progress.ChunksReused+progress.ChunksGenerated)
	hasChunks := hasFileScope || chunksTotal > 0 || progress.ChunksGenerated > 0

	return view.OutcomeBreakdown{
		ScopeLabel:  scopeLabelFor(progress.RunMode, unit, scopeTotal),
		Processed:   processed,
		ScopeTotal:  scopeTotal,
		FileRows:    outcomeFileRows(hasFileScope, embedded, unchanged, removed, pending, oversize, unreadable),
		ChunksTotal: chunksTotal,
		ChunkRows:   outcomeChunkRows(hasChunks, progress.RunMode, progress.ChunksGenerated, progress.ChunksReused),
	}
}

// outcomeFileRows builds the file children in fixed order, omitting a zero
// bucket except embedded, which always renders. Pending is transient (an
// undelivered document), oversize and unreadable are deliberate or error skips.
func outcomeFileRows(hasScope bool, embedded, unchanged, removed, pending, oversize, unreadable int32) []view.OutcomeRow {
	if !hasScope {
		return nil
	}
	rows := []view.OutcomeRow{{Glyph: glyphEmbedded, Count: embedded, Label: "embedded"}}
	if unchanged > 0 {
		rows = append(rows, view.OutcomeRow{Glyph: glyphUnchanged, Count: unchanged, Label: "unchanged"})
	}
	if removed > 0 {
		rows = append(rows, view.OutcomeRow{Glyph: glyphRemoved, Count: removed, Label: "removed"})
	}
	if pending > 0 {
		rows = append(rows, view.OutcomeRow{Glyph: glyphPending, Count: pending, Label: "pending, not sent yet"})
	}
	if oversize > 0 {
		rows = append(rows, view.OutcomeRow{Glyph: glyphOversize, Count: oversize, Label: "skipped, too large"})
	}
	if unreadable > 0 {
		rows = append(rows, view.OutcomeRow{Glyph: glyphError, Count: unreadable, Label: "error, unreadable"})
	}
	return rows
}

// outcomeChunkRows builds the chunk children: added always, reused on a
// reuse-capable pass (shown even at zero), omitted for a first build.
func outcomeChunkRows(hasChunks bool, runMode string, added, reused int32) []view.OutcomeRow {
	if !hasChunks {
		return nil
	}
	rows := []view.OutcomeRow{{Glyph: glyphEmbedded, Count: added, Label: "added"}}
	if reuseCapableRunMode(runMode) {
		rows = append(rows, view.OutcomeRow{Glyph: glyphReused, Count: reused, Label: "reused"})
	}
	return rows
}

// resolveProgressSurface reduces a job's progress into the typed view. It is
// the only reader of Progress fields for the compact job surfaces.
func resolveProgressSurface(job model.Job) view.ProgressSurface {
	progress := job.Progress
	scopeUnit := progress.ScopeUnit
	if scopeUnit == "" {
		scopeUnit = progress.Unit
	}
	if scopeUnit == "" {
		scopeUnit = "file"
	}

	active := job.State == model.JobStateQueued || job.State == model.JobStateRunning || job.State == model.JobStateCancelling

	percentLabel := fmt.Sprintf("%.1f%%", progress.OverallPercent)
	if active && !jobScopeKnown(progress) {
		if jobOperation(job.Operation) == jobOperationSync {
			percentLabel = "Changes detected, preparing to index"
		} else {
			percentLabel = "Preparing to index"
		}
	}

	scopeLine := ""
	if progress.FilesAdded > 0 || progress.FilesModified > 0 || progress.FilesRemoved > 0 {
		parts := []string{}
		if progress.FilesAdded > 0 {
			parts = append(parts, fmt.Sprintf("%s %s added", formatCount(progress.FilesAdded), scopeUnit+pluralSuffix(progress.FilesAdded)))
		}
		if progress.FilesModified > 0 {
			parts = append(parts, formatCount(progress.FilesModified)+" modified")
		}
		if progress.FilesRemoved > 0 {
			parts = append(parts, formatCount(progress.FilesRemoved)+" removed")
		}
		scopeLine = "Changed since last sync: " + strings.Join(parts, " · ")
	}

	return view.ProgressSurface{
		Heading:      progressHeading(job),
		Breakdown:    resolveOutcomeBreakdown(progress),
		ScopeLine:    scopeLine,
		PercentLabel: percentLabel,
	}
}

// pluralSuffix returns "s" for counts other than one.
func pluralSuffix(count int32) string {
	if count == 1 {
		return ""
	}
	return "s"
}

// resolveTimingView formats a job's timing block.
func resolveTimingView(job model.Job) view.TimingView {
	timing := view.TimingView{
		StartedLabel:   formatBoundaryTime(job.StartedAt),
		UpdatedLabel:   formatBoundaryTime(job.UpdatedAt),
		CompletedLabel: "",
		DurationLabel:  "",
		DurationWord:   "Elapsed",
	}
	if job.CompletedAt != nil {
		timing.CompletedLabel = formatBoundaryTime(*job.CompletedAt)
		timing.DurationWord = "Duration"
	}
	end := job.UpdatedAt
	if job.CompletedAt != nil && !job.CompletedAt.IsZero() {
		end = *job.CompletedAt
	}
	if !job.StartedAt.IsZero() && !end.IsZero() && !end.Before(job.StartedAt) {
		duration := end.Sub(job.StartedAt).Round(time.Second)
		if duration <= 0 {
			timing.DurationLabel = "0s"
		} else {
			timing.DurationLabel = duration.String()
		}
	}
	return timing
}

// resolveJobEntry assembles the full job view for list and detail surfaces.
func resolveJobEntry(job model.Job, pipelineDegraded bool, supersededByJobID string) view.JobEntryView {
	return view.JobEntryView{
		ID:            job.ID,
		CanonicalPath: job.CanonicalPath,
		Operation:     job.Operation,
		PhaseLabel:    displayJobPhase(job.Progress.Phase),
		Surface:       resolveJobSurface(job, pipelineDegraded, supersededByJobID),
		Progress:      resolveProgressSurface(job),
		Timing:        resolveTimingView(job),
	}
}

func displayJobPhase(phase string) string {
	switch jobPhase(strings.TrimSpace(phase)) {
	case jobPhaseCancelling:
		return "canceling"
	case jobPhaseCancelled:
		return "canceled"
	default:
		return phase
	}
}

// resolveListSummary tallies the job list.
func resolveListSummary(jobs []model.Job, pipelineDegraded bool) view.ListSummary {
	successors := buildJobSuccessors(jobs)
	summary := view.ListSummary{
		Total:      len(jobs),
		Queued:     0,
		Running:    0,
		Canceling:  0,
		Completed:  0,
		Failed:     0,
		Superseded: 0,
		Canceled:   0,
	}
	for _, job := range jobs {
		switch job.State {
		case model.JobStateQueued:
			summary.Queued++
		case model.JobStateRunning:
			summary.Running++
		case model.JobStateCancelling:
			summary.Canceling++
		case model.JobStateCompleted:
			summary.Completed++
		case model.JobStateFailed:
			if resolveJobSurface(job, pipelineDegraded, successors[job.ID]).Superseded {
				summary.Superseded++
			} else {
				summary.Failed++
			}
		case model.JobStateCancelled:
			summary.Canceled++
		}
	}
	return summary
}
