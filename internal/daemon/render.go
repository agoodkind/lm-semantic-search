package daemon

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"goodkind.io/claude-context-go/internal/model"
)

const (
	indexingWarningHeader = "⚠️  **Indexing in Progress**: This codebase is currently being indexed in the background. Search results may be incomplete or inaccurate until indexing completes. Progress: %.1f%%."
	indexingWarningRetry  = "🔁 Retry suggestion: call get_indexing_status (or get_indexing_job for the active job) in ~30s, or call index_codebase with wait=true on the next turn to block until the index is ready. Active job: %s."
	noResultsIndexingTip  = "Note: This codebase is still being indexed. Try searching again after indexing completes, or the query may not match any indexed content."
	searchIndexingTip     = "💡 **Tip**: This codebase is still being indexed. More results may become available as indexing progresses."
)

// formatIndexingWarning builds the in-progress search banner. The banner
// surfaces the current progress percentage, names the active job so the agent
// can poll it directly, and tells the caller exactly how to wait for the
// index to finish.
func formatIndexingWarning(progressPercent float64, activeJobID string) string {
	header := fmt.Sprintf(indexingWarningHeader, progressPercent)
	if activeJobID == "" {
		return header
	}
	return header + "\n" + fmt.Sprintf(indexingWarningRetry, activeJobID)
}

type searchView struct {
	RequestedPath string
	Query         string
	Codebase      model.Codebase
	ActiveJob     *model.Job
	Results       []model.StoredChunk
}

func renderStartIndex(requestedPath string, codebase model.Codebase, job model.Job, deduplicated bool, overlapsCodebaseID string) string {
	if deduplicated {
		return fmt.Sprintf(
			"Background indexing is already running for codebase '%s' using %s splitter.\nCurrent job: %s\n\nIndexing is running in the background. You can search the codebase while indexing is in progress, but results may be incomplete until indexing completes.",
			codebase.CanonicalPath,
			strings.ToUpper(orDefault(job.Config.SplitterType, "ast")),
			job.ID,
		)
	}

	pathInfo := ""
	if requestedPath != "" && requestedPath != codebase.CanonicalPath {
		pathInfo = fmt.Sprintf("\nNote: Input path '%s' was resolved to canonical path '%s'", requestedPath, codebase.CanonicalPath)
	}

	overlap := ""
	if overlapsCodebaseID != "" {
		overlap = fmt.Sprintf("\n⚠️  Overlap: this tree is also covered by codebase %s. Both will index files in the shared subtree independently.", overlapsCodebaseID)
	}

	return fmt.Sprintf(
		"Started background indexing for codebase '%s' using %s splitter.%s%s\n\nIndexing is running in the background. You can search the codebase while indexing is in progress, but results may be incomplete until indexing completes.",
		codebase.CanonicalPath,
		strings.ToUpper(orDefault(job.Config.SplitterType, "ast")),
		pathInfo,
		overlap,
	)
}

func renderClearIndex(codebase model.Codebase, remainingIndexed int, remainingIndexing int) string {
	message := fmt.Sprintf("Successfully cleared codebase '%s'", codebase.CanonicalPath)
	if remainingIndexed > 0 || remainingIndexing > 0 {
		message += fmt.Sprintf("\n%d other indexed codebase(s) and %d indexing codebase(s) remain", remainingIndexed, remainingIndexing)
	}
	return message
}

func renderCancelJob(job model.Job) string {
	if job.State == model.JobStateCancelled {
		return "Cancelled indexing job " + job.ID
	}
	return fmt.Sprintf("Indexing job %s is already %s", job.ID, job.State)
}

func renderSyncIndex(codebase model.Codebase, job model.Job, deduplicated bool) string {
	if deduplicated {
		return fmt.Sprintf("Sync request deduplicated onto active job %s for '%s'", job.ID, codebase.CanonicalPath)
	}
	return fmt.Sprintf("Started sync job %s for '%s'", job.ID, codebase.CanonicalPath)
}

func renderGetIndex(requestedPath string, tracked bool, codebase *model.Codebase, activeJob *model.Job, classification *model.PathClassification) string {
	classificationLine := renderClassificationLine(classification)
	body := renderGetIndexBody(requestedPath, tracked, codebase, activeJob)
	if classificationLine == "" {
		return body
	}
	return body + "\n" + classificationLine
}

func renderGetIndexBody(requestedPath string, tracked bool, codebase *model.Codebase, activeJob *model.Job) string {
	if !tracked || codebase == nil {
		return fmt.Sprintf("❌ Codebase '%s' is not indexed. Please use the index_codebase tool to index it first.", requestedPath)
	}
	// An active job always wins the display. A historical LastFailedRun
	// that lingers in the registry alongside an in-flight retry would
	// otherwise read as the current state and confuse callers.
	if activeJob != nil {
		return renderIndexingActive(codebase, activeJob)
	}
	switch codebase.Status {
	case model.CodebaseStatusNotIndexed:
		return fmt.Sprintf("❌ Codebase '%s' is not indexed. Please use the index_codebase tool to index it first.", requestedPath)
	case model.CodebaseStatusIndexed:
		return renderIndexedDetail(codebase)
	case model.CodebaseStatusIndexing:
		return renderIndexingActive(codebase, activeJob)
	case model.CodebaseStatusFailed, model.CodebaseStatusStale:
		return renderHistoricalFailure(codebase)
	default:
		return fmt.Sprintf("❌ Codebase '%s' is not indexed. Please use the index_codebase tool to index it first.", requestedPath)
	}
}

// renderClassificationLine renders a one-line summary of the per-path
// classification verdict. Returns an empty string when the verdict adds no
// useful information beyond what the body already conveys.
func renderClassificationLine(classification *model.PathClassification) string {
	if classification == nil {
		return ""
	}
	switch classification.Kind {
	case model.PathClassificationInScopeExcluded:
		parts := make([]string, 0, 2)
		if classification.ExcludedByGitignore != "" {
			parts = append(parts, "gitignore="+classification.ExcludedByGitignore)
		}
		if classification.ExcludedByPattern != "" {
			parts = append(parts, "pattern="+classification.ExcludedByPattern)
		}
		if len(parts) == 0 {
			return "🚫 Path is in scope of " + classification.CoveringCodebaseID + " but excluded by an ignore rule."
		}
		return "🚫 Path is in scope of " + classification.CoveringCodebaseID + " but excluded: " + strings.Join(parts, " ")
	case model.PathClassificationOutOfScope:
		return "🛈 Path is not under any tracked codebase."
	case model.PathClassificationInScopeUnindexed:
		return "🛈 Path is in scope of " + classification.CoveringCodebaseID + " but has no chunk row yet."
	case model.PathClassificationInScopeIndexed, model.PathClassificationUnspecified:
		return ""
	default:
		return ""
	}
}

func renderIndexedDetail(codebase *model.Codebase) string {
	view := statusView{
		Name:           filepath.Base(codebase.CanonicalPath),
		HasStats:       false,
		Files:          0,
		Chunks:         0,
		SkippedLine:    "",
		PrepareLabel:   "",
		FilesProcessed: 0,
		FilesTotal:     0,
		ChunksSoFar:    0,
		UpdatedAt:      formatStatusTime(codebase.UpdatedAt),
	}
	if run := codebase.LastSuccessfulRun; run != nil {
		view.HasStats = true
		view.Files = run.IndexedFiles
		view.Chunks = run.TotalChunks
		view.SkippedLine = renderSkippedFiles(run.SkippedFiles)
		view.UpdatedAt = formatStatusTime(run.CompletedAt)
	}
	return renderStatusTemplate("ready.md.tmpl", view)
}

func renderIndexingActive(codebase *model.Codebase, activeJob *model.Job) string {
	view := statusView{
		Name:           filepath.Base(codebase.CanonicalPath),
		HasStats:       false,
		Files:          0,
		Chunks:         0,
		SkippedLine:    "",
		PrepareLabel:   prepareLabel(activeJob),
		FilesProcessed: 0,
		FilesTotal:     0,
		ChunksSoFar:    0,
		UpdatedAt:      formatStatusTime(codebase.UpdatedAt),
	}
	embedding := false
	if activeJob != nil {
		if !activeJob.Progress.LastEventAt.IsZero() {
			view.UpdatedAt = formatStatusTime(activeJob.Progress.LastEventAt)
		}
		view.FilesProcessed = activeJob.Progress.FilesProcessed
		view.FilesTotal = activeJob.Progress.FilesTotal
		view.ChunksSoFar = activeJob.Progress.ChunksGenerated
		embedding = activeJob.Progress.FilesTotal > 0
	}
	if embedding {
		return renderStatusTemplate("indexing.md.tmpl", view)
	}
	return renderStatusTemplate("preparing.md.tmpl", view)
}

// prepareLabel names the phase before embedding starts. A watcher-driven sync
// reaches this phase because a change was detected, so it says so; a full or
// forced reindex is just preparing.
func prepareLabel(job *model.Job) string {
	if job != nil && jobOperation(job.Operation) == jobOperationSync {
		return "Changes detected, preparing to index"
	}
	return "Preparing to index"
}

// renderReconcileMagnitude summarizes a run's work for the job view: how many
// files of the run's scope are processed with the chunk count, and, for a delta
// sync, the added, modified, and removed breakdown. It returns an empty string
// when no counts are recorded yet.
func renderReconcileMagnitude(progress model.Progress) string {
	lines := make([]string, 0, 2)
	if progress.FilesTotal > 0 {
		lines = append(lines, fmt.Sprintf("📄 %d of %d files · 🧩 %d chunks", progress.FilesProcessed, progress.FilesTotal, progress.ChunksGenerated))
	}
	if progress.FilesAdded > 0 || progress.FilesModified > 0 || progress.FilesRemoved > 0 {
		lines = append(lines, fmt.Sprintf("Added %d · Modified %d · Removed %d", progress.FilesAdded, progress.FilesModified, progress.FilesRemoved))
	}
	return strings.Join(lines, "\n")
}

// renderHistoricalFailure reads as past tense so callers do not mistake an
// old failure record for a live one. When the failure carries correlation
// ids it appends a diagnostics line so the operator can grep the daemon log.
func renderHistoricalFailure(codebase *model.Codebase) string {
	if codebase.LastFailedRun == nil {
		return fmt.Sprintf("❌ Codebase '%s' last indexing attempt failed. Call index_codebase to retry.", codebase.CanonicalPath)
	}
	return fmt.Sprintf(
		"❌ Codebase '%s' last indexing attempt failed at %s (%.1f%% complete).\n🚨 Previous error: %s\n💡 Call index_codebase to retry; the previous failure no longer blocks new attempts.%s",
		codebase.CanonicalPath,
		formatLocalTime(codebase.LastFailedRun.FailedAt),
		float64(codebase.LastFailedRun.LastAttemptedPercentage),
		orDefault(codebase.LastFailedRun.Message, "unknown error"),
		renderFailureDiagnostics(codebase.LastFailedRun),
	)
}

// renderFailureDiagnostics returns a leading-newline diagnostics line naming
// the correlation ids behind a failure, or an empty string when none are
// recorded. The ids resolve against the daemon's structured logs.
func renderFailureDiagnostics(failure *model.IndexRunFailure) string {
	refs := make([]string, 0, 2)
	if failure.TraceID != "" {
		refs = append(refs, "trace_id="+failure.TraceID)
	}
	if failure.JobID != "" {
		refs = append(refs, "job_id="+failure.JobID)
	}
	if len(refs) == 0 {
		return ""
	}
	return "\n🔎 Diagnostics: " + strings.Join(refs, " ")
}

func renderListIndexes(codebases []model.Codebase) string {
	if len(codebases) == 0 {
		return "No tracked codebases."
	}

	lines := make([]string, 0, len(codebases)+1)
	lines = append(lines, fmt.Sprintf("Tracked codebases: %d", len(codebases)))
	for _, codebase := range codebases {
		lines = append(lines, fmt.Sprintf("- %s [%s]", codebase.CanonicalPath, codebase.Status))
	}
	return strings.Join(lines, "\n")
}

func renderGetJob(job *model.Job) string {
	if job == nil {
		return "Job not found."
	}
	base := fmt.Sprintf(
		"Job %s\nCodebase: %s\nOperation: %s\nState: %s\nPhase: %s\nProgress: %.1f%%",
		job.ID,
		job.CanonicalPath,
		job.Operation,
		job.State,
		job.Progress.Phase,
		job.Progress.OverallPercent,
	)
	if magnitude := renderReconcileMagnitude(job.Progress); magnitude != "" {
		base += "\n" + magnitude
	}
	return base
}

func renderListJobs(jobs []model.Job) string {
	if len(jobs) == 0 {
		return "No tracked jobs."
	}

	lines := make([]string, 0, len(jobs)+1)
	lines = append(lines, fmt.Sprintf("Tracked jobs: %d", len(jobs)))
	for _, job := range jobs {
		lines = append(lines, fmt.Sprintf("- %s [%s %.1f%%] %s", job.ID, job.State, job.Progress.OverallPercent, job.CanonicalPath))
	}
	return strings.Join(lines, "\n")
}

func renderDoctor(diagnostics []string) string {
	if len(diagnostics) == 0 {
		return "No indexing issues detected."
	}

	lines := make([]string, 0, len(diagnostics)+1)
	lines = append(lines, "Indexing diagnostics:")
	for _, diagnostic := range diagnostics {
		lines = append(lines, "- "+diagnostic)
	}
	return strings.Join(lines, "\n")
}

func renderSearch(view searchView) string {
	warning := ""
	if view.ActiveJob != nil && view.Codebase.Status == model.CodebaseStatusIndexing {
		warning = formatIndexingWarning(view.ActiveJob.Progress.OverallPercent, view.ActiveJob.ID)
	}

	if len(view.Results) == 0 {
		noResults := fmt.Sprintf("No results found for query: %q in codebase '%s'", view.Query, view.Codebase.CanonicalPath)
		if warning == "" {
			return noResults
		}
		return noResults + "\n\n" + warning + "\n\n" + noResultsIndexingTip
	}

	formatted := make([]string, 0, len(view.Results))
	for index, result := range view.Results {
		language := orDefault(result.Language, "text")
		formatted = append(formatted, fmt.Sprintf(
			"%d. Code snippet (%s) [%s]\n   Location: %s:%d-%d\n   Rank: %d\n   Context:\n```%s\n%s\n```",
			index+1,
			language,
			filepath.Base(view.Codebase.CanonicalPath),
			result.RelativePath,
			result.StartLine,
			result.EndLine,
			index+1,
			language,
			strings.TrimSpace(truncateContent(result.Content, 5000)),
		))
	}

	// Lead with the result count and the results themselves so a client that
	// shows only the first line or truncates long output still surfaces the
	// answer. The in-progress warning and tip trail the results.
	header := fmt.Sprintf("Found %d results for query: %q in codebase '%s'", len(view.Results), view.Query, view.Codebase.CanonicalPath)
	body := header + "\n\n" + strings.Join(formatted, "\n\n")
	if warning == "" {
		return body
	}
	return body + "\n\n" + warning + "\n\n" + searchIndexingTip
}

// renderSkippedFiles formats the per-run skipped-file summary for the
// human-facing GetIndex view. The first few paths are listed inline so the
// operator can spot the culprits without grepping the daemon log.
func renderSkippedFiles(skipped []string) string {
	if len(skipped) == 0 {
		return ""
	}
	const maxPreview = 3
	previewLimit := min(len(skipped), maxPreview)
	preview := strings.Join(skipped[:previewLimit], ", ")
	if len(skipped) > maxPreview {
		preview += fmt.Sprintf(", ... (+%d more)", len(skipped)-maxPreview)
	}
	return fmt.Sprintf("⏭️  Skipped: %d non-UTF-8 file(s): %s", len(skipped), preview)
}

// formatStatusTime renders a compact wall-clock time with zone for the status
// header, for example "4:52 PM PDT". The daemon stores UTC, so this converts to
// the host's local zone first, loaded by name so gosmopolitan stays satisfied.
func formatStatusTime(value time.Time) string {
	if value.IsZero() {
		return "unknown"
	}
	const layout = "3:04 PM MST"
	location, err := time.LoadLocation("Local")
	if err != nil {
		return value.Format(layout)
	}
	return value.In(location).Format(layout)
}

// formatLocalTime renders a wall-clock timestamp for human-facing MCP and CLI
// output. The daemon stores every timestamp in UTC (see clock.Now), so this
// converts to the daemon host's local time zone before formatting, including
// the zone abbreviation so operators can recognize the time at a glance. The
// zone is loaded by name rather than via [time.Local] so gosmopolitan stays
// satisfied while the resolution still resolves to the host's local zone.
func formatLocalTime(value time.Time) string {
	if value.IsZero() {
		return "unknown"
	}
	const layout = "1/2/2006, 3:04:05 PM MST"
	location, err := time.LoadLocation("Local")
	if err != nil {
		return value.Format(layout)
	}
	return value.In(location).Format(layout)
}

func truncateContent(content string, limit int) string {
	if len(content) <= limit {
		return content
	}
	if limit <= 3 {
		return content[:limit]
	}
	return content[:limit-3] + "..."
}

func orDefault(value string, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}
