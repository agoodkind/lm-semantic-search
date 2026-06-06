package daemon

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"goodkind.io/lm-semantic-search/internal/gitworktree"
	"goodkind.io/lm-semantic-search/internal/model"
)

const (
	noResultsIndexingTip = "Note: This codebase is still being indexed. Try searching again after indexing completes, or the query may not match any indexed content."
	searchIndexingTip    = "💡 **Tip**: This codebase is still being indexed. More results may become available as indexing progresses."
)

type searchView struct {
	RequestedPath string
	Query         string
	Codebase      model.Codebase
	ActiveJob     *model.Job
	Results       []model.StoredChunk
	StateNote     string
}

type jobPhase string

const (
	jobPhaseCancelling jobPhase = "cancelling"
	jobPhaseCancelled  jobPhase = "cancelled"
)

func renderStartIndex(requestedPath string, codebase model.Codebase, job model.Job, deduplicated bool, overlapsCodebaseID string, mergeNote string) string {
	if deduplicated {
		return fmt.Sprintf(
			"Background indexing is already running for codebase '%s' using %s splitter.\nCurrent job: %s\n\nIndexing is running in the background. You can search the codebase while indexing is in progress, but results may be incomplete until indexing completes.",
			codebase.CanonicalPath,
			strings.ToUpper(orDefault(job.Config.SplitterType, "ast")),
			job.ID,
		)
	}

	// The merge note already explains the relationship between the requested path
	// and the codebase, so the plain "resolved to canonical path" line would only
	// repeat it; it renders only in the ordinary, non-merge case.
	pathInfo := ""
	if mergeNote == "" && requestedPath != "" && requestedPath != codebase.CanonicalPath {
		pathInfo = fmt.Sprintf("\nNote: Input path '%s' was resolved to canonical path '%s'", requestedPath, codebase.CanonicalPath)
	}

	merge := ""
	if mergeNote != "" {
		merge = "\n" + mergeNote
	}

	overlap := ""
	if overlapsCodebaseID != "" {
		overlap = fmt.Sprintf("\n⚠️  Overlap: this tree is also covered by codebase %s. Both will index files in the shared subtree independently.", overlapsCodebaseID)
	}

	return fmt.Sprintf(
		"Started background indexing for codebase '%s' using %s splitter.%s%s%s\n\nIndexing is running in the background. You can search the codebase while indexing is in progress, but results may be incomplete until indexing completes.",
		codebase.CanonicalPath,
		strings.ToUpper(orDefault(job.Config.SplitterType, "ast")),
		pathInfo,
		merge,
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
		return "Canceled indexing job " + job.ID
	}
	return fmt.Sprintf("Indexing job %s is already %s", job.ID, displayJobState(job.State))
}

func renderSyncIndex(codebase model.Codebase, job model.Job, deduplicated bool) string {
	if deduplicated {
		return fmt.Sprintf("Sync request deduplicated onto active job %s for '%s'", job.ID, codebase.CanonicalPath)
	}
	return fmt.Sprintf("Started sync job %s for '%s'", job.ID, codebase.CanonicalPath)
}

func renderGetIndex(requestedPath string, tracked bool, codebase *model.Codebase, activeJob *model.Job, classification *model.PathClassification, indexedDescendants []model.Codebase) string {
	// A path that is not indexed as its own codebase but contains already-indexed
	// sub-folders reads as an offer to merge them into one larger index, rather
	// than a bare "not indexed" dead end.
	if !tracked && len(indexedDescendants) > 0 {
		return renderIndexedDescendantsHint(requestedPath, indexedDescendants)
	}
	lines := []string{renderGetIndexBody(requestedPath, tracked, codebase, activeJob)}
	if symlinkLine := renderSymlinkResolution(requestedPath); symlinkLine != "" {
		lines = append(lines, symlinkLine)
	}
	if worktreeLine := renderWorktreeRelation(requestedPath); worktreeLine != "" {
		lines = append(lines, worktreeLine)
	}
	if coverageLine := renderCoveringResolution(requestedPath, tracked, codebase); coverageLine != "" {
		lines = append(lines, coverageLine)
	}
	if classificationLine := renderClassificationLine(classification); classificationLine != "" {
		lines = append(lines, classificationLine)
	}
	return strings.Join(lines, "\n")
}

// renderIndexedDescendantsHint replaces the bare not-indexed message for a path
// that already has indexed sub-folders. It names the sub-folders, totals their
// indexed files, and points at the one command that builds a merged parent
// index reusing their embeddings.
func renderIndexedDescendantsHint(requestedPath string, descendants []model.Codebase) string {
	var totalFiles int32
	names := make([]string, 0, len(descendants))
	for _, child := range descendants {
		names = append(names, child.CanonicalPath)
		if child.LastSuccessfulRun != nil {
			totalFiles += child.LastSuccessfulRun.IndexedFiles
		}
	}
	return fmt.Sprintf(
		"🛈 '%s' is not indexed on its own, but %d already-indexed file(s) live under sub-folder(s): %s\n"+
			"Build one merged index that reuses those embeddings by running: index_codebase %s",
		requestedPath, totalFiles, strings.Join(names, ", "), requestedPath,
	)
}

// renderCoveringResolution names the larger index a nested query resolved to,
// scoped to the sub-path, so the operator sees that a sub-folder query is served
// by the covering parent index rather than a separate one.
func renderCoveringResolution(requestedPath string, tracked bool, codebase *model.Codebase) string {
	if !tracked || codebase == nil {
		return ""
	}
	prefix := subtreePrefix(requestedPath, codebase.CanonicalPath)
	if prefix == "" {
		return ""
	}
	return fmt.Sprintf("🔁 Resolved to larger index '%s' (scoped to %s/).", codebase.CanonicalPath, prefix)
}

// renderSymlinkResolution names the real path a symlinked query path resolves
// to, or returns an empty string when the query path traverses no symlink. A
// codebase's identity is the resolved real path, so when the caller passes a
// symlink this line states which real directory it points at.
func renderSymlinkResolution(requestedPath string) string {
	if strings.TrimSpace(requestedPath) == "" {
		return ""
	}
	absolutePath, err := filepath.Abs(requestedPath)
	if err != nil {
		return ""
	}
	resolved, err := filepath.EvalSymlinks(absolutePath)
	if err != nil || resolved == absolutePath {
		return ""
	}
	return "🔗 symlink resolved to: " + resolved
}

// renderWorktreeRelation names the main checkout and branch a linked worktree
// belongs to, so the operator sees that this index is one branch of a shared
// repository. It returns an empty string for the main worktree (no separate
// checkout to point at) and for a non-git path.
func renderWorktreeRelation(requestedPath string) string {
	if strings.TrimSpace(requestedPath) == "" {
		return ""
	}
	absolutePath, err := filepath.Abs(requestedPath)
	if err != nil {
		return ""
	}
	info, ok := gitworktree.Resolve(absolutePath)
	if !ok || !info.Linked {
		return ""
	}
	mainCheckout := filepath.Dir(info.CommonDir)
	if info.Detached {
		return fmt.Sprintf("🌿 git worktree of %s (detached HEAD %s)", mainCheckout, info.Head)
	}
	return fmt.Sprintf("🌿 git worktree of %s (branch %s)", mainCheckout, info.Branch)
}

func renderGetIndexBody(requestedPath string, tracked bool, codebase *model.Codebase, activeJob *model.Job) string {
	if !tracked || codebase == nil {
		return fmt.Sprintf("❌ Codebase '%s' is not indexed. Please use the index_codebase tool to index it first.", requestedPath)
	}
	// An active job otherwise wins the display. A historical LastFailedRun
	// that lingers in the registry alongside an in-flight retry would
	// otherwise read as the current state and confuse callers. The one
	// exception is a background incremental sync over an already-indexed
	// codebase: the live collection stays searchable while it runs, so the
	// ready view holds with a background note rather than a busy takeover.
	if activeJob != nil {
		if isBackgroundSyncReconcile(codebase, activeJob) {
			return renderIndexedWithSync(codebase, activeJob)
		}
		return renderIndexingActive(codebase, activeJob)
	}
	switch codebase.Status {
	case model.CodebaseStatusNotIndexed:
		return fmt.Sprintf("❌ Codebase '%s' is not indexed. Please use the index_codebase tool to index it first.", requestedPath)
	case model.CodebaseStatusIndexed:
		return renderIndexedDetail(codebase)
	case model.CodebaseStatusIndexing:
		return renderIndexingActive(codebase, activeJob)
	case model.CodebaseStatusFailed:
		return renderHistoricalFailure(codebase)
	case model.CodebaseStatusStale:
		return renderStaleStatus(codebase)
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

// blankStatusView returns a fully zeroed status view with only the name and
// timestamp set, so each caller fills the subset its template reads.
func blankStatusView(name string, updatedAt string) statusView {
	return statusView{
		Name:                   name,
		HasStats:               false,
		Files:                  0,
		Chunks:                 0,
		SkippedLine:            "",
		PrepareLabel:           "",
		Percent:                0,
		FilesProcessed:         0,
		FilesTotal:             0,
		ChunksSoFar:            0,
		FilesInCodebase:        0,
		FilesChanged:           0,
		FilesUnchanged:         0,
		FilesProcessedChanged:  0,
		FilesReEmbedded:        0,
		FilesRemoved:           0,
		FilesSkippedOversize:   0,
		FilesSkippedUnreadable: 0,
		ChunksAdded:            0,
		ChunksTotal:            0,
		UpdatedAt:              updatedAt,
		SyncNote:               "",
	}
}

func renderIndexedDetail(codebase *model.Codebase) string {
	view := blankStatusView(filepath.Base(codebase.CanonicalPath), formatStatusTime(codebase.UpdatedAt))
	if run := codebase.LastSuccessfulRun; run != nil {
		view.HasStats = true
		view.Files = run.IndexedFiles
		view.Chunks = run.TotalChunks
		view.SkippedLine = renderSkippedFiles(run.SkippedFiles)
		view.UpdatedAt = formatStatusTime(run.CompletedAt)
	}
	return renderStatusTemplate("ready.md.tmpl", view)
}

// isBackgroundSyncReconcile reports whether the active job is a background
// incremental sync over a codebase that already has a successful run. That
// delta writes to the live collection, so the index stays searchable while it
// runs and the ready view holds. A from-scratch "index" build (staging, not
// promoted) and a "streaming_reindex" rebuild keep their busy takeover.
func isBackgroundSyncReconcile(codebase *model.Codebase, job *model.Job) bool {
	return job != nil &&
		jobOperation(job.Operation) == jobOperationSync &&
		codebase.LastSuccessfulRun != nil
}

// backgroundSyncNote is the one-line note appended to the ready view while a
// background sync runs. Before the diff is captured the changed counts are
// zero, so it states only that a sync is underway.
func backgroundSyncNote(job *model.Job) string {
	progress := job.Progress
	changed := progress.FilesAdded + progress.FilesModified + progress.FilesRemoved
	if changed == 0 {
		return "🔄 changes detected, syncing in the background"
	}
	noun := "files"
	if changed == 1 {
		noun = "file"
	}
	percent := int32(progress.OverallPercent + 0.5)
	return fmt.Sprintf("🔄 syncing %d changed %s in the background (%d%%)", changed, noun, percent)
}

// renderIndexedWithSync renders the ready view for an already-indexed codebase
// with a background sync in flight, appending the sync note and preferring the
// job's last event time for the freshness stamp.
func renderIndexedWithSync(codebase *model.Codebase, job *model.Job) string {
	view := blankStatusView(filepath.Base(codebase.CanonicalPath), formatStatusTime(codebase.UpdatedAt))
	if run := codebase.LastSuccessfulRun; run != nil {
		view.HasStats = true
		view.Files = run.IndexedFiles
		view.Chunks = run.TotalChunks
		view.SkippedLine = renderSkippedFiles(run.SkippedFiles)
		view.UpdatedAt = formatStatusTime(run.CompletedAt)
	}
	if !job.Progress.LastEventAt.IsZero() {
		view.UpdatedAt = formatStatusTime(job.Progress.LastEventAt)
	}
	view.SyncNote = backgroundSyncNote(job)
	return renderStatusTemplate("ready.md.tmpl", view)
}

func renderIndexingActive(codebase *model.Codebase, activeJob *model.Job) string {
	view := blankStatusView(filepath.Base(codebase.CanonicalPath), formatStatusTime(codebase.UpdatedAt))
	view.PrepareLabel = prepareLabel(activeJob)
	embedding := false
	if activeJob != nil {
		progress := activeJob.Progress
		if !progress.LastEventAt.IsZero() {
			view.UpdatedAt = formatStatusTime(progress.LastEventAt)
		}
		changed := progress.FilesAdded + progress.FilesModified + progress.FilesRemoved
		view.Percent = int32(progress.OverallPercent + 0.5)
		view.FilesProcessed = progress.FilesProcessed
		view.FilesTotal = progress.FilesTotal
		view.FilesInCodebase = progress.FilesInCodebase
		view.FilesChanged = changed
		view.FilesUnchanged = max(progress.FilesInCodebase-changed, 0)
		view.FilesReEmbedded = progress.FilesEmbedded
		view.FilesRemoved = progress.FilesRemoved
		view.FilesSkippedOversize = progress.FilesSkippedOversize
		view.FilesSkippedUnreadable = progress.FilesSkippedUnreadable
		view.FilesProcessedChanged = progress.FilesEmbedded + progress.FilesRemoved + progress.FilesSkippedOversize + progress.FilesSkippedUnreadable
		view.ChunksSoFar = progress.ChunksGenerated
		view.ChunksAdded = progress.ChunksGenerated
		view.ChunksTotal = progress.ChunksTotal
		if view.ChunksTotal == 0 && codebase.LastSuccessfulRun != nil {
			view.ChunksTotal = codebase.LastSuccessfulRun.TotalChunks
		}
		// The work scope is known once the loop has a total (a from-scratch
		// build) or the diff is captured (a delta sync sets FilesInCodebase).
		// Showing the indexing view from that point, rather than waiting for the
		// first file to embed, keeps a slow first embed from reading as a stall.
		embedding = progress.FilesTotal > 0 || progress.FilesInCodebase > 0
	}
	if !embedding {
		return renderStatusTemplate("preparing.md.tmpl", view)
	}
	if activeJob != nil && jobOperation(activeJob.Operation) == jobOperationIndex {
		return renderStatusTemplate("building.md.tmpl", view)
	}
	return renderStatusTemplate("incremental.md.tmpl", view)
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
		return fmt.Sprintf("❌ Codebase '%s' is not currently indexed. Call index_codebase to build it.", codebase.CanonicalPath)
	}
	return fmt.Sprintf(
		"❌ Codebase '%s' is not currently indexed.\n🚧 %s\n💡 Call index_codebase to build it.%s",
		codebase.CanonicalPath,
		orDefault(codebase.LastFailedRun.Message, "the index could not be built"),
		renderFailureDiagnostics(codebase.LastFailedRun),
	)
}

func renderStaleStatus(codebase *model.Codebase) string {
	if codebase.LastFailedRun == nil {
		return fmt.Sprintf(
			"⚠️ Codebase '%s' is stale because its semantic collection is missing.\n💡 The daemon will rebuild it automatically on the next background repair pass.",
			codebase.CanonicalPath,
		)
	}
	return fmt.Sprintf(
		"⚠️ Codebase '%s' is stale since %s.\n🚨 Repair detail: %s\n💡 The daemon will retry automatic rebuild while the codebase remains stale.%s",
		codebase.CanonicalPath,
		formatLocalTime(codebase.LastFailedRun.FailedAt),
		orDefault(codebase.LastFailedRun.Message, "semantic collection is missing"),
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

// jobScopeKnown reports whether the daemon has measured the work scope for a
// job. It mirrors the gate renderIndexingActive uses to leave the "Preparing"
// view: before the scope is known a 0% reads as stalled rather than measured.
func jobScopeKnown(progress model.Progress) bool {
	return progress.FilesTotal > 0 || progress.FilesInCodebase > 0
}

// jobProgressDisplay returns the percent for a job whose scope is known, and the
// preparing label for an active job that has not measured progress yet, so a
// just-started job never reads as a misleading 0.0%. Terminal jobs always show
// their percent.
func jobProgressDisplay(job model.Job) string {
	active := job.State == model.JobStateQueued ||
		job.State == model.JobStateRunning ||
		job.State == model.JobStateCancelling
	if active && !jobScopeKnown(job.Progress) {
		return prepareLabel(&job)
	}
	return fmt.Sprintf("%.1f%%", job.Progress.OverallPercent)
}

func renderGetJob(job *model.Job) string {
	if job == nil {
		return "Job not found."
	}
	lines := []string{
		"Job " + job.ID,
		"Codebase: " + job.CanonicalPath,
		"Operation: " + job.Operation,
		"State: " + displayJobState(job.State),
		"Phase: " + displayJobPhase(job.Progress.Phase),
		"Progress: " + jobProgressDisplay(*job),
	}
	lines = append(lines, renderJobTimingLines(*job)...)
	if magnitude := renderReconcileMagnitude(job.Progress); magnitude != "" {
		lines = append(lines, magnitude)
	}
	if job.Error != nil && strings.TrimSpace(job.Error.Message) != "" {
		lines = append(lines, "Error: "+job.Error.Message)
	}
	return strings.Join(lines, "\n")
}

func renderListJobs(jobs []model.Job) string {
	if len(jobs) == 0 {
		return "No tracked jobs."
	}

	activeJobs := make([]model.Job, 0, len(jobs))
	terminalJobs := make([]model.Job, 0, len(jobs))
	stateCounts := map[model.JobState]int{}
	for _, job := range jobs {
		stateCounts[job.State]++
		switch job.State {
		case model.JobStateQueued, model.JobStateRunning, model.JobStateCancelling:
			activeJobs = append(activeJobs, job)
		case model.JobStateCompleted, model.JobStateFailed, model.JobStateCancelled:
			terminalJobs = append(terminalJobs, job)
		default:
			terminalJobs = append(terminalJobs, job)
		}
	}

	lines := make([]string, 0, 32)
	lines = append(lines, fmt.Sprintf("Tracked jobs: %d total", len(jobs)))
	lines = append(lines, fmt.Sprintf(
		"Active: %d queued, %d running, %d canceling",
		stateCounts[model.JobStateQueued],
		stateCounts[model.JobStateRunning],
		stateCounts[model.JobStateCancelling],
	))
	lines = append(lines, fmt.Sprintf(
		"Terminal: %d completed, %d failed, %d canceled",
		stateCounts[model.JobStateCompleted],
		stateCounts[model.JobStateFailed],
		stateCounts[model.JobStateCancelled],
	))

	if len(activeJobs) == 0 {
		lines = append(lines, "", "No active jobs.")
	} else {
		lines = append(lines, "", "Active jobs:")
		for _, job := range activeJobs {
			lines = append(lines, renderJobListEntry(job)...)
		}
	}

	const recentTerminalLimit = 8
	if len(terminalJobs) == 0 {
		return strings.Join(lines, "\n")
	}
	lines = append(lines, "")
	if len(terminalJobs) > recentTerminalLimit {
		lines = append(lines, fmt.Sprintf("Recent terminal jobs: showing %d of %d", recentTerminalLimit, len(terminalJobs)))
		for _, job := range terminalJobs[:recentTerminalLimit] {
			lines = append(lines, renderJobListEntry(job)...)
		}
		lines = append(lines, "Use `job get JOB_ID` or `--json` for full history.")
		return strings.Join(lines, "\n")
	}

	lines = append(lines, fmt.Sprintf("Terminal jobs: %d", len(terminalJobs)))
	for _, job := range terminalJobs {
		lines = append(lines, renderJobListEntry(job)...)
	}
	return strings.Join(lines, "\n")
}

func displayJobState(state model.JobState) string {
	switch state {
	case model.JobStateQueued:
		return string(model.JobStateQueued)
	case model.JobStateRunning:
		return string(model.JobStateRunning)
	case model.JobStateCancelling:
		return "canceling"
	case model.JobStateCompleted:
		return string(model.JobStateCompleted)
	case model.JobStateFailed:
		return string(model.JobStateFailed)
	case model.JobStateCancelled:
		return "canceled"
	default:
		return string(state)
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

func renderJobListEntry(job model.Job) []string {
	lines := []string{
		fmt.Sprintf(
			"- %s [%s · %s] %s %s",
			job.ID,
			displayJobState(job.State),
			jobProgressDisplay(job),
			job.Operation,
			job.CanonicalPath,
		),
	}
	lines = append(lines, renderJobTimingLines(job)...)
	if magnitude := renderReconcileMagnitude(job.Progress); magnitude != "" {
		for line := range strings.SplitSeq(magnitude, "\n") {
			lines = append(lines, "  "+line)
		}
	}
	if job.Error != nil && strings.TrimSpace(job.Error.Message) != "" {
		lines = append(lines, "  Error: "+job.Error.Message)
	}
	return lines
}

func renderJobTimingLines(job model.Job) []string {
	lines := []string{
		"  Started: " + formatLocalTime(job.StartedAt),
		"  Updated: " + formatLocalTime(job.UpdatedAt),
	}
	if job.CompletedAt != nil {
		lines = append(lines, "  Completed: "+formatLocalTime(*job.CompletedAt))
	}
	if duration := formatJobDuration(job); duration != "" {
		label := "Elapsed"
		if job.CompletedAt != nil {
			label = "Duration"
		}
		lines = append(lines, "  "+label+": "+duration)
	}
	return lines
}

func formatJobDuration(job model.Job) string {
	if job.StartedAt.IsZero() {
		return ""
	}
	end := job.UpdatedAt
	if job.CompletedAt != nil && !job.CompletedAt.IsZero() {
		end = *job.CompletedAt
	}
	if end.IsZero() || end.Before(job.StartedAt) {
		return ""
	}
	duration := end.Sub(job.StartedAt).Round(time.Second)
	if duration <= 0 {
		return "0s"
	}
	return duration.String()
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
	// When a run is in flight, the search response carries the same status block
	// get_indexing_status returns, so the caller sees the file and chunk progress
	// inline and does not need a second tool call to learn the index is building.
	status := renderSearchIndexingStatus(view)

	if len(view.Results) == 0 {
		noResults := fmt.Sprintf("No results found for query: %q in codebase '%s'", view.Query, view.Codebase.CanonicalPath)
		if status == "" && view.StateNote == "" {
			return noResults
		}
		sections := []string{noResults}
		if status != "" {
			sections = append(sections, status)
		}
		if view.StateNote != "" {
			sections = append(sections, view.StateNote)
		} else if status != "" {
			sections = append(sections, searchStatusTip(view, false))
		}
		return strings.Join(sections, "\n\n")
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
	if status == "" && view.StateNote == "" {
		return body
	}
	sections := []string{body}
	if status != "" {
		sections = append(sections, status)
	}
	if view.StateNote != "" {
		sections = append(sections, view.StateNote)
	} else if status != "" {
		sections = append(sections, searchStatusTip(view, true))
	}
	return strings.Join(sections, "\n\n")
}

// searchStatusTip picks the trailing tip for a search response that has a run
// in flight. A background-sync reconcile keeps the live collection searchable,
// so its results are current; a from-scratch build or rebuild may still be
// filling in, so it keeps the existing "still being indexed" tips.
func searchStatusTip(view searchView, hasResults bool) string {
	if isBackgroundSyncReconcile(&view.Codebase, view.ActiveJob) {
		return "💡 Results are current; a few changed files are still syncing in the background."
	}
	if hasResults {
		return searchIndexingTip
	}
	return noResultsIndexingTip
}

// renderSearchIndexingStatus returns the in-progress status block for a search
// response, matching what get_indexing_status shows: the indexing or preparing
// detail plus the symlink-resolution line when the queried path is a symlink.
// It returns an empty string when no run is in flight.
func renderSearchIndexingStatus(view searchView) string {
	if view.ActiveJob == nil {
		return ""
	}
	codebase := view.Codebase
	detail := renderIndexingActive(&codebase, view.ActiveJob)
	if isBackgroundSyncReconcile(&codebase, view.ActiveJob) {
		detail = renderIndexedWithSync(&codebase, view.ActiveJob)
	}
	lines := []string{detail}
	if symlinkLine := renderSymlinkResolution(view.RequestedPath); symlinkLine != "" {
		lines = append(lines, symlinkLine)
	}
	if worktreeLine := renderWorktreeRelation(view.RequestedPath); worktreeLine != "" {
		lines = append(lines, worktreeLine)
	}
	return strings.Join(lines, "\n")
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
