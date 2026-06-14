// Package render formats resolved view models into human-facing text.
package render

import (
	"fmt"
	"strconv"
	"strings"

	"goodkind.io/lm-semantic-search/internal/view"
)

const (
	noResultsIndexingTip = "Note: This codebase is still being indexed. Try searching again after indexing completes, or the query may not match any indexed content."
	searchIndexingTip    = "💡 **Tip**: This codebase is still being indexed. More results may become available as indexing progresses."
)

const (
	displayFailed     view.Display = "failed"
	displayMissing    view.Display = "missing"
	displayStale      view.Display = "stale"
	displayDiscovered view.Display = "discovered"
)

// StartIndex formats the start-index acknowledgment.
func StartIndex(startIndex view.StartIndexView) string {
	return renderStartIndex(startIndex)
}

// MutationAck formats a mutation acknowledgment.
func MutationAck(ack view.MutationAckView) string {
	return renderMutationAck(ack)
}

// GetIndex formats the full get-index display body.
func GetIndex(getIndex view.GetIndexView) string {
	return renderGetIndex(getIndex)
}

// ListIndexes formats the codebase list.
func ListIndexes(views []view.CodebaseRowView) string {
	return renderListIndexes(views)
}

// GetJob formats one job detail view.
func GetJob(entry view.JobEntryView, found bool) string {
	return renderGetJob(entry, found)
}

// ListJobs formats active and terminal job entries.
func ListJobs(summary view.ListSummary, active []view.JobEntryView, terminal []view.JobEntryView) string {
	return renderListJobs(summary, active, terminal)
}

// Doctor formats daemon diagnostics.
func Doctor(doctor view.DoctorView) string {
	return renderDoctor(doctor)
}

// Search formats code search results.
func Search(searchView view.SearchView) string {
	return renderSearch(searchView)
}

// ConversationSearch formats conversation search results.
func ConversationSearch(conversationView view.ConversationSearchView) string {
	return renderConversationSearch(conversationView)
}

func renderStartIndex(startIndex view.StartIndexView) string {
	if startIndex.Deduplicated {
		return fmt.Sprintf(
			"Background indexing is already running for codebase '%s' using %s splitter.\nCurrent job: %s\n\nIndexing is running in the background. You can search the codebase while indexing is in progress, but results may be incomplete until indexing completes.",
			startIndex.CanonicalPath,
			strings.ToUpper(orDefault(startIndex.SplitterType, "ast")),
			startIndex.JobID,
		)
	}

	// The merge note already explains the relationship between the requested path
	// and the codebase, so the plain "resolved to canonical path" line would only
	// repeat it; it renders only in the ordinary, non-merge case.
	pathInfo := ""
	if startIndex.MergeNote == "" && startIndex.RequestedPath != "" && startIndex.RequestedPath != startIndex.CanonicalPath {
		pathInfo = fmt.Sprintf("\nNote: Input path '%s' was resolved to canonical path '%s'", startIndex.RequestedPath, startIndex.CanonicalPath)
	}

	merge := ""
	if startIndex.MergeNote != "" {
		merge = "\n" + startIndex.MergeNote
	}

	overlap := ""
	if startIndex.OverlapsCodebaseID != "" {
		overlap = fmt.Sprintf("\n⚠️  Overlap: this tree is also covered by codebase %s. Both will index files in the shared subtree independently.", startIndex.OverlapsCodebaseID)
	}

	return fmt.Sprintf(
		"Started background indexing for codebase '%s' using %s splitter.%s%s%s\n\nIndexing is running in the background. You can search the codebase while indexing is in progress, but results may be incomplete until indexing completes.",
		startIndex.CanonicalPath,
		strings.ToUpper(orDefault(startIndex.SplitterType, "ast")),
		pathInfo,
		merge,
		overlap,
	)
}

func renderMutationAck(ack view.MutationAckView) string {
	switch ack.Kind {
	case view.AckClear:
		return fmt.Sprintf("Successfully cleared codebase '%s'", ack.Path)
	case view.AckCancel:
		if !ack.AlreadyTerminal {
			return "Canceled indexing job " + ack.JobID
		}
		return fmt.Sprintf("Indexing job %s is already %s", ack.JobID, ack.StateLabel)
	case view.AckSync:
		if ack.Deduplicated {
			return fmt.Sprintf("Sync request deduplicated onto active job %s for '%s'", ack.JobID, ack.Path)
		}
		return fmt.Sprintf("Started sync job %s for '%s'", ack.JobID, ack.Path)
	case view.AckRegisterConversation:
		return fmt.Sprintf(
			"Registered conversation collection '%s' as codebase %s using Milvus collection '%s'.",
			ack.CollectionID,
			ack.CodebaseID,
			ack.CollectionName,
		)
	case view.AckUpsertConversation:
		return fmt.Sprintf(
			"Started conversation ingest job %s for collection '%s' with %d %s.",
			ack.JobID,
			ack.CollectionID,
			ack.DocumentCount,
			plural("document", ack.DocumentCount),
		)
	case view.AckDeleteConversation:
		return fmt.Sprintf(
			"Started conversation delete job %s for conversation '%s' in collection '%s'.",
			ack.JobID,
			ack.ConversationID,
			ack.CollectionID,
		)
	case view.AckManifest:
		return fmt.Sprintf("Conversation collection '%s' needs %d of %d %s.", ack.CollectionID, ack.NeededCount, ack.TotalCount, plural("conversation", ack.TotalCount))
	default:
		return ""
	}
}

func renderGetIndex(getIndex view.GetIndexView) string {
	if !getIndex.Tracked && getIndex.DescendantsHint != "" {
		return getIndex.DescendantsHint
	}
	lines := []string{renderGetIndexBody(getIndex)}
	lines = append(lines, getIndex.ResolutionLines...)
	if getIndex.CoverageLine != "" {
		lines = append(lines, getIndex.CoverageLine)
	}
	if getIndex.ClassificationLine != "" {
		lines = append(lines, getIndex.ClassificationLine)
	}
	return strings.Join(lines, "\n")
}

func renderGetIndexBody(getIndex view.GetIndexView) string {
	// An untracked path is genuinely not indexed: offer to build it. This is the
	// only "not indexed" message; a tracked codebase always presents as one of
	// the live states below.
	if !getIndex.Tracked {
		return fmt.Sprintf("❌ Codebase '%s' is not indexed. Please use the index_codebase tool to index it first.", getIndex.RequestedPath)
	}
	// The display status is the single source of truth; the renderers below only
	// fill in detail for the bucket it picks. A live background sync over an
	// already-indexed codebase keeps the searchable ready view with a sync note
	// rather than a busy takeover. Under a hard dependency outage an incomplete
	// codebase folds to "waiting"; the banner above carries the cause, so the
	// waiting view names none.
	switch getIndex.Display {
	case displayFailed:
		return renderHistoricalFailure(getIndex.CanonicalPath, getIndex.Failure)
	case displayMissing:
		return renderMissingStatus(getIndex.CanonicalPath)
	case displayStale:
		return renderStaleStatus(getIndex.CanonicalPath, getIndex.Failure)
	default:
		return renderStatusBody(getIndex.Status, getIndex.TemplateName)
	}
}

func renderStatusBody(statusView view.StatusView, templateName string) string {
	block := strings.Join(BreakdownLines(statusView.Breakdown), "\n")
	return renderStatusTemplate(templateName, statusTemplateData{StatusView: statusView, BreakdownBlock: block})
}

// renderMissingStatus reads as a current condition, not a failure: the source
// directory is gone, so the index is held until the directory returns or the
// caller drops it.
func renderMissingStatus(canonicalPath string) string {
	return fmt.Sprintf(
		"🚫 Codebase '%s' source directory is missing.\n💡 Re-create the directory to resume indexing, or call clear_index to drop the index.",
		canonicalPath,
	)
}

// renderHistoricalFailure reads as past tense so callers do not mistake an
// old failure record for a live one. When the failure carries correlation
// ids it appends a diagnostics line so the operator can grep the daemon log.
func renderHistoricalFailure(canonicalPath string, failure view.FailureSurface) string {
	if !failure.HasFailure {
		return fmt.Sprintf("❌ Codebase '%s' could not be indexed. Re-run index_codebase to retry.", canonicalPath)
	}
	return fmt.Sprintf(
		"❌ Codebase '%s' could not be indexed.\n🚧 %s\n💡 Re-run index_codebase; if it keeps failing, check the daemon log via the failed-job reference below.%s",
		canonicalPath,
		orDefault(failure.Message, "the index could not be built"),
		renderFailureDiagnostics(failure),
	)
}

func renderStaleStatus(canonicalPath string, failure view.FailureSurface) string {
	if !failure.HasFailure {
		return fmt.Sprintf(
			"⚠️ Codebase '%s' is stale because its semantic collection is missing.\n💡 The daemon will rebuild it automatically on the next background repair pass.",
			canonicalPath,
		)
	}
	return fmt.Sprintf(
		"⚠️ Codebase '%s' is stale since %s.\n🚨 Repair detail: %s\n💡 The daemon will retry automatic rebuild while the codebase remains stale.%s",
		canonicalPath,
		failure.FailedAtLabel,
		orDefault(failure.Message, "semantic collection is missing"),
		renderFailureDiagnostics(failure),
	)
}

// renderFailureDiagnostics returns a leading-newline line naming the failed job
// and its trace id, or an empty string when neither is recorded. It leads with
// the job so it reads as the past failure's reference rather than a second
// request-trace line, leaving the envelope header as the only "trace_id=" line.
// It formats the resolved failure view, never the raw failure record.
func renderFailureDiagnostics(failure view.FailureSurface) string {
	if failure.JobID == "" && failure.TraceID == "" {
		return ""
	}
	label := "Failed job"
	if failure.JobID != "" {
		label = "Failed job " + failure.JobID
	}
	if failure.TraceID != "" {
		label += " (trace_id=" + failure.TraceID + ")"
	}
	return "\n🔎 " + label
}

func renderListIndexes(views []view.CodebaseRowView) string {
	if len(views) == 0 {
		return "No tracked codebases."
	}

	lines := make([]string, 0, len(views)+1)
	lines = append(lines, "Tracked "+countWord("codebase", len(views))+":")
	for _, row := range views {
		line := fmt.Sprintf("- %s  %s  [%s]", row.ID, row.CanonicalPath, row.Display)
		// A discovered worktree is not built yet; surface its reuse forecast so the
		// row reads as a cheap pending build rather than blank, matching what the
		// status detail shows.
		if row.Display == displayDiscovered && row.ReuseSiblingCount > 0 {
			line += fmt.Sprintf("  ♻️ reuses %d sibling %s", row.ReuseSiblingCount, plural("collection", int(row.ReuseSiblingCount)))
		}
		lines = append(lines, line)
		// An actively-indexing codebase shows its live breakdown tree inline, the
		// same tree get_indexing_status renders, indented under the row.
		if row.Active {
			for _, treeLine := range BreakdownLines(row.Breakdown) {
				lines = append(lines, "    "+treeLine)
			}
		}
	}
	return strings.Join(lines, "\n")
}

func renderGetJob(entry view.JobEntryView, found bool) string {
	if !found {
		return "Job not found."
	}
	lines := []string{
		"🧾 Job " + entry.ID,
		"📁 Codebase: " + entry.CanonicalPath,
		"⚙️ Operation: " + entry.Operation,
		"🚦 State: " + entry.Surface.StateLabel,
		"🔧 Phase: " + entry.PhaseLabel,
		"📊 Progress: " + entry.Progress.PercentLabel,
	}
	lines = append(lines, renderTimingLines(entry.Timing)...)
	lines = append(lines, renderProgressLines(entry.Progress)...)
	if entry.Surface.ErrorLine != "" {
		lines = append(lines, "🧯 Error: "+entry.Surface.ErrorLine)
	}
	return strings.Join(lines, "\n")
}

// renderProgressLines renders the resolved progress: heading, the shared
// outcome tree (file and chunk breakdown), and the typed classification line.
// The outcome tree is emitted verbatim from renderOutcomeBreakdown, so the
// compact job surfaces and the status templates show a byte-identical tree.
func renderProgressLines(progress view.ProgressSurface) []string {
	lines := make([]string, 0, 8)
	if progress.Heading != "" {
		lines = append(lines, "  "+progress.Heading)
	}
	lines = append(lines, BreakdownLines(progress.Breakdown)...)
	if progress.ScopeLine != "" {
		lines = append(lines, "  "+progress.ScopeLine)
	}
	return lines
}

// outcomePresentation is the glyph and label for one outcome kind. The lead
// glyph types the row before its count: ➕/⏭️ normal, 🗑️ removed, ⏳ pending
// (transient, will retry), 📏 skipped (deliberate policy), ⚠️ error.
type outcomePresentation struct {
	glyph string
	label string
}

// outcomeKindPresentation is the one place a semantic kind maps to its glyph
// and label. Every surface (text, TUI, the wire breakdown rendered back) reads
// it, so the vocabulary cannot diverge.
var outcomeKindPresentation = map[view.OutcomeKind]outcomePresentation{
	view.KindEmbedded:   {glyph: "➕", label: "embedded"},
	view.KindUnchanged:  {glyph: "⏭️", label: "unchanged"},
	view.KindRemoved:    {glyph: "🗑️", label: "removed"},
	view.KindPending:    {glyph: "⏳", label: "pending, not sent yet"},
	view.KindOversize:   {glyph: "📏", label: "skipped, too large"},
	view.KindUnreadable: {glyph: "⚠️", label: "error, unreadable"},
	view.KindAdded:      {glyph: "➕", label: "added"},
	view.KindReused:     {glyph: "♻️", label: "reused"},
}

// BreakdownLines formats the shared file-and-chunk outcome tree into plain
// lines. It is the one formatter for the breakdown, called by every status
// surface (compact job views, status templates, and the TUI), so a tree can
// never read differently across commands. Lines are unindented so the block is
// identical wherever it is placed. The file header renders only when there is
// measured scope; the chunk header only when there is chunk activity.
func BreakdownLines(breakdown view.OutcomeBreakdown) []string {
	lines := make([]string, 0, 8)
	if len(breakdown.FileRows) > 0 {
		lines = append(lines, fmt.Sprintf(
			"📄 %s of %s %s processed",
			formatCountString(breakdown.Processed),
			formatCountString(breakdown.ScopeTotal),
			breakdown.ScopeLabel,
		))
		lines = append(lines, renderOutcomeRows(breakdown.FileRows)...)
	}
	if len(breakdown.ChunkRows) > 0 {
		lines = append(lines, fmt.Sprintf("🧩 %s chunks total", formatCountString(breakdown.ChunksTotal)))
		lines = append(lines, renderOutcomeRows(breakdown.ChunkRows)...)
	}
	return lines
}

// renderOutcomeRows draws each child row under its tree connector: ├─ for every
// row except the last, which gets └─. The glyph and label come from the kind.
func renderOutcomeRows(rows []view.OutcomeRow) []string {
	out := make([]string, 0, len(rows))
	for index, row := range rows {
		connector := "├─"
		if index == len(rows)-1 {
			connector = "└─"
		}
		pres := outcomeKindPresentation[row.Kind()]
		out = append(out, fmt.Sprintf("%s %s %s %s", connector, pres.glyph, formatCountString(row.Count()), pres.label))
	}
	return out
}

func renderTimingLines(timing view.TimingView) []string {
	lines := []string{
		"  Started: " + timing.StartedLabel,
		"  Updated: " + timing.UpdatedLabel,
	}
	if timing.CompletedLabel != "" {
		lines = append(lines, "  Completed: "+timing.CompletedLabel)
	}
	if timing.DurationLabel != "" {
		lines = append(lines, "  "+timing.DurationWord+": "+timing.DurationLabel)
	}
	return lines
}

func renderListJobs(summary view.ListSummary, active []view.JobEntryView, terminal []view.JobEntryView) string {
	if summary.Total == 0 {
		return "No tracked jobs."
	}

	lines := make([]string, 0, 32)
	lines = append(lines, fmt.Sprintf("Tracked jobs: %d total", summary.Total))
	lines = append(lines, fmt.Sprintf("Active: %d queued, %d running, %d canceling", summary.Queued, summary.Running, summary.Canceling))
	lines = append(lines, fmt.Sprintf("Terminal: %d completed, %d failed, %d superseded, %d canceled",
		summary.Completed, summary.Failed, summary.Superseded, summary.Canceled))
	if len(active) == 0 {
		lines = append(lines, "", "No active jobs.")
	} else {
		lines = append(lines, "", "Active jobs:")
		for _, entry := range active {
			lines = append(lines, renderJobListEntry(entry)...)
		}
	}
	const recentTerminalLimit = 8
	if len(terminal) == 0 {
		return strings.Join(lines, "\n")
	}
	lines = append(lines, "")
	if len(terminal) > recentTerminalLimit {
		lines = append(lines, fmt.Sprintf("Recent terminal jobs: showing %d of %d", recentTerminalLimit, len(terminal)))
		for _, entry := range terminal[:recentTerminalLimit] {
			lines = append(lines, renderJobListEntry(entry)...)
		}
		lines = append(lines, "Use `job get JOB_ID` or `--json` for full history.")
		return strings.Join(lines, "\n")
	}
	lines = append(lines, fmt.Sprintf("Terminal jobs: %d", len(terminal)))
	for _, entry := range terminal {
		lines = append(lines, renderJobListEntry(entry)...)
	}
	return strings.Join(lines, "\n")
}

func renderJobListEntry(entry view.JobEntryView) []string {
	lines := []string{fmt.Sprintf("- %s [%s · %s] %s %s",
		entry.ID, entry.Surface.StateLabel, entry.Progress.PercentLabel, entry.Operation, entry.CanonicalPath)}
	lines = append(lines, renderTimingLines(entry.Timing)...)
	lines = append(lines, renderProgressLines(entry.Progress)...)
	if entry.Surface.ErrorLine != "" {
		lines = append(lines, "  Error: "+entry.Surface.ErrorLine)
	}
	return lines
}

// formatCountString is the render-side alias for thousands formatting; the
// value arrives pre-resolved, only the digit grouping happens here.
func formatCountString(value int32) string {
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

func renderDoctor(doctor view.DoctorView) string {
	var body string
	if len(doctor.Diagnostics) == 0 {
		body = "No indexing issues detected."
	} else {
		lines := make([]string, 0, len(doctor.Diagnostics)+1)
		lines = append(lines, "Indexing diagnostics:")
		for _, diagnostic := range doctor.Diagnostics {
			lines = append(lines, "- "+diagnostic)
		}
		body = strings.Join(lines, "\n")
	}
	return body + "\n\n" + renderDroppedSection(doctor)
}

func renderDroppedSection(doctor view.DoctorView) string {
	if len(doctor.Dropped) == 0 {
		return "Dropped codebases (completed index, now untracked, still on disk): none"
	}

	lines := make([]string, 0, len(doctor.Dropped)+1)
	lines = append(lines, fmt.Sprintf("Dropped codebases (completed index, now untracked, still on disk): %d", len(doctor.Dropped)))
	for _, path := range doctor.Dropped {
		lines = append(lines, "- "+path)
	}
	return strings.Join(lines, "\n")
}

func renderSearch(searchView view.SearchView) string {
	// When a run is in flight, the search response carries the same status block
	// get_indexing_status returns, so the caller sees the file and chunk progress
	// inline and does not need a second tool call to learn the index is building.
	status := renderSearchIndexingStatus(searchView)
	resolution := strings.Join(searchView.ResolutionLines, "\n")

	if len(searchView.Results) == 0 {
		noResults := fmt.Sprintf("🔍 No results found for query: %q in codebase '%s'", searchView.Query, searchView.CodebasePath)
		return joinSearchSections(searchView, noResults, status, resolution, false)
	}

	formatted := make([]string, 0, len(searchView.Results))
	for index, result := range searchView.Results {
		language := orDefault(result.Language, "text")
		formatted = append(formatted, fmt.Sprintf(
			"%d. Code snippet (%s) [%s]\n   Location: %s:%d-%d\n   Rank: %d\n   Context:\n```%s\n%s\n```",
			index+1,
			language,
			searchView.CodebaseName,
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
	header := fmt.Sprintf("🔍 Found %d results for query: %q in codebase '%s'", len(searchView.Results), searchView.Query, searchView.CodebasePath)
	body := header + "\n\n" + strings.Join(formatted, "\n\n")
	return joinSearchSections(searchView, body, status, resolution, true)
}

func renderConversationSearch(conversationView view.ConversationSearchView) string {
	if len(conversationView.Results) == 0 {
		return fmt.Sprintf("🔍 No conversation results found for query: %q in collection '%s'", conversationView.Query, conversationView.CollectionID)
	}

	formatted := make([]string, 0, len(conversationView.Results))
	for index, result := range conversationView.Results {
		formatted = append(formatted, fmt.Sprintf(
			"%d. Conversation message [%s]\n   Conversation: %s\n   Message index: %d\n   Role: %s\n   Timestamp Unix: %d\n   Rank: %d\n   Content:\n```\n%s\n```",
			index+1,
			conversationView.CollectionID,
			result.ConversationID,
			result.MessageIndex,
			orDefault(result.Role, "unknown"),
			result.TimestampUnix,
			index+1,
			strings.TrimSpace(truncateContent(result.Content, 5000)),
		))
	}

	header := fmt.Sprintf("🔍 Found %d conversation results for query: %q in collection '%s'", len(conversationView.Results), conversationView.Query, conversationView.CollectionID)
	return header + "\n\n" + strings.Join(formatted, "\n\n")
}

// joinSearchSections appends the identity (resolution), in-progress status, and
// trailing tip sections to a search response body. The resolution and status
// lines show whether or not a run is in flight; the tip only trails an active
// run that has no explicit state note.
func joinSearchSections(searchView view.SearchView, base string, status string, resolution string, hasResults bool) string {
	if status == "" && searchView.StateNote == "" && resolution == "" {
		return base
	}
	sections := []string{base}
	if resolution != "" {
		sections = append(sections, resolution)
	}
	if status != "" {
		sections = append(sections, status)
	}
	if searchView.StateNote != "" {
		sections = append(sections, searchView.StateNote)
	} else if status != "" {
		sections = append(sections, searchStatusTip(searchView, hasResults))
	}
	return strings.Join(sections, "\n\n")
}

// searchStatusTip picks the trailing tip for a search response that has a run
// in flight. A background-sync reconcile keeps the live collection searchable,
// so its results are current; a from-scratch build or rebuild may still be
// filling in, so it keeps the existing "still being indexed" tips.
func searchStatusTip(searchView view.SearchView, hasResults bool) string {
	if searchView.InFlightBackgroundSync {
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
func renderSearchIndexingStatus(searchView view.SearchView) string {
	if !searchView.InFlight {
		return ""
	}
	// The symlink and worktree relation lines are appended once by renderSearch
	// itself, idle or active, so they are not repeated here.
	return renderStatusBody(searchView.InFlightStatus, searchView.InFlightTemplateName)
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

func countWord(word string, count int) string {
	return fmt.Sprintf("%d %s", count, plural(word, count))
}

func plural(word string, count int) string {
	if count == 1 {
		return word
	}
	return word + "s"
}
