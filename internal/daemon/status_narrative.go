package daemon

import (
	"strings"

	"goodkind.io/lm-semantic-search/internal/render"
	"goodkind.io/lm-semantic-search/internal/view"
)

// resolveStatusNarrative builds the display-ready body lines for the non-template
// codebase states (failed, missing, stale, quarantined) at the daemon boundary.
// The render layer only joins these lines, so all status prose and its fallback
// wording live here behind the view wall rather than in render. Template states
// (preparing, building, incremental, ready, discovered, waiting) return an empty
// narrative and are rendered from their templates.
func resolveStatusNarrative(display displayStatus, canonicalPath string, failure view.FailureSurface, quarantine view.QuarantineSurface, statusView view.StatusView) view.StatusNarrative {
	switch display {
	case displayFailed:
		return view.StatusNarrative{Kind: string(displayFailed), Lines: failedNarrativeLines(canonicalPath, failure)}
	case displayMissing:
		return view.StatusNarrative{Kind: string(displayMissing), Lines: missingNarrativeLines(canonicalPath)}
	case displayStale:
		return view.StatusNarrative{Kind: string(displayStale), Lines: staleNarrativeLines(canonicalPath, failure)}
	case displayQuarantined:
		return view.StatusNarrative{Kind: string(displayQuarantined), Lines: quarantinedNarrativeLines(canonicalPath, quarantine, statusView)}
	default:
		return view.StatusNarrative{Kind: "", Lines: nil}
	}
}

// missingNarrativeLines reads as a current condition, not a failure: the source
// directory is gone, so the index is held until the directory returns or the
// caller drops it.
func missingNarrativeLines(canonicalPath string) []string {
	return []string{
		"🚫 Codebase '" + canonicalPath + "' source directory is missing.",
		"💡 Re-create the directory to resume indexing, or call clear_index to drop the index.",
	}
}

// failedNarrativeLines reads as past tense so callers do not mistake an old
// failure record for a live one. When the failure carries correlation ids it
// appends a diagnostics line so the operator can grep the daemon log.
func failedNarrativeLines(canonicalPath string, failure view.FailureSurface) []string {
	if !failure.HasFailure {
		return []string{"❌ Codebase '" + canonicalPath + "' could not be indexed. Re-run index_codebase to retry."}
	}
	lines := []string{
		"❌ Codebase '" + canonicalPath + "' could not be indexed.",
		"🚧 " + narrativeOrDefault(failure.Message, "the index could not be built"),
		"💡 Re-run index_codebase; if it keeps failing, check the daemon log via the failed-job reference below.",
	}
	if diagnostics := failureDiagnosticsLine(failure); diagnostics != "" {
		lines = append(lines, diagnostics)
	}
	return lines
}

func staleNarrativeLines(canonicalPath string, failure view.FailureSurface) []string {
	if !failure.HasFailure {
		return []string{
			"⚠️ Codebase '" + canonicalPath + "' is stale because its semantic collection is missing.",
			"💡 The daemon will rebuild it automatically on the next background repair pass.",
		}
	}
	lines := []string{
		"⚠️ Codebase '" + canonicalPath + "' is stale since " + failure.FailedAtLabel + ".",
		"🚨 Repair detail: " + narrativeOrDefault(failure.Message, "semantic collection is missing"),
		"💡 The daemon will retry automatic rebuild while the codebase remains stale.",
	}
	if diagnostics := failureDiagnosticsLine(failure); diagnostics != "" {
		lines = append(lines, diagnostics)
	}
	return lines
}

func quarantinedNarrativeLines(canonicalPath string, quarantine view.QuarantineSurface, statusView view.StatusView) []string {
	lines := []string{
		"⚠️ Codebase '" + canonicalPath + "' is quarantined after a suspicious large disappearance.",
		"🔒 Search continues to serve the last known-good index while destructive sync is paused.",
	}
	if statusView.HasStats {
		lines = append(lines, "📊 Last known good index: "+render.FormatCount(statusView.Files)+" files, "+render.FormatCount(statusView.Chunks)+" chunks")
	}
	if quarantine.HasQuarantine {
		lines = append(lines, "🧾 Last signal: "+render.FormatCount(quarantine.MissingCount)+" of "+render.FormatCount(quarantine.TotalCount)+" tracked files in a "+narrativeOrDefault(quarantine.Trigger, "suspicious")+" observation")
		lines = append(lines, "🕐 First observed: "+narrativeOrDefault(quarantine.FirstObservedLabel, "unknown")+" · Last observed: "+narrativeOrDefault(quarantine.LastObservedLabel, "unknown")+" · Observations: "+render.FormatCount(quarantine.ObservationCount))
		if strings.TrimSpace(quarantine.Reason) != "" {
			lines = append(lines, "🚧 "+quarantine.Reason)
		}
	}
	lines = append(lines, "💡 The daemon will re-check automatically and only apply deletes after later full-scan corroboration.")
	return lines
}

// failureDiagnosticsLine names the failed job and its trace id, or empty when
// neither is recorded. It leads with the job so it reads as the past failure's
// reference, leaving the envelope header as the only request "trace_id=" line.
// It formats the resolved failure view, never the raw failure record.
func failureDiagnosticsLine(failure view.FailureSurface) string {
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
	return "🔎 " + label
}

func narrativeOrDefault(value string, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}
