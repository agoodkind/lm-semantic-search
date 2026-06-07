package daemon

import (
	"embed"
	"strings"
	"text/template"
)

// statusTemplateFS holds the per-state status layouts. The wording and shape of
// the human-facing status message live in these files so they can be reviewed
// and edited as durable layout, while the Go code only computes the values.
//
//go:embed templates/status/*.md.tmpl
var statusTemplateFS embed.FS

var statusTemplates = template.Must(template.ParseFS(statusTemplateFS, "templates/status/*.md.tmpl"))

// statusView is the data a status template renders. Every field is always set;
// each state's template reads only the subset it needs.
type statusView struct {
	Name         string
	HasStats     bool
	Files        int32
	Chunks       int32
	SkippedLine  string
	PrepareLabel string
	// WaitLabel names the dependency an incomplete codebase is waiting on during a
	// hard pipeline outage; the banner carries the cause, so this stays generic.
	WaitLabel string
	Percent   int32
	// Heading names what started the in-progress run (a first build, a forced
	// reindex, or a changed-files sync), so the building view leads with the
	// trigger rather than the internal job path.
	Heading string
	// Building view: raw loop progress.
	FilesProcessed int32
	FilesTotal     int32
	// Building view chunk tree: total = reused + embedded this run, so the
	// reuse-vs-redo split is visible on screen.
	ChunksReused          int32
	ChunksEmbeddedThisRun int32
	// Incremental view: the diff breakdown against the whole codebase.
	FilesInCodebase        int32
	FilesChanged           int32
	FilesUnchanged         int32
	FilesProcessedChanged  int32
	FilesReEmbedded        int32
	FilesRemoved           int32
	FilesSkippedOversize   int32
	FilesSkippedUnreadable int32
	ChunksAdded            int32
	ChunksTotal            int32
	UpdatedAt              string
	// SyncNote, when set, appends a one-line background-sync note to the ready
	// view so a searchable incremental sync reads as ready rather than busy.
	SyncNote string
}

// renderStatusTemplate executes one embedded status template by file name and
// trims the trailing newline. The templates are embedded and validated at parse
// time, so an execution error is not expected; if one occurs the title line is
// returned rather than a partial message.
func renderStatusTemplate(name string, view statusView) string {
	var buf strings.Builder
	if err := statusTemplates.ExecuteTemplate(&buf, name, view); err != nil {
		return "📁 " + view.Name
	}
	return strings.TrimRight(buf.String(), "\n")
}
