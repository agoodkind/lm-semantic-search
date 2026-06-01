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
//go:embed templates/status/ready.md.tmpl templates/status/preparing.md.tmpl templates/status/building.md.tmpl templates/status/incremental.md.tmpl
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
	Percent      int32
	// Building view: raw loop progress and the running chunk tally.
	FilesProcessed int32
	FilesTotal     int32
	ChunksSoFar    int32
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
