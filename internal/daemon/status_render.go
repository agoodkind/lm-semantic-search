package daemon

import (
	"embed"
	"strings"
	"text/template"

	"goodkind.io/lm-semantic-search/internal/view"
)

// statusTemplateFS holds the per-state status layouts. The wording and shape of
// the human-facing status message live in these files so they can be reviewed
// and edited as durable layout, while the Go code only computes the values.
//
//go:embed templates/status/*.md.tmpl
var statusTemplateFS embed.FS

var statusTemplates = template.Must(template.ParseFS(statusTemplateFS, "templates/status/*.md.tmpl"))

// renderStatusTemplate executes one embedded status template by file name and
// trims the trailing newline. The templates are embedded and validated at parse
// time, so an execution error is not expected; if one occurs the title line is
// returned rather than a partial message.
func renderStatusTemplate(name string, statusView view.StatusView) string {
	var buf strings.Builder
	if err := statusTemplates.ExecuteTemplate(&buf, name, statusView); err != nil {
		return "📁 " + statusView.Name
	}
	return strings.TrimRight(buf.String(), "\n")
}
