package render

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

// shellQuote wraps a value in single quotes so a path containing a space or a
// shell metacharacter survives being pasted onto a command line. An embedded
// single quote is closed, escaped outside the quotes, and reopened.
func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\"'\"'") + "'"
}

// statusTemplateData is the data passed to a status template. It embeds the
// resolved StatusView (so {{ .Name }} and friends are promoted) and adds two
// pre-formatted values: BreakdownBlock, the shared outcome tree, and
// QuotedPath, the shell-quoted codebase path used by the copy-paste wait
// command. Formatting both in the render layer and injecting them as finished
// strings keeps the templates free of row logic and quoting rules, so the
// status tree stays identical to the compact job tree.
type statusTemplateData struct {
	view.StatusView
	BreakdownBlock string
	QuotedPath     string
}

// renderStatusTemplate executes one embedded status template by file name and
// trims the trailing newline. The templates are embedded and validated at parse
// time, so an execution error is not expected; if one occurs the title line is
// returned rather than a partial message.
func renderStatusTemplate(name string, data statusTemplateData) string {
	var buf strings.Builder
	if err := statusTemplates.ExecuteTemplate(&buf, name, data); err != nil {
		return "📁 " + data.Name
	}
	return strings.TrimRight(buf.String(), "\n")
}
