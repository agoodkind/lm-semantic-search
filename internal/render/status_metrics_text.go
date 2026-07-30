package render

import (
	"strconv"
	"strings"
	"time"
	"unicode"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

// StatusMetrics renders one status reply as one record per line, formatted
// "name value unit" with single spaces. This output is parsed, so digits are
// raw and a record with no unit ends after its value.
//
// An activity field is prefixed with its row position, so the flat form keeps
// the rows apart without carrying an index field that protojson would drop at
// zero.
func StatusMetrics(response *pb.GetStatusResponse) string {
	var builder strings.Builder
	writeIdentityLines(&builder, response)
	for _, metric := range response.GetMetrics() {
		writeMetricLine(&builder, metric.GetName(), metric)
	}
	for index, row := range response.GetActivity() {
		prefix := "activity." + strconv.Itoa(index) + "."
		for _, metric := range row.GetMetrics() {
			writeMetricLine(&builder, prefix+metric.GetName(), metric)
		}
	}
	return strings.TrimRight(builder.String(), "\n")
}

// writeIdentityLines emits the records that name the process the reply came
// from, before the counters. Without them a captured snapshot cannot say which
// daemon produced it or when, so two files could not be told apart.
//
// They carry the same escaping as every other value, because a socket path is
// operator-supplied.
func writeIdentityLines(builder *strings.Builder, response *pb.GetStatusResponse) {
	daemon := response.GetDaemon()
	if daemon == nil {
		return
	}
	writeIdentityLine(builder, "version", daemon.GetVersion())
	writeIdentityLine(builder, "commit", daemon.GetCommit())
	writeIdentityLine(builder, "pid", strconv.FormatInt(int64(daemon.GetPid()), 10))
	writeIdentityLine(builder, "socket", daemon.GetSocketPath())
	if readAt := response.GetReadAt(); readAt != nil {
		writeIdentityLine(builder, "read_at", readAt.AsTime().UTC().Format(time.RFC3339Nano))
	}
}

// writeIdentityLine emits one identity record, skipping a value the daemon did
// not report rather than printing an empty field.
func writeIdentityLine(builder *strings.Builder, name string, value string) {
	if value == "" {
		return
	}
	builder.WriteString(name)
	builder.WriteString(" ")
	builder.WriteString(stringValueText(value))
	builder.WriteString("\n")
}

// writeMetricLine emits one record.
func writeMetricLine(builder *strings.Builder, name string, metric *pb.Metric) {
	builder.WriteString(name)
	builder.WriteString(" ")
	builder.WriteString(MetricValueText(metric))
	if unit := metric.GetUnit(); unit != "" {
		builder.WriteString(" ")
		builder.WriteString(unit)
	}
	builder.WriteString("\n")
}

// MetricValueText renders the set oneof member. An unset value prints null,
// which is how every surface says a fact is absent rather than zero or empty.
func MetricValueText(metric *pb.Metric) string {
	switch value := metric.GetValue().(type) {
	case *pb.Metric_IntValue:
		return strconv.FormatInt(value.IntValue, 10)
	case *pb.Metric_DoubleValue:
		return strconv.FormatFloat(value.DoubleValue, 'f', -1, 64)
	case *pb.Metric_BoolValue:
		return strconv.FormatBool(value.BoolValue)
	case *pb.Metric_StringValue:
		return stringValueText(value.StringValue)
	default:
		return "null"
	}
}

// stringValueText renders a string value for a person to read. This form is
// human-facing; a machine consumer reads the JSON form, where every value is a
// typed field rather than a line of text.
//
// A value therefore keeps its spaces and prints as itself. A version string
// reads as `202607270542-fe-6e0a44c 6e0a44c built 2026-07-27T05:42:11Z` rather
// than carrying an escape in place of each space.
//
// A value is quoted only when printing it raw would damage the output. A
// newline would end the line early and leave its tail looking like a separate
// record. An unprintable rune would reach the terminal as a control sequence: a
// codebase path is operator-supplied, so a path holding an escape character
// could clear the screen or move the cursor. An empty string quotes for a third
// reason, so it does not collapse into an absent value, which prints as null.
func stringValueText(value string) string {
	if value == "" || strings.ContainsFunc(value, needsEscaping) {
		return strconv.Quote(value)
	}
	return value
}

// needsEscaping reports whether one rune would damage the line it appears on.
//
// A space is safe and stays, because this output is read rather than parsed.
// Every other whitespace rune is not: a newline ends the line and a tab
// disturbs the column layout. A quote or a backslash is what the escaping
// itself uses. Anything unprintable would reach the terminal as a control
// sequence.
func needsEscaping(candidate rune) bool {
	if candidate == ' ' {
		return false
	}
	if candidate == '"' || candidate == '\\' {
		return true
	}
	return unicode.IsSpace(candidate) || !unicode.IsPrint(candidate)
}
