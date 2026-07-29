package render

import (
	"strconv"
	"strings"
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

// escapedSpace is how a literal space survives inside a value. [strconv.Quote]
// escapes every other whitespace rune already but leaves a space as itself, so
// this is the one substitution that makes a quoted value whitespace-free.
// [strconv.Unquote] reads it back as a space.
const escapedSpace = `\x20`

// stringValueText renders a string value as exactly one whitespace-free field,
// so a record stays `name value unit` and reading a value by field position
// works with awk, cut, and read.
//
// A value carrying whitespace, a quote, or a backslash is quoted with Go
// escaping and its spaces are escaped too, so [strconv.Unquote] recovers the
// original bytes exactly. That also stops a value holding a newline from ending
// its record early and forging a second one. An empty string quotes for a
// different reason: it must not collapse into an absent value, which prints as
// null.
func stringValueText(value string) string {
	if value == "" || strings.ContainsFunc(value, needsEscaping) {
		return strings.ReplaceAll(strconv.Quote(value), " ", escapedSpace)
	}
	return value
}

// needsEscaping reports whether one rune would make a bare value ambiguous.
// Any whitespace would split the value across fields or end the record, and a
// quote or backslash is what the escaping itself uses.
func needsEscaping(candidate rune) bool {
	return unicode.IsSpace(candidate) || candidate == '"' || candidate == '\\'
}
