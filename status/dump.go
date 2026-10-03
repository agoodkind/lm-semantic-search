package status

import (
	"strconv"
	"strings"
	"unicode"
)

// Dump renders one snapshot as one record per line, formatted "name value unit"
// with single spaces. This output is parsed, so digits are raw and a record
// with no unit ends after its value.
//
// The notices come first, then the identity records, then the counters. An
// activity field is prefixed with its row position. The prefix keeps the rows
// apart without an index field.
func Dump(snapshot Snapshot) string {
	var builder strings.Builder
	for _, notice := range snapshot.Notices {
		builder.WriteString(notice)
		builder.WriteString("\n")
	}
	writeIdentityLines(&builder, snapshot.Identity)
	for _, field := range snapshot.Counters {
		writeFieldLine(&builder, field.Name, field)
	}
	for index, row := range snapshot.Activity {
		prefix := "activity." + strconv.Itoa(index) + "."
		for _, field := range row {
			writeFieldLine(&builder, prefix+field.Name, field)
		}
	}
	return strings.TrimRight(builder.String(), "\n")
}

// writeIdentityLines emits the records that identify the process the snapshot
// came from, before the counters. A captured snapshot without them cannot say
// which process produced it or when, so two files could not be told apart.
//
// They carry the same escaping as every other value, because a socket path is
// operator-supplied.
func writeIdentityLines(builder *strings.Builder, identity []Field) {
	for _, field := range identity {
		writeIdentityLine(builder, field.Name, field.Value.text)
	}
}

// writeIdentityLine emits one identity record, skipping a value the program did
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

// writeFieldLine emits one record.
func writeFieldLine(builder *strings.Builder, name string, field Field) {
	builder.WriteString(name)
	builder.WriteString(" ")
	builder.WriteString(valueText(field.Value))
	if field.Unit != "" {
		builder.WriteString(" ")
		builder.WriteString(field.Unit)
	}
	builder.WriteString("\n")
}

// valueText renders the set member of a Value. An absent value prints null,
// which is how every surface says a fact is missing rather than zero or empty.
func valueText(value Value) string {
	switch value.kind {
	case kindInteger:
		return strconv.FormatInt(value.integer, 10)
	case kindNumber:
		return strconv.FormatFloat(value.number, 'f', -1, 64)
	case kindFlag:
		return strconv.FormatBool(value.flag)
	case kindText:
		return stringValueText(value.text)
	case kindAbsent:
		return "null"
	default:
		return "null"
	}
}

// stringValueText renders a string value for a person to read. This form is
// human-facing; a machine consumer reads the JSON form, where every value is a
// typed field rather than a line of text.
//
// A value keeps its spaces and prints as itself. A version string reads as
// `202607270542-fe-6e0a44c 6e0a44c built 2026-07-27T05:42:11Z` rather than
// carrying an escape in place of each space.
//
// A value is quoted only when printing it raw would damage the output. A
// newline would end the line early and leave its tail looking like a separate
// record. An unprintable rune would arrive at the terminal as a control
// sequence: a codebase path is operator-supplied, and a path with an escape
// character could clear the screen or move the cursor. An empty string quotes
// for a third reason: unquoted, it would look like an absent value, which
// prints as null.
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
// itself uses. Anything unprintable would arrive at the terminal as a control
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
