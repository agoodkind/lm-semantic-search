package status

import (
	"strconv"
	"strings"
	"unicode"
)

// Dump renders one snapshot as one record per line, formatted "name value unit"
// with single spaces. Digits are raw, and a record with no unit ends after its
// value.
//
// The notices come first, then the identity records, then the counters. An
// activity field name starts with its row position.
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

// writeIdentityLines writes the records that identify the process of the
// snapshot, before the counters. It escapes each value like every other value,
// because a socket path is operator-supplied.
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

// valueText renders the set member of a Value. An absent value prints null.
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

// stringValueText renders a string value for a person to read. A machine
// consumer reads the JSON form.
//
// A value prints as itself, with its spaces. Three kinds of value print quoted.
// A value with a newline would end the line early. A value with an unprintable
// rune would send a control sequence to the terminal, and a codebase path is
// operator-supplied. An empty string would look like an absent value, which
// prints as null.
func stringValueText(value string) string {
	if value == "" || strings.ContainsFunc(value, needsEscaping) {
		return strconv.Quote(value)
	}
	return value
}

// needsEscaping reports whether one rune forces the value to print quoted. A
// space does not. Every other whitespace rune does: a newline ends the line and
// a tab shifts the columns. A quote and a backslash do, because the quoted form
// uses them. An unprintable rune does, because the terminal would read it as a
// control sequence.
func needsEscaping(candidate rune) bool {
	if candidate == ' ' {
		return false
	}
	if candidate == '"' || candidate == '\\' {
		return true
	}
	return unicode.IsSpace(candidate) || !unicode.IsPrint(candidate)
}
