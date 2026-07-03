package semantic

import (
	"fmt"
	"strings"
)

// escapeMilvusString escapes one value for a Milvus string literal. Cursor
// conversation ids can contain raw newlines, so control bytes must become
// parser-safe escapes before any relativePath expression reaches Milvus.
func escapeMilvusString(value string) string {
	value = strings.ReplaceAll(value, `\`, `\\`)
	value = strings.ReplaceAll(value, `"`, `\"`)

	var builder strings.Builder
	builder.Grow(len(value))
	for index := range len(value) {
		byteValue := value[index]
		switch byteValue {
		case '\n':
			builder.WriteString(`\n`)
		case '\r':
			builder.WriteString(`\r`)
		case '\t':
			builder.WriteString(`\t`)
		default:
			if byteValue < 0x20 {
				fmt.Fprintf(&builder, `\%03o`, byteValue)
				continue
			}
			builder.WriteByte(byteValue)
		}
	}
	return builder.String()
}
