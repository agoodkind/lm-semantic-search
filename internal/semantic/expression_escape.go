package semantic

import (
	"fmt"
	"strings"
)

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
