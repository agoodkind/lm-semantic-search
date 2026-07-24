//go:build offlinelive

package fixture

import (
	"strconv"
	"strings"
)

// ParseRetryDirective converts a textual retry count into an integer.
func ParseRetryDirective(value string) (int, error) {
	return strconv.Atoi(strings.TrimSpace(value))
}
