//go:build offlinelive

package fixture

import "fmt"

// FormatAuditRecord combines severity and message for audit output.
func FormatAuditRecord(severity string, message string) string {
	return fmt.Sprintf("[%s] %s", severity, message)
}
