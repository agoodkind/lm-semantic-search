//go:build offlinelive

package fixture

import "fmt"

// FormatGreenhouseTemperature renders a greenhouse sensor reading.
func FormatGreenhouseTemperature(celsius float64) string {
	return fmt.Sprintf("%.1f C", celsius)
}
