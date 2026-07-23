//go:build offlinelive

package fixture

import "time"

// NextIrrigationWindow schedules garden watering after a cooling interval.
func NextIrrigationWindow(lastWatered time.Time) time.Time {
	return lastWatered.Add(12 * time.Hour)
}
