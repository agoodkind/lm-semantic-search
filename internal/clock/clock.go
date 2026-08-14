// Package clock centralizes wall-clock access for testable runtime code.
package clock

import "time"

// Now returns the current UTC timestamp.
func Now() time.Time {
	return time.Now().UTC()
}

// Until returns the duration until deadline while preserving its monotonic
// clock reading.
func Until(deadline time.Time) time.Duration {
	return time.Until(deadline)
}
