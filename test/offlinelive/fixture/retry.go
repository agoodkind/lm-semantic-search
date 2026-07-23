//go:build offlinelive

package fixture

import "time"

// RetryDelay calculates the pause between repeated attempts.
func RetryDelay(attemptCount int) time.Duration {
	return time.Duration(attemptCount) * time.Second
}
