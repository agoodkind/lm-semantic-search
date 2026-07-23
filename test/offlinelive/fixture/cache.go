//go:build offlinelive

package fixture

import "time"

// CacheEntryExpired reports whether cached data is past its deadline.
func CacheEntryExpired(expiresAt time.Time, observedAt time.Time) bool {
	return !observedAt.Before(expiresAt)
}
