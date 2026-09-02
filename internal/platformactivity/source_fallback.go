//go:build (!darwin && !linux) || (darwin && !cgo)

package platformactivity

// New returns the unavailable fallback when no native source is available.
func New() Source {
	return NewUnavailable(unavailableSourceReason)
}
