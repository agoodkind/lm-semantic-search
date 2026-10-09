//go:build linux

package platform

// New returns a source that always classifies the network as unknown.
func New() Source {
	return unknownSource{}
}
