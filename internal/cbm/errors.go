package cbm

import "errors"

// ErrUnsupportedPlatform reports that the graph engine is unavailable on the
// current GOOS/GOARCH pair.
var ErrUnsupportedPlatform = errors.New("cbm graph engine is unsupported on this platform")
