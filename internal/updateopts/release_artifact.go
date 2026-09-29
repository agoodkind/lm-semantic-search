//go:build lmsrelease

package updateopts

// builtAsReleaseArtifact reports whether the release pipeline compiled this
// binary. The release compile stage adds the lmsrelease build tag through
// GOFLAGS (see the Makefile); make install, go build, and go test omit it.
func builtAsReleaseArtifact() bool {
	return true
}
