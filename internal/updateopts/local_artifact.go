//go:build !lmsrelease

package updateopts

// builtAsReleaseArtifact reports whether the release pipeline compiled this
// binary. A build without the lmsrelease tag is a local build, even when make
// install stamped it at a release tag with the same version string as the
// release.
func builtAsReleaseArtifact() bool {
	return false
}
