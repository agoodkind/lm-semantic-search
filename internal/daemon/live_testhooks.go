//go:build live

package daemon

// SetDerivedPipelineVersionForLiveTest overrides derivedPipelineVersion for the
// live marker harness and returns a restore closure. It exists only under the
// `live` build tag, so a normal production build never compiles this file and
// never exposes a mutator for the pipeline version. The live harness uses it to
// simulate the "pipeline changed, re-examine everything" migration without
// editing the chunking logic.
func SetDerivedPipelineVersionForLiveTest(version string) func() {
	previous := derivedPipelineVersion
	derivedPipelineVersion = version
	return func() { derivedPipelineVersion = previous }
}
