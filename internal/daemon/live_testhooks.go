//go:build live

package daemon

// derivedPipelineVersion is the version knob bumped whenever conversation tool or
// thinking chunking changes. Post-PR1 the presence-based backfill classifier no
// longer consults it for a forcing decision, so nothing in the production build
// reads it today; it lives under the live tag alongside its only reader until the
// force path (a later PR) reintroduces a production consumer. It is a var rather
// than a const so SetDerivedPipelineVersionForLiveTest can reassign it.
var derivedPipelineVersion = "1"

// SetDerivedPipelineVersionForLiveTest overrides derivedPipelineVersion for the
// live backfill harness and returns a restore closure. It exists only under the
// `live` build tag, so a normal production build never compiles this file and
// never exposes a mutator for the pipeline version. The live harness uses it to
// simulate the "pipeline changed, re-examine everything" migration without
// editing the chunking logic.
func SetDerivedPipelineVersionForLiveTest(version string) func() {
	previous := derivedPipelineVersion
	derivedPipelineVersion = version
	return func() { derivedPipelineVersion = previous }
}
