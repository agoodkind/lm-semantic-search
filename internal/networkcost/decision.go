package networkcost

import (
	"fmt"
	"strings"
)

// Preference sets the download policy for paths not classified as normal.
type Preference string

// The allow policy permits downloads on every network.
// Warn permits downloads with a warning on expensive, constrained, or unknown
// paths. Defer prevents downloads on those paths.
const (
	PreferenceAllow   Preference = "allow"
	PreferenceWarn    Preference = "warn"
	PreferenceDefer   Preference = "defer"
	DefaultPreference            = PreferenceWarn
)

// Decision represents a download action for the caller to perform.
type Decision string

// The caller starts the download for either download decision.
// The caller reports a warning for DecisionDownloadWithWarning.
// The caller postpones the download for DecisionDefer.
const (
	DecisionDownload            Decision = "download"
	DecisionDownloadWithWarning Decision = "download_with_warning"
	DecisionDefer               Decision = "defer"
)

// ParsePreference trims spaces and ignores letter case.
// ParsePreference rejects values other than allow, warn, and defer.
func ParsePreference(value string) (Preference, error) {
	preference := Preference(strings.ToLower(strings.TrimSpace(value)))
	switch preference {
	case PreferenceAllow, PreferenceWarn, PreferenceDefer:
		return preference, nil
	default:
		return "", fmt.Errorf(
			"unknown model download network policy %q: use %q, %q, or %q",
			value,
			PreferenceAllow,
			PreferenceWarn,
			PreferenceDefer,
		)
	}
}

// Decide returns download for an enabled override, allow, or a normal path.
// Decide returns defer for other paths under defer.
// Decide returns download_with_warning for other paths under warn.
// Unknown is not normal. Invalid preferences behave as warn.
func Decide(
	classification Classification,
	preference Preference,
	override bool,
) Decision {
	if override || preference == PreferenceAllow {
		return DecisionDownload
	}
	if classification == ClassificationNormal {
		return DecisionDownload
	}
	if preference == PreferenceDefer {
		return DecisionDefer
	}
	return DecisionDownloadWithWarning
}
