package updateopts

import (
	"strings"

	"golang.org/x/mod/semver"
	"goodkind.io/gklog/version"
)

// RunningReleaseTag returns the running binary's release tag, and false when
// the version stamp marks a dev or locally built binary, which has no release
// to match.
func RunningReleaseTag() (string, bool) {
	if isLocalBuild(version.Version, version.Dirty == "true") {
		return "", false
	}
	return strings.TrimSpace(version.Version), true
}

// AutomaticUpdatesEnabled reports whether the daemon may run the scheduled
// update check and apply. It requires the release pipeline's build tag, because
// make install at a release tag stamps the same version string as the release.
func AutomaticUpdatesEnabled() bool {
	return builtAsReleaseArtifact() && !isLocalBuild(version.Version, version.Dirty == "true")
}

// isLocalBuild reports whether a version stamp marks a dev or locally built
// binary. A clean build at a release tag passes this check, so callers that
// must exclude every local build also check builtAsReleaseArtifact.
func isLocalBuild(version string, dirty bool) bool {
	if dirty {
		return true
	}
	trimmed := strings.TrimSpace(version)
	if trimmed == "" || trimmed == "dev" || trimmed == "unknown" {
		return true
	}
	if hasGitDescribeSuffix(trimmed) {
		return true
	}
	return !semver.IsValid(trimmed) && !isRollingRelease(trimmed)
}

func isRollingRelease(version string) bool {
	fields := strings.Split(version, "-")
	if len(fields) != 3 {
		return false
	}
	timestamp, sequence, commit := fields[0], fields[1], fields[2]
	if len(timestamp) != 12 || !isDigits(timestamp) {
		return false
	}
	if !isAlphaNumeric(sequence) {
		return false
	}
	return len(commit) >= 7 && isHex(commit)
}

func hasGitDescribeSuffix(version string) bool {
	lastDash := strings.LastIndex(version, "-")
	if lastDash < 0 {
		return false
	}
	objectName := version[lastDash+1:]
	if len(objectName) < 2 || objectName[0] != 'g' || !isHex(objectName[1:]) {
		return false
	}
	countField := version[:lastDash]
	countDash := strings.LastIndex(countField, "-")
	if countDash <= 0 {
		return false
	}
	return isDigits(countField[countDash+1:])
}

func isAlphaNumeric(value string) bool {
	if value == "" {
		return false
	}
	for _, character := range value {
		isDigit := character >= '0' && character <= '9'
		isLower := character >= 'a' && character <= 'z'
		isUpper := character >= 'A' && character <= 'Z'
		if !isDigit && !isLower && !isUpper {
			return false
		}
	}
	return true
}

func isDigits(value string) bool {
	if value == "" {
		return false
	}
	for _, character := range value {
		if character < '0' || character > '9' {
			return false
		}
	}
	return true
}

func isHex(value string) bool {
	if value == "" {
		return false
	}
	for _, character := range value {
		isDigit := character >= '0' && character <= '9'
		isLower := character >= 'a' && character <= 'f'
		isUpper := character >= 'A' && character <= 'F'
		if !isDigit && !isLower && !isUpper {
			return false
		}
	}
	return true
}
