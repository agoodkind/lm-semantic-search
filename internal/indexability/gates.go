package indexability

import (
	"os"
	"strconv"
	"unicode/utf8"
)

const defaultMaxFileBytes int64 = 2 * 1024 * 1024

// SkipReason names the file-set gate that declined a file.
type SkipReason string

const (
	// SkipKeep marks content that stays in the indexable file set.
	SkipKeep SkipReason = ""
	// SkipNotRegular marks directories, symlinked directories, and special files.
	SkipNotRegular SkipReason = "not-regular"
	// SkipOversize marks regular files above the configured byte cap.
	SkipOversize SkipReason = "oversize"
	// SkipNonUTF8 marks files whose content is not valid UTF-8.
	SkipNonUTF8 SkipReason = "non-utf-8"
)

// maxFileBytes reads INDEX_MAX_FILE_BYTES. An unset or unparseable value falls
// back to 2 MiB. A value of 0 or below disables the size cap. It is unexported
// so the size cap is reachable only through Decide; buildRules bakes the value
// into each codebase's rules.
func maxFileBytes() int64 {
	raw := os.Getenv("INDEX_MAX_FILE_BYTES")
	if raw == "" {
		return defaultMaxFileBytes
	}
	parsed, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return defaultMaxFileBytes
	}
	return parsed
}

// eligibleByStat reports whether file metadata is eligible for indexing. It is
// unexported so the stat-stage gate is reachable only through Decide.
func eligibleByStat(info os.FileInfo, maxBytes int64) (bool, SkipReason) {
	if !info.Mode().IsRegular() {
		return false, SkipNotRegular
	}
	if maxBytes > 0 && info.Size() > maxBytes {
		return false, SkipOversize
	}
	return true, SkipKeep
}

// eligibleContent reports whether file bytes are eligible for indexing. It is
// unexported so the content-stage gate is reachable only through DecideContent.
func eligibleContent(data []byte) (bool, SkipReason) {
	if !utf8.Valid(data) {
		return false, SkipNonUTF8
	}
	return true, SkipKeep
}

// reasonForSkip maps a stat or content gate SkipReason to the Decision Reason
// the resolver returns from Decide and DecideContent.
func reasonForSkip(skip SkipReason) Reason {
	switch skip {
	case SkipNotRegular:
		return ReasonNotRegular
	case SkipOversize:
		return ReasonOversize
	case SkipNonUTF8:
		return ReasonNonUTF8
	case SkipKeep:
		return Keep
	default:
		return Keep
	}
}
