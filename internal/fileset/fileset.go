// Package fileset defines the shared file eligibility gates for indexing.
package fileset

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

// MaxFileBytes reads INDEX_MAX_FILE_BYTES. An unset or unparseable value falls
// back to 2 MiB. A value of 0 or below disables the size cap.
func MaxFileBytes() int64 {
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

// EligibleByStat reports whether file metadata is eligible for indexing.
func EligibleByStat(info os.FileInfo, maxBytes int64) (bool, SkipReason) {
	if !info.Mode().IsRegular() {
		return false, SkipNotRegular
	}
	if maxBytes > 0 && info.Size() > maxBytes {
		return false, SkipOversize
	}
	return true, SkipKeep
}

// EligibleContent reports whether file bytes are eligible for indexing.
func EligibleContent(data []byte) (bool, SkipReason) {
	if !utf8.Valid(data) {
		return false, SkipNonUTF8
	}
	return true, SkipKeep
}
