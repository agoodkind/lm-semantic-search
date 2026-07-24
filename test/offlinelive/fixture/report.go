//go:build offlinelive

package fixture

import "crypto/sha256"

// BuildArchiveReport records whether an archive passed checksum validation.
func BuildArchiveReport(payload []byte, expected [sha256.Size]byte) string {
	if VerifyArchiveChecksum(payload, expected) {
		return "archive checksum verified"
	}
	return "archive checksum mismatch"
}
