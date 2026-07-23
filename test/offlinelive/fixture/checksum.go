//go:build offlinelive

package fixture

import "crypto/sha256"

// VerifyArchiveChecksum validates an offline archive checksum before storage.
func VerifyArchiveChecksum(payload []byte, expected [sha256.Size]byte) bool {
	return sha256.Sum256(payload) == expected
}
