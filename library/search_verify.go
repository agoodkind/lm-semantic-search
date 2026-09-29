package library

import (
	"crypto/sha256"
	"sync"
)

// verifiedVectors records the vector identities that VerifyStrong confirmed
// for search at one catalog visibility revision. A publication or delete
// increments the visibility revision, and the first search at a higher
// revision empties the record. A search at a lower revision than the record
// neither reads nor writes it. The zero value is empty and ready to use. Each
// recorded identity costs one SHA-256 key in memory, and the record grows to
// at most the number of distinct vectors that searches at one revision score.
type verifiedVectors struct {
	mutex      sync.Mutex
	revision   int64
	identities map[[sha256.Size]byte]struct{}
}

// identityKey returns the SHA-256 of the vector ID, identity digest, and
// checksum of identity. A changed digest or checksum in the catalog produces
// another key.
func identityKey(identity VectorIdentity) [sha256.Size]byte {
	return sha256.Sum256([]byte(identity.ID + "\x00" + identity.IdentityDigest + "\x00" + identity.Checksum))
}

// unverified returns the identities of block that no search verified at
// revision. It empties the record when revision is higher than the recorded
// revision.
func (verified *verifiedVectors) unverified(revision int64, block []VectorIdentity) []VectorIdentity {
	verified.mutex.Lock()
	defer verified.mutex.Unlock()
	if revision > verified.revision || verified.identities == nil {
		verified.revision = revision
		verified.identities = map[[sha256.Size]byte]struct{}{}
	}
	if revision < verified.revision {
		return block
	}
	pending := make([]VectorIdentity, 0, len(block))
	for _, identity := range block {
		if _, found := verified.identities[identityKey(identity)]; !found {
			pending = append(pending, identity)
		}
	}
	return pending
}

// record adds identities that VerifyStrong confirmed at revision. It ignores
// identities verified at a revision other than the recorded one.
func (verified *verifiedVectors) record(revision int64, identities []VectorIdentity) {
	verified.mutex.Lock()
	defer verified.mutex.Unlock()
	if revision != verified.revision || verified.identities == nil {
		return
	}
	for _, identity := range identities {
		verified.identities[identityKey(identity)] = struct{}{}
	}
}
