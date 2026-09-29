package library

import (
	"context"
	"crypto/sha256"
	"log/slog"
	"sync"
)

// maxVerifiedVectors bounds the identities that one verification record
// keeps. Each identity is one SHA-256 key in a map, about 50 bytes with map
// overhead. The bound limits the record to about 50 MiB. Production has about
// 527,125 distinct inputs.
const maxVerifiedVectors = 1 << 20

// verifiedVectors records the vector identities that VerifyStrong confirmed
// for search at one catalog visibility revision. A publication or delete
// increments the visibility revision, and the first search at a higher
// revision empties the record. A search at a lower revision than the record
// neither reads nor writes it. The record keeps at most limit identities, or
// maxVerifiedVectors when limit is zero. At the bound it stops adding
// identities, and every search verifies each unrecorded identity again. The
// zero value is empty and ready to use.
type verifiedVectors struct {
	mutex      sync.Mutex
	revision   int64
	identities map[[sha256.Size]byte]struct{}
	limit      int
	full       bool
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
		verified.full = false
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

// record adds identities that VerifyStrong confirmed at revision while the
// record is below its bound. It ignores identities verified at a revision
// other than the recorded one. It logs a warning once per revision when the
// record becomes full.
func (verified *verifiedVectors) record(ctx context.Context, revision int64, identities []VectorIdentity) {
	verified.mutex.Lock()
	defer verified.mutex.Unlock()
	if revision != verified.revision || verified.identities == nil {
		return
	}
	limit := verified.limit
	if limit == 0 {
		limit = maxVerifiedVectors
	}
	for _, identity := range identities {
		if len(verified.identities) >= limit {
			break
		}
		verified.identities[identityKey(identity)] = struct{}{}
	}
	if len(verified.identities) >= limit && !verified.full {
		verified.full = true
		slog.WarnContext(ctx, "search verification record is full; unrecorded vectors are verified on every search",
			"revision", revision, "identities", len(verified.identities))
	}
}
