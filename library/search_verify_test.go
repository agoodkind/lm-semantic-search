package library

import (
	"context"
	"fmt"
	"slices"
	"testing"
)

func verifiedTestIdentities(count int) []VectorIdentity {
	identities := make([]VectorIdentity, 0, count)
	for index := range count {
		identities = append(identities, VectorIdentity{
			ID:             fmt.Sprintf("v%d", index),
			IdentityDigest: fmt.Sprintf("digest%d", index),
			Checksum:       fmt.Sprintf("checksum%d", index),
		})
	}
	return identities
}

// TestVerifiedVectorsStopsRecordingAtItsBound records 5 verified identities
// in a record bounded at 3. The 2 identities over the bound stay unverified
// for every later search at the revision, and a higher revision empties the
// record.
func TestVerifiedVectorsStopsRecordingAtItsBound(t *testing.T) {
	ctx := context.Background()
	identities := verifiedTestIdentities(5)
	verified := verifiedVectors{limit: 3}
	if pending := verified.unverified(1, identities); !slices.Equal(pending, identities) {
		t.Fatalf("empty record returned %d unverified identities, want all 5", len(pending))
	}
	verified.record(ctx, 1, identities)
	for search := range 2 {
		if pending := verified.unverified(1, identities); !slices.Equal(pending, identities[3:]) {
			t.Fatalf("search %d at the full revision returned %v unverified, want the 2 identities over the bound", search, pending)
		}
	}
	if pending := verified.unverified(2, identities); !slices.Equal(pending, identities) {
		t.Fatalf("higher revision returned %d unverified identities, want all 5", len(pending))
	}
	verified.record(ctx, 2, identities[:1])
	if pending := verified.unverified(2, identities); !slices.Equal(pending, identities[1:]) {
		t.Fatalf("after one record at the higher revision %v stay unverified, want 4 identities", pending)
	}
}
