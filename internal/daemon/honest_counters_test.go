package daemon

import "testing"

// TestChunkSplitReportsActualReusedNotSeedPool guards the honest-counter fix:
// the reused figure is the chunks actually served from the pool this run, which
// never exceeds processed, while the seeded pool size is reported only as the
// reuse-vectors-loaded figure. A first build that seeds a large sibling pool but
// processes few chunks must not report reused greater than processed.
func TestChunkSplitReportsActualReusedNotSeedPool(t *testing.T) {
	t.Parallel()

	t.Run("reused stays at the per-file accrual, not the seed pool", func(t *testing.T) {
		state := deltaState{
			plan:         deltaPlan{},
			snapshotPath: "",
			working:      nil,
			source:       nil,
			semantic:     false,
			staging:      true,
			reuse:        nil,
			chunkCounts:  &chunkCounters{processed: 764, reused: 729, embedded: 35, reuseVectorsLoaded: 729},
			seededReuse:  2668,
		}

		processed, reused, embedded, loaded := state.chunkSplit()

		if reused != 729 {
			t.Fatalf("reused = %d, want 729 (actual per-file reuse, not the 2668 seed pool)", reused)
		}
		if reused > processed {
			t.Fatalf("reused %d exceeds processed %d; reused must never exceed processed", reused, processed)
		}
		if embedded != 35 {
			t.Fatalf("embedded = %d, want 35", embedded)
		}
		if loaded != 2668 {
			t.Fatalf("reuse-vectors-loaded = %d, want 2668 (the seed pool size is reported here)", loaded)
		}
	})

	t.Run("nil accumulator reports zero reused with the seed pool only as loaded", func(t *testing.T) {
		state := deltaState{
			plan:         deltaPlan{},
			snapshotPath: "",
			working:      nil,
			source:       nil,
			semantic:     false,
			staging:      true,
			reuse:        nil,
			chunkCounts:  nil,
			seededReuse:  2668,
		}

		processed, reused, embedded, loaded := state.chunkSplit()

		if processed != 0 || reused != 0 || embedded != 0 {
			t.Fatalf("processed/reused/embedded = %d/%d/%d, want 0/0/0 before any per-file work", processed, reused, embedded)
		}
		if loaded != 2668 {
			t.Fatalf("reuse-vectors-loaded = %d, want 2668", loaded)
		}
	})
}
