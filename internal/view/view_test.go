package view

import "testing"

func findTestRow(t *testing.T, rows []OutcomeRow, kind OutcomeKind) OutcomeRow {
	t.Helper()
	for _, row := range rows {
		if row.Kind() == kind {
			return row
		}
	}
	t.Fatalf("row %q not found in %+v", kind, rows)
	return OutcomeRow{}
}

func hasTestRow(rows []OutcomeRow, kind OutcomeKind) bool {
	for _, row := range rows {
		if row.Kind() == kind {
			return true
		}
	}
	return false
}

// The view package must stay pure data: this test locks the absence of an
// internal/model dependency at the package level. The render wall depends on
// view being importable without model.
func TestViewHasNoModelDependency(t *testing.T) {
	// Compile-time property: if any view type referenced internal/model this
	// package would import it and the import-list assertion in the render
	// package tests would fail. This test exists to document the invariant.
	_ = ProgressSurface{}
	_ = JobEntryView{}
	_ = GetIndexView{}
}

func TestResolveBreakdownFirstBuildReusePresentation(t *testing.T) {
	t.Parallel()

	t.Run("cold first build keeps full build label and omits reused row", func(t *testing.T) {
		t.Parallel()
		breakdown := ResolveBreakdown(ProgressCounts{
			RunMode:        string(RunModeFirstBuild),
			FilesTotal:     100,
			FilesProcessed: 10,
			FilesEmbedded:  10,
			ChunksEmbedded: 55,
		})

		if breakdown.ScopeLabel != "files (full build)" {
			t.Fatalf("scope label = %q, want %q", breakdown.ScopeLabel, "files (full build)")
		}
		if hasTestRow(breakdown.ChunkRows, KindReused) {
			t.Fatalf("cold first build should omit reused row: %+v", breakdown.ChunkRows)
		}
	})

	t.Run("seeded first build names reuse and shows reused row", func(t *testing.T) {
		t.Parallel()
		breakdown := ResolveBreakdown(ProgressCounts{
			RunMode:            string(RunModeFirstBuild),
			FilesTotal:         100,
			FilesProcessed:     1,
			FilesEmbedded:      1,
			ChunksProcessed:    678,
			ChunksReused:       623,
			ChunksEmbedded:     55,
			ReuseVectorsLoaded: 2316,
		})

		if breakdown.ScopeLabel != "files (first build, reusing prior vectors)" {
			t.Fatalf("scope label = %q, want seeded first build label", breakdown.ScopeLabel)
		}
		if reused := findTestRow(t, breakdown.ChunkRows, KindReused).Count(); reused != 623 {
			t.Fatalf("reused chunks = %d, want 623", reused)
		}
	})

	t.Run("first build with served reuse still shows reused row", func(t *testing.T) {
		t.Parallel()
		breakdown := ResolveBreakdown(ProgressCounts{
			RunMode:         string(RunModeFirstBuild),
			FilesTotal:      100,
			FilesProcessed:  1,
			FilesEmbedded:   1,
			ChunksProcessed: 678,
			ChunksReused:    623,
			ChunksEmbedded:  55,
		})

		if breakdown.ScopeLabel != "files (full build)" {
			t.Fatalf("scope label = %q, want %q", breakdown.ScopeLabel, "files (full build)")
		}
		if reused := findTestRow(t, breakdown.ChunkRows, KindReused).Count(); reused != 623 {
			t.Fatalf("reused chunks = %d, want 623", reused)
		}
	})
}

// TestResolveOverallPercentCountsReusedWork proves the headline percent reads by
// corpus completion (reused plus embedded over a known corpus chunk total) and
// only ever raises the run's file-cursor percent, never lowers it.
func TestResolveOverallPercentCountsReusedWork(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name       string
		counts     ProgressCounts
		runPercent float64
		want       float64
	}{
		{
			// The core fix: a superseded-and-restarted run reset its file cursor
			// low, but reused work covers most of the corpus, so it reads mostly
			// done rather than near-zero.
			name: "restart with large reuse reads mostly done",
			counts: ProgressCounts{
				ChunksReused: 37800, ChunksEmbedded: 200, ChunksTotal: 40000,
			},
			runPercent: 2.3,
			want:       95.0,
		},
		{
			// A reuse-heavy reexamine embeds little but reused work is most of the
			// corpus, so the reused work counts as done in the top line.
			name: "reuse heavy reexamine reads by reused work",
			counts: ProgressCounts{
				ChunksReused: 30000, ChunksEmbedded: 1000, ChunksTotal: 40000,
			},
			runPercent: 5.0,
			want:       77.5,
		},
		{
			// A cold first build has no corpus headroom: ChunksTotal is the running
			// sum, so the file-cursor percent stands.
			name: "cold first build keeps file cursor percent",
			counts: ProgressCounts{
				ChunksEmbedded: 71, ChunksTotal: 0,
			},
			runPercent: 42.0,
			want:       42.0,
		},
		{
			// A seeded first build has reuse but ChunksTotal still only equals the
			// running sum, so it is not a corpus denominator and the percent holds.
			name: "seeded first build keeps file cursor percent",
			counts: ProgressCounts{
				ChunksProcessed: 678, ChunksReused: 623, ChunksEmbedded: 55, ChunksTotal: 678,
			},
			runPercent: 10.0,
			want:       10.0,
		},
		{
			// A genuine mid-build with a real corpus total but little done keeps the
			// higher file-cursor reading; the corpus figure never lowers it.
			name: "corpus figure never lowers a higher file percent",
			counts: ProgressCounts{
				ChunksGenerated: 1043, ChunksTotal: 57240,
			},
			runPercent: 37.0,
			want:       37.0,
		},
		{
			name:       "no chunk activity leaves the run percent untouched",
			counts:     ProgressCounts{ChunksTotal: 0},
			runPercent: 49.8,
			want:       49.8,
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			got := ResolveOverallPercent(testCase.counts, testCase.runPercent)
			if got != testCase.want {
				t.Fatalf("ResolveOverallPercent = %.4f, want %.4f", got, testCase.want)
			}
		})
	}
}

func TestResolveBreakdownChangedModeKeepsReusePresentation(t *testing.T) {
	t.Parallel()
	breakdown := ResolveBreakdown(ProgressCounts{
		RunMode:        string(RunModeChanged),
		FilesTotal:     5,
		FilesProcessed: 1,
		FilesAdded:     5,
		FilesEmbedded:  1,
		ChunksEmbedded: 7,
	})

	if breakdown.ScopeLabel != "changed files" {
		t.Fatalf("scope label = %q, want %q", breakdown.ScopeLabel, "changed files")
	}
	if reused := findTestRow(t, breakdown.ChunkRows, KindReused).Count(); reused != 0 {
		t.Fatalf("reused chunks = %d, want 0 for unchanged changed-mode behavior", reused)
	}
}

// TestResolvePhasePercentReadsTheCurrentScope proves the phase percent answers
// how far the run is through the work it is doing now, and that it prefers the
// file cursor over the embed-batch cursor whenever a file scope exists. The
// batch denominator belongs to the item being embedded and resets between
// items, so reading it while files are in flight would saw-tooth.
func TestResolvePhasePercentReadsTheCurrentScope(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name   string
		counts ProgressCounts
		want   float64
	}{
		{
			name:   "file scope reads by the file cursor",
			counts: ProgressCounts{FilesTotal: 452, FilesProcessed: 226},
			want:   50.0,
		},
		{
			// The file cursor wins even while an embed loop reports batches, so a
			// multi-item run climbs once rather than resetting per item.
			name: "file scope wins over the embed batch cursor",
			counts: ProgressCounts{
				FilesTotal: 100, FilesProcessed: 25,
				EmbeddingBatchesTotal: 10, EmbeddingBatchesCompleted: 9,
			},
			want: 25.0,
		},
		{
			// One large item with no file scope still shows movement across its
			// embed batches instead of sitting at zero until it finishes.
			name:   "batch scope carries a run with no file scope",
			counts: ProgressCounts{EmbeddingBatchesTotal: 16, EmbeddingBatchesCompleted: 12},
			want:   75.0,
		},
		{
			name:   "a finished file scope reads fully done",
			counts: ProgressCounts{FilesTotal: 58, FilesProcessed: 58},
			want:   100.0,
		},
		{
			// A cursor can pass its denominator when the scope was measured before
			// the run counted extra items; a percent above 100 reads as a defect.
			name:   "a cursor past its denominator clamps to fully done",
			counts: ProgressCounts{FilesTotal: 10, FilesProcessed: 14},
			want:   100.0,
		},
		{
			name:   "an unmeasured scope claims nothing",
			counts: ProgressCounts{ChunksEmbedded: 400, ChunksTotal: 40000},
			want:   0,
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			got := ResolvePhasePercent(testCase.counts)
			if got != testCase.want {
				t.Fatalf("ResolvePhasePercent = %.4f, want %.4f", got, testCase.want)
			}
		})
	}
}
