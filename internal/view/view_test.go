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
	return NewOutcomeRow("", 0)
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
