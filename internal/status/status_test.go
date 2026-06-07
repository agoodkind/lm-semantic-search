package status

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

// Every base display is reachable from a healthy snapshot, and the table never
// yields not_indexed for a tracked codebase.
func TestResolveDisplayBase(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		in   Inputs
		want Display
	}{
		{"active reconcile", Inputs{HasActiveJob: true, BackgroundSyncReconcile: true}, DisplayIndexed},
		{"active scoped", Inputs{HasActiveJob: true, JobScopeKnown: true}, DisplayIndexing},
		{"active preparing", Inputs{HasActiveJob: true}, DisplayPreparing},
		{"indexed", Inputs{Status: model.CodebaseStatusIndexed}, DisplayIndexed},
		{"stale", Inputs{Status: model.CodebaseStatusStale}, DisplayStale},
		{"failed", Inputs{Status: model.CodebaseStatusFailed}, DisplayFailed},
		{"missing", Inputs{Status: model.CodebaseStatusMissing}, DisplayMissing},
		{"interrupted indexing", Inputs{Status: model.CodebaseStatusIndexing}, DisplayPreparing},
		{"not indexed", Inputs{Status: model.CodebaseStatusNotIndexed}, DisplayPreparing},
	}
	for _, testCase := range cases {
		got := ResolveDisplay(testCase.in)
		if got != testCase.want {
			t.Errorf("%s: ResolveDisplay = %q, want %q", testCase.name, got, testCase.want)
		}
		if string(got) == string(model.CodebaseStatusNotIndexed) {
			t.Errorf("%s: a tracked codebase must never present as not_indexed", testCase.name)
		}
	}
}

// The degraded fold turns a not-embedding codebase into waiting (preparing and
// indexed), while a live scoped job stays indexing and a local terminal state is
// untouched. This is the rule that keeps a surface from reading "ready" while a
// search would fail.
func TestResolveDisplayDegradedFold(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		in   Inputs
		want Display
	}{
		{"interrupted folds to waiting", Inputs{Status: model.CodebaseStatusIndexing, Dependency: EmbedderBusy}, DisplayWaiting},
		{"not indexed folds to waiting", Inputs{Status: model.CodebaseStatusNotIndexed, Dependency: EmbedderUnreachable}, DisplayWaiting},
		{"indexed folds to waiting", Inputs{Status: model.CodebaseStatusIndexed, Dependency: EmbedderBusy}, DisplayWaiting},
		{"reconcile folds to waiting", Inputs{HasActiveJob: true, BackgroundSyncReconcile: true, Dependency: StoreUnavailable}, DisplayWaiting},
		{"live scoped stays indexing", Inputs{HasActiveJob: true, JobScopeKnown: true, Dependency: EmbedderBusy}, DisplayIndexing},
		{"stale stays stale", Inputs{Status: model.CodebaseStatusStale, Dependency: EmbedderBusy}, DisplayStale},
		{"failed stays failed", Inputs{Status: model.CodebaseStatusFailed, Dependency: EmbedderBusy}, DisplayFailed},
		{"missing stays missing", Inputs{Status: model.CodebaseStatusMissing, Dependency: EmbedderBusy}, DisplayMissing},
	}
	for _, testCase := range cases {
		got := ResolveDisplay(testCase.in)
		if got != testCase.want {
			t.Errorf("%s: ResolveDisplay(degraded) = %q, want %q", testCase.name, got, testCase.want)
		}
	}
}

// Every display has a non-empty glyph and label, and the lookups fall back
// safely for an unknown value.
func TestGlyphAndLabelCoverEveryDisplay(t *testing.T) {
	t.Parallel()
	for _, display := range []Display{
		DisplayPreparing, DisplayIndexing, DisplayIndexed, DisplayWaiting,
		DisplayStale, DisplayFailed, DisplayMissing,
	} {
		if GlyphFor(display) == "" {
			t.Errorf("%s: empty glyph", display)
		}
		if LabelFor(display) == "" {
			t.Errorf("%s: empty label", display)
		}
	}
	if GlyphFor(Display("bogus")) != "•" {
		t.Error("unknown display should fall back to the bullet glyph")
	}
	if LabelFor(Display("bogus")) != "bogus" {
		t.Error("unknown display should fall back to its raw token")
	}
}

// Healthy has no banner; every degraded mode resolves a non-empty headline, and
// an unrecognized degraded mode falls back to the generic headline.
func TestBannerHeadlineCoversEveryMode(t *testing.T) {
	t.Parallel()
	if BannerHeadlineFor(Healthy) != "" {
		t.Error("healthy mode must have no banner headline")
	}
	for _, mode := range []DependencyMode{EmbedderUnreachable, EmbedderRejected, EmbedderBusy, StoreUnavailable} {
		if !mode.Degraded() {
			t.Errorf("%s should report degraded", mode)
		}
		if BannerHeadlineFor(mode) == "" {
			t.Errorf("%s: empty banner headline", mode)
		}
	}
	if BannerHeadlineFor(DependencyMode("future_mode")) != genericDegradedHeadline {
		t.Error("unknown degraded mode should fall back to the generic headline")
	}
}

// Resolve sets BannerPresent from the mode and fills glyph and label from the
// resolved display, for both a healthy and a degraded snapshot.
func TestResolveSurfaceBanner(t *testing.T) {
	t.Parallel()
	healthy := Resolve(Inputs{Status: model.CodebaseStatusIndexed, Dependency: Healthy})
	if healthy.BannerPresent || healthy.BannerHeadline != "" {
		t.Errorf("healthy surface must carry no banner: %+v", healthy)
	}
	if healthy.Display != DisplayIndexed || healthy.Glyph != GlyphFor(DisplayIndexed) || healthy.Label != LabelFor(DisplayIndexed) {
		t.Errorf("healthy indexed surface mismatched: %+v", healthy)
	}
	degraded := Resolve(Inputs{Status: model.CodebaseStatusIndexed, Dependency: EmbedderBusy})
	if !degraded.BannerPresent || degraded.BannerHeadline == "" {
		t.Errorf("degraded surface must carry a banner: %+v", degraded)
	}
	if degraded.Display != DisplayWaiting {
		t.Errorf("degraded indexed surface should fold to waiting: %+v", degraded)
	}
}

// Each search outcome maps to its state note: only the repairing outcome carries
// text; the rest are silent.
func TestStateNoteForEveryOutcome(t *testing.T) {
	t.Parallel()
	withNote := map[SearchOutcome]bool{
		SearchNone:      false,
		SearchOK:        false,
		SearchEmpty:     false,
		SearchRepairing: true,
		SearchNotReady:  false,
		SearchMissing:   false,
		SearchFallback:  false,
	}
	for outcome, wantNote := range withNote {
		note := StateNoteFor(outcome)
		if wantNote && note == "" {
			t.Errorf("%s: expected a state note", outcome)
		}
		if !wantNote && note != "" {
			t.Errorf("%s: expected no state note, got %q", outcome, note)
		}
	}
}
