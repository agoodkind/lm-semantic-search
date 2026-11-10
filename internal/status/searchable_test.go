package status

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

// ResolveSearchable is the single fold for "can this path serve a search now":
// it is true only when the path is in-scope indexed and the shared backend is
// not degraded. Resolve must expose the same value on Surface.Searchable so every
// surface reads one resolution.
func TestResolveSearchable(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		eligible   bool
		dependency DependencyMode
		want       bool
	}{
		{"indexed and healthy", true, Healthy, true},
		{"indexed but store down", true, StoreUnavailable, false},
		{"indexed but embedder unreachable", true, EmbedderUnreachable, false},
		{"indexed but embedder busy", true, EmbedderBusy, false},
		{"not eligible and healthy", false, Healthy, false},
		{"not eligible and degraded", false, StoreUnavailable, false},
	}
	for _, testCase := range cases {
		in := Inputs{SearchableEligible: testCase.eligible, Dependency: testCase.dependency, Search: SearchNone}
		if got := ResolveSearchable(in); got != testCase.want {
			t.Fatalf("%s: ResolveSearchable = %v, want %v", testCase.name, got, testCase.want)
		}
		if got := Resolve(in).Searchable; got != testCase.want {
			t.Fatalf("%s: Resolve().Searchable = %v, want %v", testCase.name, got, testCase.want)
		}
	}
}

// Maintenance mode flips searchable false for an otherwise searchable path
// without raising the dependency banner or changing the display: the daemon is
// healthy, the operator has paused it.
func TestResolveSearchableMaintenance(t *testing.T) {
	t.Parallel()

	in := Inputs{Status: model.CodebaseStatusIndexed, SearchableEligible: true, Dependency: Healthy, Collection: CollectionReady, Maintenance: true}
	surface := Resolve(in)
	if surface.Searchable {
		t.Fatal("Searchable = true during maintenance, want false")
	}
	if surface.BannerPresent {
		t.Fatal("maintenance raised the dependency banner")
	}
	if surface.Display != DisplayIndexed {
		t.Fatalf("Display = %q during maintenance, want %q", surface.Display, DisplayIndexed)
	}
}

// Per-path collection readiness distinguishes requests the daemon accepts from
// those it rejects. Idle and loading accept a search and resolve residency on
// demand without raising a global dependency banner.
func TestResolveSearchableCollection(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		collection CollectionReadiness
		want       bool
	}{
		{"not probed stays searchable", CollectionNotApplicable, true},
		{"ready is searchable", CollectionReady, true},
		{"idle is searchable", CollectionIdle, true},
		{"absent blocks", CollectionAbsent, false},
		{"building blocks", CollectionBuilding, false},
		{"loading is searchable", CollectionLoading, true},
		{"unknown blocks", CollectionUnknown, false},
	}
	for _, testCase := range cases {
		in := Inputs{SearchableEligible: true, Dependency: Healthy, Collection: testCase.collection}
		if got := ResolveSearchable(in); got != testCase.want {
			t.Fatalf("%s: ResolveSearchable = %v, want %v", testCase.name, got, testCase.want)
		}
		// A per-path not-ready collection must not present the global store banner.
		if surface := Resolve(in); surface.BannerPresent {
			t.Fatalf("%s: per-path readiness must not raise the global banner", testCase.name)
		}
	}
}
