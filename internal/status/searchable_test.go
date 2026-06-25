package status

import "testing"

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

// Per-path collection readiness blocks search on its own, with the store globally
// healthy, so a not-loaded collection is not searchable without raising any global
// dependency banner. The zero value (not probed) and ready do not block.
func TestResolveSearchableCollection(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name       string
		collection CollectionReadiness
		want       bool
	}{
		{"not probed stays searchable", CollectionNotApplicable, true},
		{"ready is searchable", CollectionReady, true},
		{"absent blocks", CollectionAbsent, false},
		{"building blocks", CollectionBuilding, false},
		{"loading blocks", CollectionLoading, false},
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
