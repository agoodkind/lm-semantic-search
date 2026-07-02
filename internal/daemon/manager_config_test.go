package daemon

import (
	"slices"
	"testing"
)

func TestNormalizeSubmoduleListRejectsRootEscapesAndAbsolutePaths(t *testing.T) {
	input := []string{
		"third_party/lib",
		"a/..",
		"..",
		"../outside",
		"/tmp/lib",
	}

	got := normalizeSubmoduleList(input)
	want := []string{"third_party/lib"}
	if !slices.Equal(got, want) {
		t.Fatalf("normalizeSubmoduleList() = %v, want %v", got, want)
	}
}
