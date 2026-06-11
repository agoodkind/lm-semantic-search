package daemon

import "testing"

func TestPlural(t *testing.T) {
	cases := []struct {
		word string
		n    int
		want string
	}{
		{"codebase", 0, "codebases"},
		{"codebase", 1, "codebase"},
		{"codebase", 2, "codebases"},
		{"file", 0, "files"},
		{"file", 1, "file"},
		{"file", 3, "files"},
		{"sub-folder", 1, "sub-folder"},
		{"sub-folder", 2, "sub-folders"},
	}
	for _, testCase := range cases {
		got := plural(testCase.word, testCase.n)
		if got != testCase.want {
			t.Errorf("plural(%q, %d) = %q, want %q", testCase.word, testCase.n, got, testCase.want)
		}
	}
}
