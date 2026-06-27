package discovery

import "testing"

func TestSimpleGlobMatch(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		text    string
		pattern string
		want    bool
	}{
		{name: "exact match", text: "node_modules", pattern: "node_modules", want: true},
		{name: "star wildcard", text: "main.test", pattern: "*.test", want: true},
		{name: "no match", text: "main.go", pattern: "*.test", want: false},
		{name: "dot is literal", text: "mainXgo", pattern: "main.go", want: false},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			if got := simpleGlobMatch(testCase.text, testCase.pattern); got != testCase.want {
				t.Fatalf("simpleGlobMatch(%q, %q) = %v, want %v", testCase.text, testCase.pattern, got, testCase.want)
			}
		})
	}
}

func BenchmarkSimpleGlobMatch(b *testing.B) {
	b.ReportAllocs()
	for range b.N {
		_ = simpleGlobMatch("node_modules/pkg/index.js", "node_modules")
	}
}
