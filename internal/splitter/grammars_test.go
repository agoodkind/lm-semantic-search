package splitter

import (
	"context"
	"testing"
)

// TestModuleGrammarsParseThroughAST proves each grammar added as a Go-module
// dependency loads at runtime (its parser-format version is accepted) and chunks
// through the AST path rather than the recursive fallback. A rejected ABI would
// surface here as a "fallback" strategy.
func TestModuleGrammarsParseThroughAST(t *testing.T) {
	t.Parallel()

	cases := []struct {
		language string
		path     string
		source   string
	}{
		{"php", "greet.php", "<?php\nfunction greet() { return \"hi\"; }\n"},
		{"ruby", "greet.rb", "def greet\n  \"hi\"\nend\n"},
		{"bash", "greet.sh", "greet() {\n  echo hi\n}\n"},
		{"json", "data.json", "{\"name\": \"demo\", \"items\": [1, 2, 3]}\n"},
		{"html", "page.html", "<html><body><div><p>hi</p></div></body></html>\n"},
		{"css", "styles.css", ".box { color: red; margin: 0; }\n"},
		{"kotlin", "Greet.kt", "fun greet(): String { return \"hi\" }\n"},
		{"objective-c", "Greeter.m", "@interface Greeter : NSObject\n- (void)greet;\n@end\n"},
		{"dart", "main.dart", "class Greeter {\n  String greet() => \"hi\";\n}\n"},
		{"swift", "Greeter.swift", "struct Greeter {\n    func greet() -> String { return \"hi\" }\n}\n"},
	}

	for _, testCase := range cases {
		t.Run(testCase.language, func(t *testing.T) {
			t.Parallel()
			dispatcher := NewDispatcher()
			result, err := dispatcher.SplitFile(context.Background(), testCase.path, []byte(testCase.source))
			if err != nil {
				t.Fatalf("SplitFile(%s) returned error: %v", testCase.path, err)
			}
			if result.Strategy != "ast" {
				t.Fatalf("strategy for %s = %q, want ast (grammar did not load)", testCase.language, result.Strategy)
			}
			if len(result.Chunks) == 0 {
				t.Fatalf("expected at least one chunk for %s", testCase.language)
			}
		})
	}
}
