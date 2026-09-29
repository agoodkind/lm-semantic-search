package library

import (
	"errors"
	"slices"
	"strings"
	"testing"
)

// The expected tokens below are the tokens that Milvus 2.6.18 RunAnalyzer
// returned for the same inputs in TestLibraryLexicalAnalyzerParity.
func TestLexicalTokensMatchMilvusStandardAnalyzer(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name string
		text string
		want []string
	}{
		{name: "ASCII words and punctuation", text: "Hello, happy tax payer!", want: []string{"hello", "happy", "tax", "payer"}},
		{name: "underscore and dot separate tokens", text: "foo_bar v2.6.18", want: []string{"foo", "bar", "v2", "6", "18"}},
		{name: "U+0130 lowercases to two characters", text: "İstanbul", want: []string{"i̇stanbul"}},
		{name: "combining acute accent separates", text: "été", want: []string{"e", "te"}},
		{name: "CJK run is one token", text: "中文分词", want: []string{"中文分词"}},
		{name: "NUL ends the text", text: "before\x00after nul", want: []string{"before"}},
		{name: "Unicode 17 letter separates", text: "x\U00010940y", want: []string{"x", "y"}},
		{name: "no token", text: "!!! --- \t\n", want: nil},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			if got := lexicalTokens(testCase.text); !slices.Equal(got, testCase.want) {
				t.Fatalf("lexicalTokens(%q) = %q, want %q", testCase.text, got, testCase.want)
			}
		})
	}
}

func TestAnalyzeLexicalMergesTokensThatShareAHash(t *testing.T) {
	t.Parallel()
	longPrefix := strings.Repeat("p", lexicalHashPrefixBytes)
	for _, testCase := range []struct {
		name string
		text string
	}{
		{name: "CRC-32 collision", text: "hzswstzcso bkqqpyieph hzswstzcso"},
		{name: "shared 100-byte prefix", text: longPrefix + "alpha " + longPrefix + "omega " + longPrefix + "alpha"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			document := analyzeLexical(testCase.text)
			if len(document.terms) != 1 || document.terms[0].frequency != 3 || document.length != 3 {
				t.Fatalf("analyzeLexical(%q) = %+v, want one term with frequency 3 and length 3", testCase.text, document)
			}
		})
	}
}

func TestValidateLexicalAnalyzer(t *testing.T) {
	t.Parallel()
	if err := validateLexicalAnalyzer(StandardAnalyzer); err != nil {
		t.Fatalf("validateLexicalAnalyzer(%q) = %v, want nil", StandardAnalyzer, err)
	}
	if err := validateLexicalAnalyzer("milvus-english-v1"); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("validateLexicalAnalyzer(other identity) = %v, want ErrInvalidRequest", err)
	}
}
