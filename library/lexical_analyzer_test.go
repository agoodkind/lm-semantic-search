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

// The expected classes and lowercase forms come from the Unicode 16.0.0
// UnicodeData.txt and DerivedCoreProperties.txt entries of each character.
// The results must not depend on the Go unicode tables of the toolchain, which
// are 15.0.0 in Go 1.26 and 17.0.0 in Go 1.27.
func TestLexicalCharacterTablesFollowUnicode16(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name         string
		character    rune
		alphanumeric bool
		lowercase    string
	}{
		{name: "U+0660 Arabic-Indic digit zero, Nd", character: 0x0660, alphanumeric: true, lowercase: "٠"},
		{name: "U+0130 dotted capital letter", character: 0x0130, alphanumeric: true, lowercase: "i̇"},
		{name: "U+1E030 Cyrillic modifier small a, Unicode 15.0", character: 0x1E030, alphanumeric: true, lowercase: "\U0001E030"},
		{name: "U+1C89 Cyrillic capital TJE, Unicode 16.0", character: 0x1C89, alphanumeric: true, lowercase: "ᲊ"},
		{name: "U+A7CB Latin capital rams horn, Unicode 16.0", character: 0xA7CB, alphanumeric: true, lowercase: "ɤ"},
		{name: "U+10D50 Garay capital A, Unicode 16.0", character: 0x10D50, alphanumeric: true, lowercase: "\U00010D70"},
		{name: "U+10940 Sidetic, Unicode 17.0", character: 0x10940, alphanumeric: false, lowercase: "\U00010940"},
		{name: "U+11F5A Kawi sign nukta, Mn", character: 0x11F5A, alphanumeric: false, lowercase: "\U00011F5A"},
		{name: "U+0301 combining acute, Mn", character: 0x0301, alphanumeric: false, lowercase: "́"},
		{name: "U+3000 ideographic space", character: 0x3000, alphanumeric: false, lowercase: "　"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			if got := isLexicalTokenCharacter(testCase.character); got != testCase.alphanumeric {
				t.Fatalf("isLexicalTokenCharacter(U+%04X) = %v, want %v", testCase.character, got, testCase.alphanumeric)
			}
			if got := lowercaseLexicalCharacter(testCase.character); got != testCase.lowercase {
				t.Fatalf("lowercaseLexicalCharacter(U+%04X) = %q, want %q", testCase.character, got, testCase.lowercase)
			}
		})
	}
}
