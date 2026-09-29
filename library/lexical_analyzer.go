package library

import (
	"fmt"
	"hash/crc32"
	"math"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	// lexicalHashPrefixBytes is the number of leading token bytes that the
	// Milvus BM25 function hashes. Tokens that share these bytes share a term.
	lexicalHashPrefixBytes = 100
	// lexicalMaxTermFrequency is the largest term frequency the Milvus BM25
	// function can represent. It counts a token by adding 1 to a float32, and
	// the float32 sum stops growing at 2^24.
	lexicalMaxTermFrequency = 1 << 24
	// dottedCapitalI lowercases to two characters under Rust char::to_lowercase
	// and to one character under [unicode.ToLower].
	dottedCapitalI = 'İ'
	// dottedCapitalILower is the Rust char::to_lowercase expansion of
	// dottedCapitalI.
	dottedCapitalILower = "i̇"
)

// lexicalTerm is one distinct term hash of an analyzed text and the number of
// tokens with that hash.
type lexicalTerm struct {
	hash      uint32
	frequency uint32
}

// lexicalDocument is the analyzed form of one SearchText. Terms are sorted by
// ascending hash with distinct hashes. Length is the sum of every term
// frequency, which Milvus uses as the document length.
type lexicalDocument struct {
	terms  []lexicalTerm
	length uint64
}

// analyzeLexical analyzes text with [StandardAnalyzer] and merges tokens that
// share a hash. The caller validates that text is UTF-8.
func analyzeLexical(text string) lexicalDocument {
	frequencies := make(map[uint32]uint32)
	forEachLexicalToken(text, func(token string) {
		hash := lexicalTermHash(token)
		if frequencies[hash] < lexicalMaxTermFrequency {
			frequencies[hash]++
		}
	})
	document := lexicalDocument{terms: make([]lexicalTerm, 0, len(frequencies)), length: 0}
	for hash, frequency := range frequencies {
		document.terms = append(document.terms, lexicalTerm{hash: hash, frequency: frequency})
		document.length += uint64(frequency)
	}
	slices.SortFunc(document.terms, func(left lexicalTerm, right lexicalTerm) int {
		switch {
		case left.hash < right.hash:
			return -1
		case left.hash > right.hash:
			return 1
		default:
			return 0
		}
	})
	return document
}

// lexicalTokens returns the lowercased tokens of text in order, before hashing.
func lexicalTokens(text string) []string {
	var tokens []string
	forEachLexicalToken(text, func(token string) {
		tokens = append(tokens, token)
	})
	return tokens
}

// forEachLexicalToken calls emit with every lowercased token of text in order.
// A token is a maximal run of characters that Rust char::is_alphanumeric
// accepts. Milvus passes text to tantivy as a C string, and tantivy reads no
// byte after the first NUL. This analyzer also stops at the first NUL.
func forEachLexicalToken(text string, emit func(string)) {
	if end := strings.IndexByte(text, 0); end >= 0 {
		text = text[:end]
	}
	start := -1
	for offset, character := range text {
		if isLexicalTokenCharacter(character) {
			if start < 0 {
				start = offset
			}
			continue
		}
		if start >= 0 {
			emit(lowercaseLexicalToken(text[start:offset]))
			start = -1
		}
	}
	if start >= 0 {
		emit(lowercaseLexicalToken(text[start:]))
	}
}

// isLexicalTokenCharacter reports whether Rust 1.89 char::is_alphanumeric
// accepts character: the Alphabetic derived property or a general category of
// Nd, Nl, or No. Alphabetic is Lu, Ll, Lt, Lm, Lo, Nl, and Other_Alphabetic.
// Rust 1.89 uses Unicode 16.0.0 tables. It rejects every character in
// unicode17Alphanumerics.
func isLexicalTokenCharacter(character rune) bool {
	if character < utf8.RuneSelf {
		return 'a' <= character && character <= 'z' ||
			'A' <= character && character <= 'Z' ||
			'0' <= character && character <= '9'
	}
	if unicode.Is(unicode17Alphanumerics, character) {
		return false
	}
	return unicode.IsLetter(character) ||
		unicode.IsNumber(character) ||
		unicode.Is(unicode.Other_Alphabetic, character)
}

// lexicalUnicodeVersion is the Go unicode table version that
// unicode17Alphanumerics corrects.
const lexicalUnicodeVersion = "17.0.0"

// unicode17Alphanumerics lists every character that Unicode 17.0.0 first
// assigns and that Go 1.26 classifies as a letter, a number, or
// Other_Alphabetic. The Milvus 2.6.18 analyzer treats each of them as a
// separator. The list equals both the Unicode 17.0.0 DerivedAge assignments
// with those properties and the 4,672 differences that an exhaustive
// RunAnalyzer scan of every scalar value reported against Milvus 2.6.18.
var unicode17Alphanumerics = &unicode.RangeTable{
	R16: []unicode.Range16{
		{Lo: 0x088F, Hi: 0x088F, Stride: 1},
		{Lo: 0x0C5C, Hi: 0x0C5C, Stride: 1},
		{Lo: 0x0CDC, Hi: 0x0CDC, Stride: 1},
		{Lo: 0xA7CE, Hi: 0xA7CF, Stride: 1},
		{Lo: 0xA7D2, Hi: 0xA7D2, Stride: 1},
		{Lo: 0xA7D4, Hi: 0xA7D4, Stride: 1},
		{Lo: 0xA7F1, Hi: 0xA7F1, Stride: 1},
	},
	R32: []unicode.Range32{
		{Lo: 0x10940, Hi: 0x10959, Stride: 1},
		{Lo: 0x10EC5, Hi: 0x10EC7, Stride: 1},
		{Lo: 0x10EFA, Hi: 0x10EFB, Stride: 1},
		{Lo: 0x11B60, Hi: 0x11B67, Stride: 1},
		{Lo: 0x11DB0, Hi: 0x11DDB, Stride: 1},
		{Lo: 0x11DE0, Hi: 0x11DE9, Stride: 1},
		{Lo: 0x16EA0, Hi: 0x16EB8, Stride: 1},
		{Lo: 0x16EBB, Hi: 0x16ED3, Stride: 1},
		{Lo: 0x16FF2, Hi: 0x16FF6, Stride: 1},
		{Lo: 0x187F8, Hi: 0x187FF, Stride: 1},
		{Lo: 0x18D09, Hi: 0x18D1E, Stride: 1},
		{Lo: 0x18D80, Hi: 0x18DF2, Stride: 1},
		{Lo: 0x1E6C0, Hi: 0x1E6DE, Stride: 1},
		{Lo: 0x1E6E0, Hi: 0x1E6F5, Stride: 1},
		{Lo: 0x1E6FE, Hi: 0x1E6FF, Stride: 1},
		{Lo: 0x2B73A, Hi: 0x2B73F, Stride: 1},
		{Lo: 0x2CEA2, Hi: 0x2CEAD, Stride: 1},
		{Lo: 0x323B0, Hi: 0x33479, Stride: 1},
	},
	LatinOffset: 0,
}

// validateLexicalAnalyzer reports whether the running binary reproduces
// [StandardAnalyzer] for identity. The correction table matches one version of
// the Go unicode tables. Another version classifies a different set of
// characters, and this function then returns an error that wraps
// [ErrInvalidRequest].
func validateLexicalAnalyzer(identity string) error {
	if identity != StandardAnalyzer {
		return invalidRequest(fmt.Sprintf("analyzer identity %q is not %q", identity, StandardAnalyzer))
	}
	if unicode.Version != lexicalUnicodeVersion {
		return invalidRequest(fmt.Sprintf(
			"analyzer %s was verified with Go unicode tables %s, and this binary uses %s",
			StandardAnalyzer,
			lexicalUnicodeVersion,
			unicode.Version,
		))
	}
	return nil
}

// lowercaseLexicalToken applies tantivy LowerCaser. An ASCII token is
// lowercased byte by byte. Every character of any other token is replaced by
// its Rust char::to_lowercase mapping. Over every token character, that
// mapping equals [unicode.ToLower] except for dottedCapitalI.
func lowercaseLexicalToken(token string) string {
	ascii := true
	for index := range len(token) {
		if token[index] >= utf8.RuneSelf {
			ascii = false
			break
		}
	}
	if ascii {
		return strings.ToLower(token)
	}
	var lowered strings.Builder
	lowered.Grow(len(token))
	for _, character := range token {
		if character == dottedCapitalI {
			lowered.WriteString(dottedCapitalILower)
			continue
		}
		lowered.WriteRune(unicode.ToLower(character))
	}
	return lowered.String()
}

// lexicalTermHash returns the Milvus BM25 function hash of token: CRC-32 IEEE
// over at most the first lexicalHashPrefixBytes bytes, modulo [math.MaxUint32].
func lexicalTermHash(token string) uint32 {
	hashed := token
	if len(hashed) > lexicalHashPrefixBytes {
		hashed = hashed[:lexicalHashPrefixBytes]
	}
	return crc32.ChecksumIEEE([]byte(hashed)) % math.MaxUint32
}
