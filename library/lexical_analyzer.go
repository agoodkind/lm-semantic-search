package library

//go:generate go run ./internal/lexicalunicodegen -output lexical_unicode_tables.go

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

// lexicalLowercaseMapping is one character and its Rust char::to_lowercase
// result.
type lexicalLowercaseMapping struct {
	from rune
	to   string
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
// accepts character. It reads the generated Unicode 16.0.0 table
// lexicalAlphanumeric and none of the Go unicode tables of the build
// toolchain.
func isLexicalTokenCharacter(character rune) bool {
	if character < utf8.RuneSelf {
		return 'a' <= character && character <= 'z' ||
			'A' <= character && character <= 'Z' ||
			'0' <= character && character <= '9'
	}
	return unicode.Is(lexicalAlphanumeric, character)
}

// validateLexicalAnalyzer returns an error that wraps [ErrInvalidRequest] when
// identity is not [StandardAnalyzer], the only analyzer this build
// implements.
func validateLexicalAnalyzer(identity string) error {
	if identity != StandardAnalyzer {
		return invalidRequest(fmt.Sprintf("analyzer identity %q is not %q", identity, StandardAnalyzer))
	}
	return nil
}

// lowercaseLexicalToken applies tantivy LowerCaser. An ASCII token is
// lowercased byte by byte. Every character of any other token is replaced by
// its Rust char::to_lowercase result from the generated Unicode 16.0.0 table
// lexicalLowercase.
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
		lowered.WriteString(lowercaseLexicalCharacter(character))
	}
	return lowered.String()
}

// lowercaseLexicalCharacter returns the Rust char::to_lowercase result of one
// character.
func lowercaseLexicalCharacter(character rune) string {
	index, found := slices.BinarySearchFunc(lexicalLowercase, character, func(mapping lexicalLowercaseMapping, target rune) int {
		return int(mapping.from - target)
	})
	if found {
		return lexicalLowercase[index].to
	}
	return string(character)
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
