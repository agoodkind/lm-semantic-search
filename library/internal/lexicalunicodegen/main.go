// Command lexicalunicodegen writes the Unicode 16.0.0 character tables of the
// library lexical analyzer. Milvus 2.6.18 tokenizes with tantivy built by
// Rust 1.89, and Rust 1.89 derives char::is_alphanumeric and
// char::to_lowercase from Unicode 16.0.0. The command reads the checked-in
// copies of the UCD files in the ucd directory, checks their pinned SHA-256
// values, and writes a Go file that classifies and lowercases characters
// without the Go unicode package tables of the build toolchain.
//
// Run it from the library directory with go generate.
package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"go/format"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

const (
	unicodeVersion = "16.0.0"
	ucdBaseURL     = "https://www.unicode.org/Public/" + unicodeVersion + "/ucd/"
	outputFileMode = 0o644
	// maximumRange16 is the largest code point of a unicode.Range16 entry.
	maximumRange16 = 0xFFFF
	// maximumLatin1 is the largest Latin-1 code point. unicode.RangeTable
	// LatinOffset counts the R16 entries at or below it.
	maximumLatin1 = 0xFF
	// rangesPerLine and mappingsPerLine set how many generated table entries
	// share one source line.
	rangesPerLine   = 3
	mappingsPerLine = 3
)

// ucdFile is one pinned Unicode Character Database file.
type ucdFile struct {
	name   string
	sha256 string
}

var (
	unicodeDataFile = ucdFile{
		name:   "UnicodeData.txt",
		sha256: "ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f",
	}
	derivedCorePropertiesFile = ucdFile{
		name:   "DerivedCoreProperties.txt",
		sha256: "39d35161f2954497f69e08bdb9e701493f476a3d30222de20028feda36c1dabd",
	}
	specialCasingFile = ucdFile{
		name:   "SpecialCasing.txt",
		sha256: "8d5de354eef79f2395a54c9c7dcebbaf3d30fc962d0f85611ea97aa973a0c451",
	}
)

// lowercaseMapping is one character with a lowercase form that differs from
// the character.
type lowercaseMapping struct {
	from rune
	to   []rune
}

// characterTables are the analyzer tables derived from the UCD files.
type characterTables struct {
	alphanumeric []bool
	lowercase    []lowercaseMapping
}

func main() {
	slog.Debug("lexical unicode generator process entry")
	os.Exit(runMain())
}

func runMain() int {
	output := flag.String("output", "lexical_unicode_tables.go", "path of the generated Go file")
	ucdDirectory := flag.String("ucd", "internal/lexicalunicodegen/ucd", "directory with the pinned UCD files")
	flag.Parse()

	source, err := generate(*ucdDirectory)
	if err != nil {
		slog.Error("generate lexical unicode tables failed", "err", err)
		return 1
	}
	if err := os.WriteFile(*output, source, outputFileMode); err != nil {
		slog.Error("write lexical unicode tables failed", "path", *output, "err", err)
		return 1
	}
	slog.Info("wrote lexical unicode tables", "path", *output, "unicode", unicodeVersion)
	return 0
}

// generate reads the pinned UCD files from ucdDirectory and returns the
// formatted Go source of the tables.
func generate(ucdDirectory string) ([]byte, error) {
	files := map[string][]byte{}
	for _, file := range []ucdFile{unicodeDataFile, derivedCorePropertiesFile, specialCasingFile} {
		content, err := readPinned(ucdDirectory, file)
		if err != nil {
			return nil, err
		}
		files[file.name] = content
	}
	tables, err := buildTables(files)
	if err != nil {
		return nil, err
	}
	return render(tables)
}

// readPinned reads one UCD file and requires its pinned SHA-256.
func readPinned(ucdDirectory string, file ucdFile) ([]byte, error) {
	path := filepath.Join(ucdDirectory, file.name)
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, generatorError(fmt.Errorf("read %s: %w", path, err))
	}
	digest := sha256.Sum256(content)
	if got := hex.EncodeToString(digest[:]); got != file.sha256 {
		return nil, generatorError(fmt.Errorf("%s has SHA-256 %s, want %s", path, got, file.sha256))
	}
	return content, nil
}

// buildTables derives the Rust 1.89 char::is_alphanumeric set, which is the
// Alphabetic property or a general category of Nd, Nl, or No, and the
// char::to_lowercase mappings, which are the UnicodeData simple lowercase
// mappings replaced by the unconditional SpecialCasing lowercase mappings.
func buildTables(files map[string][]byte) (characterTables, error) {
	tables := characterTables{alphanumeric: make([]bool, utf8.MaxRune+1), lowercase: nil}
	lowercase := map[rune][]rune{}
	if err := readUnicodeData(files[unicodeDataFile.name], tables.alphanumeric, lowercase); err != nil {
		return characterTables{}, err
	}
	if err := readAlphabetic(files[derivedCorePropertiesFile.name], tables.alphanumeric); err != nil {
		return characterTables{}, err
	}
	if err := readSpecialCasing(files[specialCasingFile.name], lowercase); err != nil {
		return characterTables{}, err
	}
	for from, to := range lowercase {
		if len(to) != 1 || to[0] != from {
			tables.lowercase = append(tables.lowercase, lowercaseMapping{from: from, to: to})
		}
	}
	slices.SortFunc(tables.lowercase, func(left lowercaseMapping, right lowercaseMapping) int {
		return int(left.from - right.from)
	})
	return tables, nil
}

// readUnicodeData marks the Nd, Nl, and No characters, including the
// characters of First and Last range pairs, and reads the simple lowercase
// mappings.
func readUnicodeData(content []byte, alphanumeric []bool, lowercase map[rune][]rune) error {
	rangeStart := rune(-1)
	scanner := bufio.NewScanner(bytes.NewReader(content))
	for scanner.Scan() {
		fields := strings.Split(scanner.Text(), ";")
		if len(fields) != 15 {
			return generatorError(fmt.Errorf("UnicodeData line %q has %d fields, want 15", scanner.Text(), len(fields)))
		}
		character, err := parseCodePoint(fields[0])
		if err != nil {
			return err
		}
		numeric := fields[2] == "Nd" || fields[2] == "Nl" || fields[2] == "No"
		first, last := character, character
		switch {
		case strings.HasSuffix(fields[1], ", First>"):
			rangeStart = character
			continue
		case strings.HasSuffix(fields[1], ", Last>"):
			if rangeStart < 0 {
				return generatorError(fmt.Errorf("UnicodeData range end %04X has no start", character))
			}
			first = rangeStart
			rangeStart = -1
		}
		for value := first; value <= last; value++ {
			if numeric {
				alphanumeric[value] = true
			}
		}
		if fields[13] != "" {
			lower, err := parseCodePoint(fields[13])
			if err != nil {
				return err
			}
			lowercase[character] = []rune{lower}
		}
	}
	if err := scanner.Err(); err != nil {
		return generatorError(fmt.Errorf("scan UnicodeData: %w", err))
	}
	return nil
}

// readAlphabetic marks every character with the Alphabetic property.
func readAlphabetic(content []byte, alphanumeric []bool) error {
	scanner := bufio.NewScanner(bytes.NewReader(content))
	for scanner.Scan() {
		line, _, _ := strings.Cut(scanner.Text(), "#")
		fields := strings.Split(line, ";")
		if len(fields) != 2 || strings.TrimSpace(fields[1]) != "Alphabetic" {
			continue
		}
		first, last, err := parseCodePointRange(strings.TrimSpace(fields[0]))
		if err != nil {
			return err
		}
		for value := first; value <= last; value++ {
			alphanumeric[value] = true
		}
	}
	if err := scanner.Err(); err != nil {
		return generatorError(fmt.Errorf("scan DerivedCoreProperties: %w", err))
	}
	return nil
}

// readSpecialCasing replaces the lowercase mapping of each character that has
// an unconditional SpecialCasing entry.
func readSpecialCasing(content []byte, lowercase map[rune][]rune) error {
	scanner := bufio.NewScanner(bytes.NewReader(content))
	for scanner.Scan() {
		line, _, _ := strings.Cut(scanner.Text(), "#")
		fields := strings.Split(line, ";")
		if len(fields) != 5 || strings.TrimSpace(fields[4]) != "" {
			continue
		}
		character, err := parseCodePoint(strings.TrimSpace(fields[0]))
		if err != nil {
			return err
		}
		var mapping []rune
		for _, value := range strings.Fields(fields[1]) {
			parsed, err := parseCodePoint(value)
			if err != nil {
				return err
			}
			mapping = append(mapping, parsed)
		}
		lowercase[character] = mapping
	}
	if err := scanner.Err(); err != nil {
		return generatorError(fmt.Errorf("scan SpecialCasing: %w", err))
	}
	return nil
}

func parseCodePoint(text string) (rune, error) {
	value, err := strconv.ParseUint(text, 16, 32)
	if err != nil || value > utf8.MaxRune {
		return 0, generatorError(fmt.Errorf("code point %q is invalid: %w", text, err))
	}
	return rune(value), nil
}

func parseCodePointRange(text string) (rune, rune, error) {
	firstText, lastText, isRange := strings.Cut(text, "..")
	first, err := parseCodePoint(firstText)
	if err != nil {
		return 0, 0, err
	}
	if !isRange {
		return first, first, nil
	}
	last, err := parseCodePoint(lastText)
	if err != nil {
		return 0, 0, err
	}
	return first, last, nil
}

// characterRange is one run of consecutive alphanumeric characters.
type characterRange struct {
	low, high rune
}

func alphanumericRanges(alphanumeric []bool) []characterRange {
	var ranges []characterRange
	for value := rune(0); value <= utf8.MaxRune; value++ {
		if !alphanumeric[value] {
			continue
		}
		if length := len(ranges); length > 0 && ranges[length-1].high == value-1 {
			ranges[length-1].high = value
			continue
		}
		ranges = append(ranges, characterRange{low: value, high: value})
	}
	return ranges
}

// render writes the generated Go source and formats it.
func render(tables characterTables) ([]byte, error) {
	var source strings.Builder
	source.WriteString("// Code generated by go run ./internal/lexicalunicodegen; DO NOT EDIT.\n\n")
	fmt.Fprintf(&source, "// The tables come from the Unicode %s Character Database files in\n", unicodeVersion)
	source.WriteString("// internal/lexicalunicodegen/ucd, copied from:\n")
	for _, file := range []ucdFile{unicodeDataFile, derivedCorePropertiesFile, specialCasingFile} {
		fmt.Fprintf(&source, "// %s%s SHA-256 %s\n", ucdBaseURL, file.name, file.sha256)
	}
	source.WriteString("\npackage library\n\nimport \"unicode\"\n\n")

	ranges := alphanumericRanges(tables.alphanumeric)
	var small, large []characterRange
	latinOffset := 0
	for _, characters := range ranges {
		if characters.low <= maximumRange16 && characters.high > maximumRange16 {
			small = append(small, characterRange{low: characters.low, high: maximumRange16})
			characters.low = maximumRange16 + 1
		}
		if characters.high <= maximumRange16 {
			small = append(small, characters)
			if characters.high <= maximumLatin1 {
				latinOffset++
			}
			continue
		}
		large = append(large, characters)
	}
	source.WriteString("// lexicalAlphanumeric contains the characters that Rust char::is_alphanumeric\n")
	source.WriteString("// accepts: the Alphabetic property or a general category of Nd, Nl, or No.\n")
	source.WriteString("var lexicalAlphanumeric = &unicode.RangeTable{\n\tR16: []unicode.Range16{\n")
	writeRanges(&source, small)
	source.WriteString("\t},\n\tR32: []unicode.Range32{\n")
	writeRanges(&source, large)
	fmt.Fprintf(&source, "\t},\n\tLatinOffset: %d,\n}\n\n", latinOffset)

	source.WriteString("// lexicalLowercase lists every character with a Rust char::to_lowercase\n")
	source.WriteString("// result that differs from the character, in ascending order.\n")
	source.WriteString("var lexicalLowercase = []lexicalLowercaseMapping{\n")
	for index, mapping := range tables.lowercase {
		if index%mappingsPerLine == 0 {
			source.WriteString("\t")
		}
		// %+q escapes every non-ASCII character. Plain %q leaves a character
		// unescaped when the toolchain's strconv.IsPrint accepts it, and those
		// tables differ between Go releases.
		fmt.Fprintf(&source, "{from: 0x%04X, to: %+q},", mapping.from, string(mapping.to))
		if index%mappingsPerLine == mappingsPerLine-1 || index == len(tables.lowercase)-1 {
			source.WriteString("\n")
			continue
		}
		source.WriteString(" ")
	}
	source.WriteString("}\n")
	formatted, err := format.Source([]byte(source.String()))
	if err != nil {
		return nil, generatorError(fmt.Errorf("format generated source: %w", err))
	}
	return formatted, nil
}

func writeRanges(source *strings.Builder, ranges []characterRange) {
	for index, characters := range ranges {
		if index%rangesPerLine == 0 {
			source.WriteString("\t\t")
		}
		fmt.Fprintf(source, "{Lo: 0x%04X, Hi: 0x%04X, Stride: 1},", characters.low, characters.high)
		if index%rangesPerLine == rangesPerLine-1 || index == len(ranges)-1 {
			source.WriteString("\n")
			continue
		}
		source.WriteString(" ")
	}
}

// generatorError logs a generator failure and returns it.
func generatorError(err error) error {
	slog.Error("lexical unicode generator failed", "err", err)
	return err
}
