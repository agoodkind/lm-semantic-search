package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestGeneratedTablesMatchTheCheckedInFile regenerates the lexical tables from
// the pinned UCD files in ucd and requires the result to equal
// library/lexical_unicode_tables.go byte for byte.
func TestGeneratedTablesMatchTheCheckedInFile(t *testing.T) {
	generated, err := generate("ucd")
	if err != nil {
		t.Fatalf("generate the tables from the pinned UCD files: %v", err)
	}
	checkedIn, err := os.ReadFile(filepath.Join("..", "..", "lexical_unicode_tables.go"))
	if err != nil {
		t.Fatalf("read the checked-in tables: %v", err)
	}
	if !bytes.Equal(generated, checkedIn) {
		t.Fatalf("library/lexical_unicode_tables.go differs from the tables generated from ucd; run go generate in library")
	}
}
