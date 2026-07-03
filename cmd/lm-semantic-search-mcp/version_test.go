package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestWriteVersionIncludesValidatorToken(t *testing.T) {
	var stdout bytes.Buffer

	if err := writeVersion(&stdout); err != nil {
		t.Fatalf("writeVersion returned error: %v", err)
	}
	output := stdout.String()
	if !strings.Contains(output, "version:") {
		t.Fatalf("version output = %q, want version: token", output)
	}
	if !strings.Contains(output, "commit=") {
		t.Fatalf("version output = %q, want commit field", output)
	}
	if !strings.Contains(output, "build_time=") {
		t.Fatalf("version output = %q, want build_time field", output)
	}
}
