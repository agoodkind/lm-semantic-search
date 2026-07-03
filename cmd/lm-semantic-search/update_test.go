package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestUpdateStatusTreatsMissingStateAsEmpty(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("CLAUDE_CONTEXTD_CONFIG_ROOT", t.TempDir())
	t.Setenv("CLAUDE_CONTEXTD_STATE_ROOT", t.TempDir())

	cmd := newUpdateStatusCmd()
	var stdout bytes.Buffer
	cmd.SetOut(&stdout)

	if err := cmd.Execute(); err != nil {
		t.Fatalf("update status returned error: %v", err)
	}

	output := stdout.String()
	if !strings.Contains(output, "current version:") {
		t.Fatalf("update status output = %q, want current version", output)
	}
	if !strings.Contains(output, "current commit:") {
		t.Fatalf("update status output = %q, want current commit", output)
	}
	if !strings.Contains(output, "current buildHash:") {
		t.Fatalf("update status output = %q, want current buildHash", output)
	}
	if strings.Contains(output, "last check:") {
		t.Fatalf("update status output = %q, want no last check for empty state", output)
	}
	if strings.Contains(output, "applied tag:") {
		t.Fatalf("update status output = %q, want no applied tag for empty state", output)
	}
}
