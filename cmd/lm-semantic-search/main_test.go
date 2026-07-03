package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"goodkind.io/gklog/correlation"
)

func TestRootNoArgsShowsHelp(t *testing.T) {
	root, stdout, _ := testRoot()
	root.SetArgs([]string{})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute root help: %v", err)
	}
	output := stdout.String()
	if !strings.Contains(output, "Usage:") {
		t.Fatalf("help output missing Usage: %q", output)
	}
	if !strings.Contains(output, "codebase") {
		t.Fatalf("help output missing codebase group: %q", output)
	}
}

func TestRootUnknownCommandErrors(t *testing.T) {
	root, _, _ := testRoot()
	root.SetArgs([]string{"definitely-not-a-lm-semantic-command"})

	err := root.Execute()
	if err == nil {
		t.Fatalf("expected unknown command error")
	}
	if !strings.Contains(err.Error(), "unknown command") {
		t.Fatalf("error = %q, want unknown command", err.Error())
	}
}

func TestRootRegistersGroupedCommands(t *testing.T) {
	root, _, _ := testRoot()
	expected := [][]string{
		{"codebase", "list"},
		{"codebase", "status"},
		{"codebase", "index"},
		{"codebase", "sync"},
		{"codebase", "search"},
		{"codebase", "clear"},
		{"job", "list"},
		{"job", "get"},
		{"job", "cancel"},
		{"daemon", "status"},
		{"daemon", "stop"},
		{"daemon", "doctor"},
		{"update", "check"},
		{"update", "apply"},
		{"update", "status"},
	}
	for _, path := range expected {
		if _, _, err := root.Find(path); err != nil {
			t.Fatalf("command %v not registered: %v", path, err)
		}
	}
}

func TestCodebaseStatusRequiresPath(t *testing.T) {
	root, _, _ := testRoot()
	root.SetArgs([]string{"codebase", "status"})

	err := root.Execute()
	if err == nil {
		t.Fatalf("expected missing path error")
	}
	if err.Error() != "codebase status requires PATH" {
		t.Fatalf("error = %q", err.Error())
	}
}

// TestIndexRejectsWaitOutsideHumanMode proves --wait cannot interleave
// progress rendering with machine output.
func TestIndexRejectsWaitOutsideHumanMode(t *testing.T) {
	root, _, _ := testRoot()
	root.SetArgs([]string{"--json", "codebase", "index", "/tmp/x", "--wait"})

	err := root.Execute()
	if err == nil {
		t.Fatal("expected --wait with --json to error")
	}
	if !strings.Contains(err.Error(), "--wait requires human output") {
		t.Fatalf("error = %q, want --wait requires human output", err.Error())
	}
}

// TestRuntimeErrorsDoNotPrintUsage proves a post-parse failure prints no
// usage block (the Ctrl-C dump from the incident).
func TestRuntimeErrorsDoNotPrintUsage(t *testing.T) {
	root, stdout, stderr := testRoot()
	root.SetArgs([]string{"daemon", "status", "--socket", "/nonexistent/socket.sock"})

	_ = root.Execute()
	combined := stdout.String() + stderr.String()
	if strings.Contains(combined, "Usage:") {
		t.Fatalf("runtime error printed usage:\n%s", combined)
	}
}

func TestJobGetRequiresID(t *testing.T) {
	root, _, _ := testRoot()
	root.SetArgs([]string{"job", "get"})

	err := root.Execute()
	if err == nil {
		t.Fatalf("expected missing job id error")
	}
	if err.Error() != "job get requires JOB_ID" {
		t.Fatalf("error = %q", err.Error())
	}
}

func TestVersionCommandPrintsValidatorToken(t *testing.T) {
	root, stdout, _ := testRoot()
	root.SetArgs([]string{"version"})

	if err := root.Execute(); err != nil {
		t.Fatalf("version command returned error: %v", err)
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

func TestDaemonRequiresSubcommand(t *testing.T) {
	root, _, _ := testRoot()
	root.SetArgs([]string{"daemon"})

	err := root.Execute()
	if err == nil {
		t.Fatalf("expected daemon subcommand error")
	}
	if err.Error() != "daemon requires a subcommand" {
		t.Fatalf("error = %q", err.Error())
	}
}

func TestFormatCLIErrorKeepsCorrelationHeaderFirst(t *testing.T) {
	t.Parallel()
	message := correlation.HeaderMarker + "trace_id=abc span_id=def\ninternal error; see daemon logs"
	got := formatCLIError(assertError(message))
	if got != message+"\n" {
		t.Fatalf("formatCLIError returned %q", got)
	}
}

func TestFormatCLIErrorPrefixesOrdinaryErrors(t *testing.T) {
	t.Parallel()
	got := formatCLIError(assertError("plain failure"))
	if got != "Error: plain failure\n" {
		t.Fatalf("formatCLIError returned %q", got)
	}
}

func testRoot() (*cobra.Command, *bytes.Buffer, *bytes.Buffer) {
	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}
	root := newRoot("/tmp/lm-semantic-search.sock")
	root.SetOut(stdout)
	root.SetErr(stderr)
	return root, stdout, stderr
}

func assertError(message string) error {
	return &staticError{message: message}
}

type staticError struct {
	message string
}

func (err *staticError) Error() string {
	return err.message
}
