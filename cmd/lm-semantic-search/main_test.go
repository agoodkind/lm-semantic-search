package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/spf13/cobra"
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

func testRoot() (*cobra.Command, *bytes.Buffer, *bytes.Buffer) {
	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}
	root := newRoot("/tmp/lm-semantic-search.sock")
	root.SetOut(stdout)
	root.SetErr(stderr)
	return root, stdout, stderr
}
