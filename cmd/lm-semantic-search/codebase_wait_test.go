package main

import "testing"

func TestCodebaseWaitRequiresHumanOutputMode(t *testing.T) {
	t.Parallel()

	root, _, stderr := testRoot()
	root.SetArgs([]string{"--json", "codebase", "wait", "/tmp/repo"})

	err := root.Execute()
	if err == nil {
		t.Fatal("expected codebase wait to fail in JSON mode")
	}
	if err.Error() != "wait requires human output mode" {
		t.Fatalf("error = %q, want wait requires human output mode", err.Error())
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty because Execute returns the error", stderr.String())
	}
}
