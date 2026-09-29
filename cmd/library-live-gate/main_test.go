package main

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// gateFixture is a real Go test package with passing, failing, skipping, and
// tagged tests. Each case selects a subset with -run.
var gateFixture = map[string]string{
	"go.mod": "module example.com/gatefixture\n\ngo 1.24\n",
	"fixture_test.go": `package gatefixture

import "testing"

func TestGatePassFirst(t *testing.T) {}

func TestGatePassSecond(t *testing.T) {
	t.Run("child", func(t *testing.T) {})
}

func TestGateFailBroken(t *testing.T) { t.Fatal("broken on purpose") }

func TestGateSkipOnly(t *testing.T) { t.Skip("dependency unavailable on purpose") }

func TestGateMixedPass(t *testing.T) {}

func TestGateMixedSkip(t *testing.T) { t.Skip("mixed skip on purpose") }

func TestGateSubSkip(t *testing.T) {
	t.Run("skipped child", func(t *testing.T) { t.Skip("child skip on purpose") })
}
`,
	"slow_test.go": `package gatefixture

import (
	"testing"
	"time"
)

func TestGateSlowFinishes(t *testing.T) {}

func TestGateSlowSleeps(t *testing.T) { time.Sleep(time.Minute) }
`,
	"tagged_test.go": `//go:build gatetag

package gatefixture

import "testing"

func TestGateTagged(t *testing.T) {}
`,
}

func buildGate(t *testing.T) string {
	t.Helper()
	binary := filepath.Join(t.TempDir(), "library-live-gate")
	build := exec.Command("go", "build", "-o", binary, ".")
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build gate: %v\n%s", err, output)
	}
	return binary
}

func writeFixture(t *testing.T) string {
	t.Helper()
	directory := t.TempDir()
	for name, content := range gateFixture {
		if err := os.WriteFile(filepath.Join(directory, name), []byte(content), 0o600); err != nil {
			t.Fatalf("write fixture %s: %v", name, err)
		}
	}
	return directory
}

func TestGateExitStatusAndDiagnostics(t *testing.T) {
	t.Parallel()
	binary := buildGate(t)
	fixture := writeFixture(t)

	for _, testCase := range []struct {
		name       string
		arguments  []string
		wantStatus int
		wantStderr []string
	}{
		{
			name:       "all selected tests pass",
			arguments:  []string{"-run", "^TestGatePass", "./"},
			wantStatus: exitPassed,
			wantStderr: []string{"go test -json -count=1 -timeout 30m0s", "PASS: 3 selected, 3 passed"},
		},
		{
			name:       "go test times out",
			arguments:  []string{"-timeout", "2s", "-run", "^TestGateSlow", "./"},
			wantStatus: 1,
			wantStderr: []string{
				"go test -json -count=1 -timeout 2s",
				"go test timed out (panic: test timed out after 2s",
				"unfinished tests: TestGateSlowSleeps",
			},
		},
		{
			name:       "non-positive timeout",
			arguments:  []string{"-timeout", "0s", "-run", "^TestGatePass", "./"},
			wantStatus: exitUsage,
			wantStderr: []string{"usage: library-live-gate [-timeout DURATION]"},
		},
		{
			name:       "a selected test fails",
			arguments:  []string{"-run", "^TestGate(Pass|Fail)", "./"},
			wantStatus: 1,
			wantStderr: []string{"go test exited with status 1", "failed tests: TestGateFailBroken"},
		},
		{
			name:       "only skipped tests",
			arguments:  []string{"-run", "^TestGateSkipOnly$", "./"},
			wantStatus: exitGateFailed,
			wantStderr: []string{"skipped tests: TestGateSkipOnly", "dependency unavailable on purpose", "no selected test passed"},
		},
		{
			name:       "a skip beside a pass",
			arguments:  []string{"-run", "^TestGateMixed", "./"},
			wantStatus: exitGateFailed,
			wantStderr: []string{"skipped tests: TestGateMixedSkip"},
		},
		{
			name:       "a skipped subtest",
			arguments:  []string{"-run", "^TestGateSubSkip$", "./"},
			wantStatus: exitGateFailed,
			wantStderr: []string{"skipped tests: TestGateSubSkip/skipped_child"},
		},
		{
			name:       "no test matches",
			arguments:  []string{"-run", "^TestGateNothing$", "./"},
			wantStatus: exitGateFailed,
			wantStderr: []string{`no selected test ran for -run "^TestGateNothing$"`},
		},
		{
			name:       "tagged test without its tag",
			arguments:  []string{"-run", "^TestGateTagged$", "./"},
			wantStatus: exitGateFailed,
			wantStderr: []string{"no selected test ran"},
		},
		{
			name:       "tagged test with its tag",
			arguments:  []string{"-tags", "gatetag", "-run", "^TestGateTagged$", "./"},
			wantStatus: exitPassed,
			wantStderr: []string{"go test -json -count=1 -timeout 30m0s -tags gatetag", "PASS: 1 selected, 1 passed"},
		},
		{
			name:       "missing pattern",
			arguments:  []string{"./"},
			wantStatus: exitUsage,
			wantStderr: []string{"usage: library-live-gate"},
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			command := exec.Command(binary, testCase.arguments...)
			command.Dir = fixture
			command.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=")
			var stdout, stderr bytes.Buffer
			command.Stdout = &stdout
			command.Stderr = &stderr
			err := command.Run()
			status := 0
			var exitErr *exec.ExitError
			if errors.As(err, &exitErr) {
				status = exitErr.ExitCode()
			} else if err != nil {
				t.Fatalf("run gate: %v", err)
			}
			if status != testCase.wantStatus {
				t.Fatalf("exit status = %d, want %d\nstderr:\n%s\nstdout:\n%s", status, testCase.wantStatus, stderr.String(), stdout.String())
			}
			for _, want := range testCase.wantStderr {
				if !strings.Contains(stderr.String(), want) {
					t.Fatalf("stderr does not contain %q:\n%s", want, stderr.String())
				}
			}
		})
	}
}
