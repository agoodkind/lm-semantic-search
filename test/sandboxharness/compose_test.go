//go:build restartacceptance

package sandboxharness

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"
)

type recordingCommandRunner struct {
	calls  [][]string
	failUp bool
}

func (runner *recordingCommandRunner) Run(_ context.Context, _ map[string]string, name string, arguments ...string) ([]byte, error) {
	runner.calls = append(runner.calls, append([]string{name}, arguments...))
	if runner.failUp && arguments[len(arguments)-1] == "--wait" {
		return nil, errors.New("injected startup failure")
	}
	return nil, nil
}

func TestComposeProjectStartsWaitsRunsAndCleans(t *testing.T) {
	runner := &recordingCommandRunner{}
	project := ComposeProject{
		Runner:         runner,
		Name:           "sandbox-run-accepted",
		File:           "/guest/run/compose.yaml",
		CleanupTimeout: time.Second,
	}
	ran := false
	if err := project.Run(context.Background(), func(context.Context) error {
		ran = true
		return nil
	}); err != nil {
		t.Fatalf("run compose project: %v", err)
	}
	want := [][]string{
		{"docker", "compose", "-p", "sandbox-run-accepted", "-f", "/guest/run/compose.yaml", "up", "-d", "--wait"},
		{"docker", "compose", "-p", "sandbox-run-accepted", "-f", "/guest/run/compose.yaml", "down", "--volumes", "--remove-orphans"},
	}
	if !ran || !reflect.DeepEqual(runner.calls, want) {
		t.Fatalf("ran=%v calls=%v, want %v", ran, runner.calls, want)
	}
}

func TestComposeProjectCleansAfterPartialStartup(t *testing.T) {
	runner := &recordingCommandRunner{failUp: true}
	project := ComposeProject{
		Runner:         runner,
		Name:           "sandbox-run-accepted",
		File:           "/guest/run/compose.yaml",
		CleanupTimeout: time.Second,
	}
	if err := project.Run(context.Background(), func(context.Context) error { return nil }); err == nil {
		t.Fatal("startup failure was accepted")
	}
	if len(runner.calls) != 2 || runner.calls[1][len(runner.calls[1])-1] != "--remove-orphans" {
		t.Fatalf("cleanup calls = %v", runner.calls)
	}
}
