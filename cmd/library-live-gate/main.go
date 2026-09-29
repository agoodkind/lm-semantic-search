// Command library-live-gate runs one shared search library lane suite and
// fails unless every selected test ran and passed. It runs
// "go test -json -count=1 -timeout TIMEOUT -tags TAGS -run PATTERN PACKAGE",
// decodes the test events, and rejects a failing command, a go test timeout,
// zero selected tests, any skipped test, and a run with no passing test. A
// package-level pass event does not count as a selected test.
//
// Usage:
//
//	library-live-gate [-timeout 30m] -tags live -run '^TestLibraryWrite' ./test/live/
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"time"
)

const (
	exitPassed       = 0
	exitGateFailed   = 1
	exitUsage        = 2
	exitInterrupted  = 130
	maxEventBytes    = 16 << 20
	testActionRun    = "run"
	testActionPass   = "pass"
	testActionFail   = "fail"
	testActionSkip   = "skip"
	testActionOutput = "output"
	// timeoutPanicPrefix starts the line that the go test binary prints when
	// its -timeout expires.
	timeoutPanicPrefix = "panic: test timed out after"
)

// defaultTestTimeout is the go test -timeout of one lane run. Go's own default
// of 10 minutes ended a contended make library-live-l1 run at 2026-09-29 05:54
// UTC, while baseline ingestion used the embedding endpoint. The longest
// complete contended run, make library-live-l1 at 05:49:58 to 05:56:52 UTC,
// took 6 minutes 54 seconds; an uncontended run at 02:59:55 to 03:02:38 UTC
// took 2 minutes 43 seconds. Thirty minutes is more than four times the
// longest complete contended run.
const defaultTestTimeout = 30 * time.Minute

// testEvent is one line of "go test -json" output.
type testEvent struct {
	Action  string `json:"Action"`
	Package string `json:"Package"`
	Test    string `json:"Test"`
	Output  string `json:"Output"`
}

// testOutcome records one selected test and its output lines.
type testOutcome struct {
	started bool
	result  string
	output  strings.Builder
}

// gateReport summarizes the selected tests of one run.
type gateReport struct {
	outcomes map[string]*testOutcome
	order    []string
	// timeout is the go test timeout panic line, or empty when the run did not
	// time out.
	timeout string
}

func main() {
	slog.Info("library_live_gate.invoked")
	os.Exit(realMain())
}

func realMain() int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return run(ctx, os.Args[1:], os.Stdout, os.Stderr)
}

// run executes the gate and returns the process exit status.
func run(ctx context.Context, arguments []string, stdout io.Writer, stderr io.Writer) int {
	flags := flag.NewFlagSet("library-live-gate", flag.ContinueOnError)
	flags.SetOutput(stderr)
	tags := flags.String("tags", "", "build tags for go test")
	pattern := flags.String("run", "", "go test -run pattern that selects the lane tests")
	timeout := flags.Duration("timeout", defaultTestTimeout, "go test -timeout of the whole run")
	if err := flags.Parse(arguments); err != nil {
		return exitUsage
	}
	if *pattern == "" || flags.NArg() != 1 || *timeout <= 0 {
		writeLine(stderr, "library-live-gate: usage: library-live-gate [-timeout DURATION] [-tags TAGS] -run PATTERN PACKAGE")
		return exitUsage
	}
	packagePath := flags.Arg(0)

	goArguments := []string{"test", "-json", "-count=1", "-timeout", timeout.String()}
	if *tags != "" {
		goArguments = append(goArguments, "-tags", *tags)
	}
	goArguments = append(goArguments, "-run", *pattern, packagePath)
	writeLine(stderr, "library-live-gate: go "+strings.Join(goArguments, " "))

	command := exec.CommandContext(ctx, "go", goArguments...)
	command.Stderr = stderr
	events, err := command.StdoutPipe()
	if err != nil {
		slog.ErrorContext(ctx, "library_live_gate.stdout_pipe_failed", "err", err)
		writeLine(stderr, fmt.Sprintf("library-live-gate: open go test output: %v", err))
		return exitGateFailed
	}
	if err := command.Start(); err != nil {
		slog.ErrorContext(ctx, "library_live_gate.start_failed", "err", err)
		writeLine(stderr, fmt.Sprintf("library-live-gate: start go test: %v", err))
		return exitGateFailed
	}
	report, decodeErr := decodeEvents(events, stdout)
	waitErr := command.Wait()

	if ctx.Err() != nil {
		writeLine(stderr, "library-live-gate: interrupted")
		return exitInterrupted
	}
	if decodeErr != nil {
		slog.ErrorContext(ctx, "library_live_gate.decode_failed", "err", decodeErr)
		writeLine(stderr, fmt.Sprintf("library-live-gate: decode go test events: %v", decodeErr))
		return exitGateFailed
	}
	commandStatus := exitPassed
	if waitErr != nil {
		var exitErr *exec.ExitError
		if !errors.As(waitErr, &exitErr) {
			slog.ErrorContext(ctx, "library_live_gate.wait_failed", "err", waitErr)
			writeLine(stderr, fmt.Sprintf("library-live-gate: run go test: %v", waitErr))
			return exitGateFailed
		}
		commandStatus = exitErr.ExitCode()
	}
	return judge(report, commandStatus, *pattern, packagePath, stderr)
}

// decodeEvents reads test events, copies each output line to stdout, and
// records the final action of every selected test.
func decodeEvents(events io.Reader, stdout io.Writer) (gateReport, error) {
	report := gateReport{outcomes: make(map[string]*testOutcome), order: nil, timeout: ""}
	scanner := bufio.NewScanner(events)
	scanner.Buffer(make([]byte, 0, 64<<10), maxEventBytes)
	for scanner.Scan() {
		line := scanner.Bytes()
		var event testEvent
		if err := json.Unmarshal(line, &event); err != nil {
			// go test prints build failures as plain text on stdout.
			writeLine(stdout, string(line))
			continue
		}
		if event.Action == testActionOutput {
			_, _ = io.WriteString(stdout, event.Output)
			if report.timeout == "" && strings.HasPrefix(event.Output, timeoutPanicPrefix) {
				report.timeout = strings.TrimSpace(event.Output)
			}
		}
		if event.Test == "" {
			continue
		}
		outcome, found := report.outcomes[event.Test]
		if !found {
			outcome = &testOutcome{started: false, result: "", output: strings.Builder{}}
			report.outcomes[event.Test] = outcome
			report.order = append(report.order, event.Test)
		}
		switch event.Action {
		case testActionRun:
			outcome.started = true
		case testActionPass, testActionFail, testActionSkip:
			outcome.result = event.Action
		case testActionOutput:
			outcome.output.WriteString(event.Output)
		}
	}
	if err := scanner.Err(); err != nil {
		slog.Error("library_live_gate.read_events_failed", "err", err)
		return report, fmt.Errorf("read go test events: %w", err)
	}
	return report, nil
}

// judge applies the gate rules and writes one verdict line with diagnostics.
func judge(report gateReport, commandStatus int, pattern string, packagePath string, stderr io.Writer) int {
	var passed, failed, skipped, unfinished []string
	for _, name := range report.order {
		outcome := report.outcomes[name]
		if !outcome.started {
			continue
		}
		switch outcome.result {
		case testActionPass:
			passed = append(passed, name)
		case testActionFail:
			failed = append(failed, name)
		case testActionSkip:
			skipped = append(skipped, name)
		default:
			unfinished = append(unfinished, name)
		}
	}
	sort.Strings(failed)
	sort.Strings(skipped)
	sort.Strings(unfinished)
	summary := fmt.Sprintf(
		"%d selected, %d passed, %d failed, %d skipped, %d unfinished",
		len(passed)+len(failed)+len(skipped)+len(unfinished),
		len(passed),
		len(failed),
		len(skipped),
		len(unfinished),
	)

	var problems []string
	if report.timeout != "" {
		problems = append(problems, fmt.Sprintf("go test timed out (%s)", report.timeout))
	}
	if commandStatus != exitPassed {
		problems = append(problems, fmt.Sprintf("go test exited with status %d", commandStatus))
	}
	if len(failed) > 0 {
		problems = append(problems, "failed tests: "+strings.Join(failed, ", "))
	}
	if len(skipped) > 0 {
		problems = append(problems, "skipped tests: "+strings.Join(skipped, ", "))
		for _, name := range skipped {
			writeLine(stderr, fmt.Sprintf("library-live-gate: skip output for %s:\n%s", name, report.outcomes[name].output.String()))
		}
	}
	if len(unfinished) > 0 {
		problems = append(problems, "unfinished tests: "+strings.Join(unfinished, ", "))
	}
	if len(passed)+len(failed)+len(skipped)+len(unfinished) == 0 {
		problems = append(problems, fmt.Sprintf("no selected test ran for -run %q in %s", pattern, packagePath))
	} else if len(passed) == 0 {
		problems = append(problems, "no selected test passed")
	}

	if len(problems) == 0 {
		writeLine(stderr, "library-live-gate: PASS: "+summary)
		return exitPassed
	}
	writeLine(stderr, "library-live-gate: FAIL: "+summary+"; "+strings.Join(problems, "; "))
	if commandStatus != exitPassed {
		return commandStatus
	}
	return exitGateFailed
}

func writeLine(writer io.Writer, line string) {
	_, _ = io.WriteString(writer, line+"\n")
}
