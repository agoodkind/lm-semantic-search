//go:build restartacceptance

package restartacceptance

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestExecuteRestartAcceptanceRunsOneCompleteLifecycle(t *testing.T) {
	operations := newLifecycleOperationsRecorder()
	if err := executeRestartAcceptance(context.Background(), operations.operations()); err != nil {
		t.Fatalf("execute restart acceptance: %v", err)
	}
	wantEvents := []string{"prepare"}
	for _, name := range acceptanceScenarioNames {
		wantEvents = append(wantEvents, "case-"+name)
	}
	wantEvents = append(wantEvents, "clone-confirmation", "cleanup", "finish-passed")
	if !slices.Equal(operations.events, wantEvents) {
		t.Fatalf("lifecycle events = %v, want %v", operations.events, wantEvents)
	}
	if operations.finished.RunID != operations.run.ID || operations.finished.Status != "passed" || operations.finished.Error != "" {
		t.Fatalf("terminal result = %+v", operations.finished)
	}
	if operations.cleanupDeadline.IsZero() {
		t.Fatal("cleanup context has no deadline")
	}
	cleanupBudget := time.Until(operations.cleanupDeadline)
	if cleanupBudget <= 119*time.Second || cleanupBudget > 2*time.Minute {
		t.Fatalf("cleanup deadline budget = %s, want fixed two-minute bound", cleanupBudget)
	}
}

func TestExecuteRestartAcceptanceRecordsFailureAndCleansEveryStartedStage(t *testing.T) {
	failureEvents := []string{
		"case-a",
		"case-h",
		"clone-confirmation",
	}
	for _, failureEvent := range failureEvents {
		t.Run(failureEvent, func(t *testing.T) {
			operations := newLifecycleOperationsRecorder()
			operations.failEvent = failureEvent
			err := executeRestartAcceptance(context.Background(), operations.operations())
			if err == nil || !strings.Contains(err.Error(), "injected "+failureEvent) {
				t.Fatalf("execute error = %v", err)
			}
			if !slices.Contains(operations.events, "cleanup") {
				t.Fatalf("cleanup missing from events %v", operations.events)
			}
			if operations.finished.Status != "failed" || !strings.Contains(operations.finished.Error, "injected "+failureEvent) {
				t.Fatalf("terminal result = %+v", operations.finished)
			}
		})
	}
}

func TestExecuteRestartAcceptanceJoinsCleanupAndArtifactFailures(t *testing.T) {
	operations := newLifecycleOperationsRecorder()
	operations.cleanupErr = errors.New("cleanup failed")
	operations.finishErr = errors.New("artifact failed")
	err := executeRestartAcceptance(context.Background(), operations.operations())
	if err == nil || !strings.Contains(err.Error(), "cleanup failed") || !strings.Contains(err.Error(), "artifact failed") {
		t.Fatalf("execute error = %v", err)
	}
	if operations.finished.Status != "failed" || !strings.Contains(operations.finished.Error, "cleanup failed") {
		t.Fatalf("terminal result = %+v", operations.finished)
	}
}

type lifecycleOperationsRecorder struct {
	run             acceptanceRun
	events          []string
	finished        acceptanceResult
	failEvent       string
	cleanupErr      error
	finishErr       error
	cleanupDeadline time.Time
}

func newLifecycleOperationsRecorder() *lifecycleOperationsRecorder {
	return &lifecycleOperationsRecorder{
		run: acceptanceRun{
			ID:    "20260812T010203Z-abcdef01",
			Paths: pathsForRun("/tmp/20260812T010203Z-abcdef01"),
		},
	}
}

func (recorder *lifecycleOperationsRecorder) operations() acceptanceLifecycleOperations {
	return acceptanceLifecycleOperations{
		Prepare:      recorder.prepare,
		RunCase:      recorder.runCase,
		ConfirmClone: recorder.confirm,
		Cleanup:      recorder.cleanup,
		Finish:       recorder.finish,
	}
}

func (recorder *lifecycleOperationsRecorder) prepare(context.Context) (acceptanceRun, error) {
	recorder.events = append(recorder.events, "prepare")
	return recorder.run, nil
}

func (recorder *lifecycleOperationsRecorder) runCase(_ context.Context, _ acceptanceRun, name string) error {
	event := "case-" + name
	recorder.events = append(recorder.events, event)
	if recorder.failEvent == event {
		return errors.New("injected " + event)
	}
	return nil
}

func (recorder *lifecycleOperationsRecorder) confirm(context.Context, acceptanceRun) error {
	event := "clone-confirmation"
	recorder.events = append(recorder.events, event)
	if recorder.failEvent == event {
		return errors.New("injected " + event)
	}
	return nil
}

func (recorder *lifecycleOperationsRecorder) cleanup(ctx context.Context, _ acceptanceRun) error {
	recorder.events = append(recorder.events, "cleanup")
	recorder.cleanupDeadline, _ = ctx.Deadline()
	return recorder.cleanupErr
}

func (recorder *lifecycleOperationsRecorder) finish(_ acceptanceRun, result acceptanceResult) error {
	recorder.finished = result
	recorder.events = append(recorder.events, "finish-"+result.Status)
	return recorder.finishErr
}
