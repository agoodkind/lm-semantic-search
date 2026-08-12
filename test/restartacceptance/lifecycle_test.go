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
	wantEvents := []string{"validate-production", "prepare", "capture-baseline"}
	for _, name := range acceptanceScenarioNames {
		wantEvents = append(wantEvents, "capture-"+name, "case-"+name, "capture-"+name+"-after", "audit-"+name)
	}
	wantEvents = append(wantEvents, "production-confirmation", "capture-final", "audit-final", "cleanup", "finish-passed")
	if !slices.Equal(operations.events, wantEvents) {
		t.Fatalf("lifecycle events = %v, want %v", operations.events, wantEvents)
	}
	if operations.finished.RunID != operations.run.ID || operations.finished.Status != "passed" || operations.finished.Error != "" {
		t.Fatalf("terminal result = %+v", operations.finished)
	}
	if len(operations.caseTokens) != len(acceptanceScenarioNames) {
		t.Fatalf("case tokens = %d, want %d", len(operations.caseTokens), len(acceptanceScenarioNames))
	}
	for index, token := range operations.caseTokens {
		wantHash := "capture-" + acceptanceScenarioNames[index]
		if token.ContentHash != wantHash {
			t.Fatalf("case %s token = %q, want freshly captured %q", acceptanceScenarioNames[index], token.ContentHash, wantHash)
		}
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
		"capture-baseline",
		"capture-a",
		"case-a",
		"capture-a-after",
		"audit-a",
		"capture-h",
		"case-h",
		"capture-h-after",
		"audit-h",
		"production-confirmation",
		"capture-final",
		"audit-final",
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

func TestExecuteRestartAcceptanceRejectsActiveProductionBeforePrepare(t *testing.T) {
	operations := newLifecycleOperationsRecorder()
	operations.failEvent = "validate-production"
	err := executeRestartAcceptance(context.Background(), operations.operations())
	if err == nil || !strings.Contains(err.Error(), "injected validate-production") {
		t.Fatalf("execute error = %v", err)
	}
	if !slices.Equal(operations.events, []string{"validate-production"}) {
		t.Fatalf("lifecycle events = %v, want only initial validation", operations.events)
	}
	if operations.finished.Status != "" {
		t.Fatalf("unexpected terminal result = %+v", operations.finished)
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

func TestValidateRestartAcceptanceConfirmationsRequiresBothExactValues(t *testing.T) {
	tests := []struct {
		restart    string
		production string
		wantErr    bool
	}{
		{wantErr: true},
		{restart: restartAcceptanceConfirmation, wantErr: true},
		{restart: restartAcceptanceConfirmation, production: "other", wantErr: true},
		{restart: restartAcceptanceConfirmation, production: "default"},
	}
	for _, test := range tests {
		err := validateRestartAcceptanceConfirmations(test.restart, test.production)
		if (err != nil) != test.wantErr {
			t.Fatalf("confirmations restart=%q production=%q error=%v, want error=%t", test.restart, test.production, err, test.wantErr)
		}
	}
}

func TestRunProductionConfirmationUsesOnlyReadOnlyProductionTestsAndExactOptIn(t *testing.T) {
	runner := &recordingRunner{outputs: [][]byte{[]byte("ok")}}
	if err := runProductionConfirmation(context.Background(), runner); err != nil {
		t.Fatalf("run production confirmation: %v", err)
	}
	wantCall := []string{
		"go", "test", "-tags=live production", "-count=1",
		"-run=^(TestValidateProductionOptIn|TestProductionResidencyConfiguration|TestProductionReadOnlySearchConfirmation)$",
		"./test/live/",
	}
	if !slices.Equal(runner.calls[0], wantCall) {
		t.Fatalf("production confirmation call = %v, want %v", runner.calls[0], wantCall)
	}
	if runner.environments[0][productionDatabaseConfirmation] != "default" {
		t.Fatalf("production confirmation environment = %v", runner.environments[0])
	}
}

func TestRunProductionConfirmationFailsLoudly(t *testing.T) {
	runner := &recordingRunner{failAt: 1}
	if err := runProductionConfirmation(context.Background(), runner); err == nil {
		t.Fatal("failed production confirmation was accepted")
	}
}

type lifecycleOperationsRecorder struct {
	run             acceptanceRun
	events          []string
	caseTokens      []inventoryToken
	finished        acceptanceResult
	failEvent       string
	cleanupErr      error
	finishErr       error
	captures        int
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
		ValidateProduction: recorder.validateProduction,
		Prepare:            recorder.prepare,
		CaptureProduction:  recorder.capture,
		RunCase:            recorder.runCase,
		ConfirmProduction:  recorder.confirm,
		AuditProduction:    recorder.audit,
		Cleanup:            recorder.cleanup,
		Finish:             recorder.finish,
	}
}

func (recorder *lifecycleOperationsRecorder) validateProduction(context.Context) error {
	event := "validate-production"
	recorder.events = append(recorder.events, event)
	if recorder.failEvent == event {
		return errors.New("injected " + event)
	}
	return nil
}

func (recorder *lifecycleOperationsRecorder) prepare(context.Context) (acceptanceRun, error) {
	recorder.events = append(recorder.events, "prepare")
	return recorder.run, nil
}

func (recorder *lifecycleOperationsRecorder) capture(context.Context, acceptanceRun) (inventoryToken, error) {
	recorder.captures++
	label := "baseline"
	if recorder.captures > 1 && recorder.captures < len(acceptanceScenarioNames)*2+2 {
		position := recorder.captures - 2
		label = acceptanceScenarioNames[position/2]
		if position%2 == 1 {
			label += "-after"
		}
	}
	if recorder.captures == len(acceptanceScenarioNames)*2+2 {
		label = "final"
	}
	event := "capture-" + label
	recorder.events = append(recorder.events, event)
	if recorder.failEvent == event {
		return inventoryToken{}, errors.New("injected " + event)
	}
	return inventoryToken{
		RunID:       recorder.run.ID,
		CapturedAt:  time.Date(2026, 8, 12, 1, 2, int(recorder.captures), 0, time.UTC),
		ContentHash: event,
		Inventory: productionInventory{
			Databases:   []string{"default"},
			Collections: collectionCensus{{Database: "default", Collection: "operator"}: "hash"},
		},
	}, nil
}

func (recorder *lifecycleOperationsRecorder) runCase(_ context.Context, _ acceptanceRun, name string, token inventoryToken) error {
	event := "case-" + name
	recorder.events = append(recorder.events, event)
	recorder.caseTokens = append(recorder.caseTokens, token)
	if recorder.failEvent == event {
		return errors.New("injected " + event)
	}
	return nil
}

func (recorder *lifecycleOperationsRecorder) confirm(context.Context, acceptanceRun) error {
	event := "production-confirmation"
	recorder.events = append(recorder.events, event)
	if recorder.failEvent == event {
		return errors.New("injected " + event)
	}
	return nil
}

func (recorder *lifecycleOperationsRecorder) audit(_ context.Context, before inventoryToken, after inventoryToken) error {
	label := strings.TrimPrefix(after.ContentHash, "capture-")
	if label == "final" {
		if before.ContentHash != "capture-baseline" {
			return errors.New("wrong final audit baseline")
		}
	} else {
		name := strings.TrimSuffix(label, "-after")
		if label == name || before.ContentHash != "capture-"+name {
			return errors.New("wrong case audit endpoints")
		}
		label = name
	}
	event := "audit-" + label
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
