//go:build restartacceptance

package restartacceptance

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"
	"time"
)

const restartAcceptanceCleanupTimeout = 2 * time.Minute

var acceptanceScenarioNames = []string{"a", "b", "c", "d", "e", "f", "g", "h"}

type acceptanceLifecycleOperations struct {
	Prepare           func(context.Context) (acceptanceRun, error)
	CaptureProduction func(context.Context, acceptanceRun) (inventoryToken, error)
	RunCase           func(context.Context, acceptanceRun, string, inventoryToken) error
	ConfirmProduction func(context.Context, acceptanceRun) error
	AuditProduction   func(context.Context, inventoryToken, inventoryToken) error
	Cleanup           func(context.Context, acceptanceRun) error
	Finish            func(acceptanceRun, acceptanceResult) error
}

func executeRestartAcceptance(ctx context.Context, operations acceptanceLifecycleOperations) (runErr error) {
	if err := validateLifecycleOperations(operations); err != nil {
		return err
	}
	run, err := operations.Prepare(ctx)
	if err != nil {
		return fmt.Errorf("prepare restart acceptance: %w", err)
	}
	defer func() {
		cleanupContext, cancelCleanup := context.WithTimeout(context.Background(), restartAcceptanceCleanupTimeout)
		cleanupErr := operations.Cleanup(cleanupContext, run)
		cancelCleanup()
		if cleanupErr != nil {
			runErr = errors.Join(runErr, fmt.Errorf("cleanup restart acceptance: %w", cleanupErr))
		}
		result := acceptanceResult{RunID: run.ID, Status: "passed"}
		if runErr != nil {
			result.Status = "failed"
			result.Error = runErr.Error()
		}
		if finishErr := operations.Finish(run, result); finishErr != nil {
			runErr = errors.Join(runErr, fmt.Errorf("write restart acceptance artifact: %w", finishErr))
		}
	}()

	baseline, err := operations.CaptureProduction(ctx, run)
	if err != nil {
		runErr = fmt.Errorf("capture baseline production inventory: %w", err)
		return runErr
	}
	for _, name := range acceptanceScenarioNames {
		caseInventory, captureErr := operations.CaptureProduction(ctx, run)
		if captureErr != nil {
			runErr = fmt.Errorf("refresh production inventory for scenario %s: %w", name, captureErr)
			return runErr
		}
		if caseErr := operations.RunCase(ctx, run, name, caseInventory); caseErr != nil {
			runErr = fmt.Errorf("scenario %s: %w", name, caseErr)
			return runErr
		}
		afterCase, captureErr := operations.CaptureProduction(ctx, run)
		if captureErr != nil {
			runErr = fmt.Errorf("capture production inventory after scenario %s: %w", name, captureErr)
			return runErr
		}
		if auditErr := operations.AuditProduction(ctx, caseInventory, afterCase); auditErr != nil {
			runErr = fmt.Errorf("audit production after scenario %s: %w", name, auditErr)
			return runErr
		}
	}
	if err := operations.ConfirmProduction(ctx, run); err != nil {
		runErr = fmt.Errorf("production confirmation: %w", err)
		return runErr
	}
	after, err := operations.CaptureProduction(ctx, run)
	if err != nil {
		runErr = fmt.Errorf("capture final production inventory: %w", err)
		return runErr
	}
	if err := operations.AuditProduction(ctx, baseline, after); err != nil {
		runErr = fmt.Errorf("audit production baseline: %w", err)
		return runErr
	}
	return nil
}

func validateLifecycleOperations(operations acceptanceLifecycleOperations) error {
	if operations.Prepare == nil || operations.CaptureProduction == nil || operations.RunCase == nil ||
		operations.ConfirmProduction == nil || operations.AuditProduction == nil ||
		operations.Cleanup == nil || operations.Finish == nil {
		return fmt.Errorf("restart acceptance lifecycle operations are incomplete")
	}
	return nil
}

func validateRestartAcceptanceConfirmations(restartConfirmation string, productionConfirmation string) error {
	if err := validateOptIn(restartConfirmation); err != nil {
		return err
	}
	if productionConfirmation != "default" {
		return fmt.Errorf("%s must equal %q", productionDatabaseConfirmation, "default")
	}
	return nil
}

func runProductionConfirmation(ctx context.Context, runner commandRunner) error {
	if runner == nil {
		return fmt.Errorf("production confirmation requires a command runner")
	}
	environment := map[string]string{productionDatabaseConfirmation: "default"}
	arguments := []string{
		"test",
		"-tags=live production",
		"-count=1",
		"-run=^(TestValidateProductionOptIn|TestProductionResidencyConfiguration|TestProductionReadOnlySearchConfirmation)$",
		"./test/live/",
	}
	if _, err := runner.Run(ctx, environment, "go", arguments...); err != nil {
		return fmt.Errorf("run read-only live production tests: %w", err)
	}
	return nil
}

type execCommandRunner struct{}

func (execCommandRunner) Run(
	ctx context.Context,
	environment map[string]string,
	name string,
	arguments ...string,
) ([]byte, error) {
	command := exec.CommandContext(ctx, name, arguments...)
	command.Env = mergeEnvironment(os.Environ(), environment)
	output, err := command.CombinedOutput()
	if err != nil {
		message := strings.TrimSpace(string(output))
		if message == "" {
			return nil, fmt.Errorf("run %s: %w", name, err)
		}
		return nil, fmt.Errorf("run %s: %w; output: %s", name, err, message)
	}
	return output, nil
}

func mergeEnvironment(base []string, overrides map[string]string) []string {
	values := make(map[string]string, len(base)+len(overrides))
	for _, entry := range base {
		key, value, found := strings.Cut(entry, "=")
		if found {
			values[key] = value
		}
	}
	for key, value := range overrides {
		values[key] = value
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := make([]string, 0, len(keys))
	for _, key := range keys {
		result = append(result, key+"="+values[key])
	}
	return result
}
