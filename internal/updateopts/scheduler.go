package updateopts

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"goodkind.io/go-makefile/selfupdate"
	"goodkind.io/lm-semantic-search/internal/clock"
)

const updateInitialDelay = time.Minute

var (
	nextUpdateDelayFunc   = nextUpdateDelay
	runScheduledApplyFunc = runScheduledApply
)

// RunApplyScheduler runs the daemon-owned multi-binary update loop.
func RunApplyScheduler(ctx context.Context, overrides Overrides, stopForRelaunch func()) {
	log := overrides.Log
	if log == nil {
		log = slog.Default()
	}

	for {
		delay := nextUpdateDelayFunc(overrides)
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}

		applied, err := runScheduledApplyIteration(ctx, overrides, log)
		if err != nil {
			log.WarnContext(ctx, "scheduled update failed", "err", err)
			continue
		}
		if applied {
			if stopForRelaunch != nil {
				stopForRelaunch()
			}
			return
		}
	}
}

func runScheduledApplyIteration(ctx context.Context, overrides Overrides, log *slog.Logger) (applied bool, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			log.ErrorContext(ctx, "update scheduler panic", "err", recovered)
			applied = false
			err = nil
		}
	}()
	return runScheduledApplyFunc(ctx, overrides, log)
}

func nextUpdateDelay(overrides Overrides) time.Duration {
	option, err := CheckOptions(overrides)
	if err != nil {
		return updateInitialDelay
	}
	state, err := selfupdate.LoadState(option.StatePath)
	if err == nil && !state.NextCheckAt.IsZero() {
		delay := state.NextCheckAt.Sub(clock.Now())
		if delay > 0 {
			return delay
		}
	}
	return updateInitialDelay
}

func runScheduledApply(ctx context.Context, overrides Overrides, log *slog.Logger) (bool, error) {
	option, err := CheckOptions(overrides)
	if err != nil {
		log.WarnContext(ctx, "scheduled update options failed", "err", err)
		return false, err
	}
	option.Log = log.With(slog.String("component", "update"))
	if versionSkipsScheduledApply(option.Config.CurrentVersion) {
		_, checkErr := selfupdate.Check(ctx, option)
		if checkErr != nil {
			log.WarnContext(ctx, "scheduled update fallback check failed", "err", checkErr)
			return false, fmt.Errorf("scheduled update fallback check: %w", checkErr)
		}
		return false, nil
	}

	overrides.Log = option.Log
	result, err := ApplyAll(ctx, overrides)
	if err != nil {
		log.WarnContext(ctx, "scheduled update apply failed", "err", err)
		return false, fmt.Errorf("scheduled update apply: %w", err)
	}
	return result.Applied, nil
}

func versionSkipsScheduledApply(currentVersion string) bool {
	return currentVersion == "" || currentVersion == "dev" || currentVersion == "unknown"
}
