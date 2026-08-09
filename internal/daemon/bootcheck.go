package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"goodkind.io/gklog/correlation"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const (
	// defaultBootSelfCheckDelay is the pause between startup and the self-check.
	// It lets the store client finish its boot dial or first reconnect, so a
	// healthy daemon is not reported unusable purely because the check raced
	// startup, while still surfacing a bad restore within seconds rather than at
	// the first user query hours later.
	defaultBootSelfCheckDelay = 3 * time.Second
	// bootSelfCheckTimeout bounds a remote-backend check. Local stores are
	// skipped because reads may perform crash recovery, and ONNX inference is
	// skipped because its native call cannot guarantee context cancellation.
	bootSelfCheckTimeout = 20 * time.Second
	// bootSelfCheckMaxAttempts gives startup races two chances to recover after
	// the first probe. Immediate failures terminate in about nine seconds with
	// the default delay, while three hung probes still terminate in under
	// seventy seconds instead of becoming a daemon-lifetime retry loop.
	bootSelfCheckMaxAttempts = 3
	// bootSelfCheckQuery is the fixed probe text. Its wording is irrelevant to the
	// verdict: a nearest-neighbour search answers for any text, so the check reads
	// the absence of an error, never the contents or the count of the results.
	bootSelfCheckQuery = "daemon boot self check"
	// bootSelfCheckLimit keeps the probe to a single row, which is the smallest
	// query that still exercises the whole read path.
	bootSelfCheckLimit = 1
)

// bootSelfCheckOutcome names what one boot self-check concluded. Skipped is a
// clean result, not a failure: a daemon with nothing indexed yet has no data to
// prove usable.
type bootSelfCheckOutcome string

const (
	bootSelfCheckPassed  bootSelfCheckOutcome = "passed"
	bootSelfCheckSkipped bootSelfCheckOutcome = "skipped"
	bootSelfCheckFailed  bootSelfCheckOutcome = "failed"
	// A superseded pass completed after newer dependency-failure evidence. It
	// leaves readiness degraded and schedules another attempt.
	bootSelfCheckSuperseded bootSelfCheckOutcome = "superseded"
)

type bootSelfCheckExhaustedError struct {
	Attempts int
	Outcome  bootSelfCheckOutcome
	Cause    error
}

func (err *bootSelfCheckExhaustedError) Error() string {
	if err.Cause != nil {
		return fmt.Sprintf(
			"boot self-check exhausted %d attempts with outcome %s: %v",
			err.Attempts,
			err.Outcome,
			err.Cause,
		)
	}
	return fmt.Sprintf(
		"boot self-check exhausted %d attempts with outcome %s",
		err.Attempts,
		err.Outcome,
	)
}

func (err *bootSelfCheckExhaustedError) Unwrap() error {
	return err.Cause
}

// StartBootSelfCheck launches the boot self-check and returns
// immediately, so startup never waits on it and the daemon serves regardless of
// the outcome. Remote checks run under their own deadline and make at most three
// attempts, stopping sooner when one passes, skips, or the runtime context ends.
func (manager *Manager) StartBootSelfCheck(ctx context.Context) {
	delay := manager.bootSelfCheckDelay
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(
					ctx, "daemon.selfcheck.panic",
					"component", "daemon",
					"subcomponent", "selfcheck",
					"err", fmt.Errorf("panic: %v", recovered),
				)
			}
		}()

		if !waitForBootSelfCheck(ctx, delay) {
			return
		}

		checkCtx := correlation.WithContext(ctx, correlation.New("").WithIdentityAttributes(
			correlation.IdentityAttribute{Key: "origin", Value: "boot-selfcheck"},
		))
		_, _ = manager.runBootSelfCheckSequence(checkCtx, delay)
	}()
}

func (manager *Manager) runBootSelfCheckSequence(
	ctx context.Context,
	retryDelay time.Duration,
) (bootSelfCheckOutcome, error) {
	outcome := bootSelfCheckSkipped
	var lastErr error
	for attempt := 1; attempt <= bootSelfCheckMaxAttempts; attempt++ {
		boundedCtx, cancel := context.WithTimeout(ctx, bootSelfCheckTimeout)
		outcome, lastErr = manager.runBootSelfCheck(boundedCtx)
		cancel()
		if outcome == bootSelfCheckPassed || outcome == bootSelfCheckSkipped {
			return outcome, nil
		}
		if ctx.Err() != nil {
			contextErr := ctx.Err()
			slog.ErrorContext(
				ctx, "daemon.selfcheck.cancelled",
				"component", "daemon",
				"subcomponent", "selfcheck",
				"outcome", outcome,
				"err", contextErr,
			)
			return outcome, fmt.Errorf("boot self-check sequence cancelled: %w", contextErr)
		}
		if attempt < bootSelfCheckMaxAttempts &&
			!waitForBootSelfCheck(ctx, retryDelay) {
			contextErr := ctx.Err()
			slog.ErrorContext(
				ctx, "daemon.selfcheck.cancelled",
				"component", "daemon",
				"subcomponent", "selfcheck",
				"outcome", outcome,
				"err", contextErr,
			)
			return outcome, fmt.Errorf("boot self-check retry wait cancelled: %w", contextErr)
		}
	}

	exhaustedErr := &bootSelfCheckExhaustedError{
		Attempts: bootSelfCheckMaxAttempts,
		Outcome:  outcome,
		Cause:    lastErr,
	}
	slog.ErrorContext(
		ctx, "daemon.selfcheck.exhausted",
		"component", "daemon",
		"subcomponent", "selfcheck",
		"attempts", bootSelfCheckMaxAttempts,
		"outcome", outcome,
		"err", exhaustedErr,
	)
	return outcome, exhaustedErr
}

func waitForBootSelfCheck(ctx context.Context, delay time.Duration) bool {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// runBootSelfCheck proves the remote search path end to end once, by running a
// real one-row query against a collection the registry says is indexed. It
// skips the local backend and every ONNX embedder because their native inference
// boundary cannot guarantee cancellation and local collection loading may
// perform crash recovery.
//
// The remote query is read-only and idempotent. It deletes nothing and triggers
// no indexing.
//
// The outcome feeds the same shared-dependency health record that job and search
// outcomes feed, so a failure raises the existing status banner rather than
// introducing a second notion of health. A pass clears a boot-time banner only
// when no newer dependency failure arrived while the query was in flight.
func (manager *Manager) runBootSelfCheck(ctx context.Context) (bootSelfCheckOutcome, error) {
	if reason := manager.bootSelfCheckSkipReason(); reason != "" {
		slog.InfoContext(
			ctx, "daemon.selfcheck.skipped",
			"component", "daemon",
			"subcomponent", "selfcheck",
			"reason", reason,
		)
		return bootSelfCheckSkipped, nil
	}
	if manager.semantic == nil {
		slog.InfoContext(
			ctx, "daemon.selfcheck.skipped",
			"component", "daemon",
			"subcomponent", "selfcheck",
			"reason", "no vector backend configured",
		)
		return bootSelfCheckSkipped, nil
	}
	if !manager.semantic.Available() {
		manager.noteDependencyFailure(semantic.ErrUnavailable)
		slog.ErrorContext(
			ctx, "daemon.selfcheck.failed",
			"component", "daemon",
			"subcomponent", "selfcheck",
			"reason", "vector store is not connected",
			"err", semantic.ErrUnavailable,
		)
		return bootSelfCheckFailed, semantic.ErrUnavailable
	}

	codebase, found := manager.bootSelfCheckTarget()
	if !found {
		slog.InfoContext(
			ctx, "daemon.selfcheck.skipped",
			"component", "daemon",
			"subcomponent", "selfcheck",
			"reason", "no indexed codebase to query",
		)
		return bootSelfCheckSkipped, nil
	}

	failureGeneration := manager.dependencyFailureGenerationNow()
	startedAt := clock.Now()
	err := manager.runBootSelfCheckQuery(ctx, codebase.CanonicalPath)
	durationMS := clock.Now().Sub(startedAt).Milliseconds()
	if err != nil {
		manager.noteDependencyFailure(err)
		checkErr := fmt.Errorf("boot self-check query against %s: %w", codebase.CanonicalPath, err)
		slog.ErrorContext(
			ctx, "daemon.selfcheck.failed",
			"component", "daemon",
			"subcomponent", "selfcheck",
			"codebase_id", codebase.ID,
			"path", codebase.CanonicalPath,
			"duration_ms", durationMS,
			"err", checkErr,
		)
		return bootSelfCheckFailed, checkErr
	}

	if !manager.noteDependencyHealthyIfGeneration(failureGeneration) {
		slog.WarnContext(
			ctx, "daemon.selfcheck.superseded",
			"component", "daemon",
			"subcomponent", "selfcheck",
			"codebase_id", codebase.ID,
			"path", codebase.CanonicalPath,
			"duration_ms", durationMS,
		)
		return bootSelfCheckSuperseded, nil
	}
	slog.InfoContext(
		ctx, "daemon.selfcheck.passed",
		"component", "daemon",
		"subcomponent", "selfcheck",
		"codebase_id", codebase.ID,
		"path", codebase.CanonicalPath,
		"duration_ms", durationMS,
	)
	return bootSelfCheckPassed, nil
}

func (manager *Manager) runBootSelfCheckQuery(ctx context.Context, codebasePath string) error {
	collectionName := manager.semantic.CollectionName(codebasePath)
	if err := manager.semantic.PrepareCollection(ctx, collectionName); err != nil {
		wrappedErr := fmt.Errorf(
			"prepare boot self-check collection %s: %w",
			collectionName,
			err,
		)
		slog.WarnContext(ctx, "daemon.selfcheck.prepare_failed", "err", wrappedErr)
		return wrappedErr
	}
	lease, err := manager.semantic.AcquireCollection(ctx, collectionName)
	if err != nil {
		wrappedErr := fmt.Errorf(
			"acquire boot self-check collection %s: %w",
			collectionName,
			err,
		)
		slog.WarnContext(ctx, "daemon.selfcheck.acquire_failed", "err", wrappedErr)
		return wrappedErr
	}
	defer lease.Release()
	_, err = manager.semantic.Search(
		ctx,
		codebasePath,
		bootSelfCheckQuery,
		bootSelfCheckLimit,
		nil,
		"",
	)
	if err != nil {
		wrappedErr := fmt.Errorf("search boot self-check collection %s: %w", collectionName, err)
		slog.WarnContext(ctx, "daemon.selfcheck.search_failed", "err", wrappedErr)
		return wrappedErr
	}
	return nil
}

func (manager *Manager) bootSelfCheckSkipReason() string {
	if manager.config.IndexBackend == config.IndexBackendLocal {
		return "local backend search may perform crash recovery"
	}
	if manager.config.EmbeddingProvider == config.EmbeddingProviderONNX {
		return "ONNX inference cannot guarantee cancellation"
	}
	return ""
}

// bootSelfCheckTarget picks the codebase whose collection the check queries: the
// most recently completed index, so the probe hits the data most likely to
// matter and the choice is stable across boots. Only a codebase the registry
// records as indexed qualifies, because anything else has no collection the
// daemon expects to answer. The codebase id breaks ties so two runs on the same
// registry pick the same target.
func (manager *Manager) bootSelfCheckTarget() (model.Codebase, bool) {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	var best model.Codebase
	var bestCompletedAt time.Time
	found := false
	for _, codebase := range manager.codebases {
		if codebase.Status != model.CodebaseStatusIndexed || codebase.CanonicalPath == "" {
			continue
		}
		var completedAt time.Time
		if codebase.LastSuccessfulRun != nil {
			completedAt = codebase.LastSuccessfulRun.CompletedAt
		}
		if !found || completedAt.After(bestCompletedAt) ||
			(completedAt.Equal(bestCompletedAt) && codebase.ID < best.ID) {
			best = codebase
			bestCompletedAt = completedAt
			found = true
		}
	}
	return best, found
}
