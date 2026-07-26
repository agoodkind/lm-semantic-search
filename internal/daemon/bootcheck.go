package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"goodkind.io/gklog/correlation"
	"goodkind.io/lm-semantic-search/internal/clock"
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
	// bootSelfCheckTimeout bounds the whole check. A store or embedder that never
	// answers ends the check as a failure instead of leaving a goroutine parked on
	// it for the life of the process.
	bootSelfCheckTimeout = 20 * time.Second
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
)

// StartBootSelfCheck launches the one-shot boot self-check and returns
// immediately, so startup never waits on it and the daemon serves regardless of
// the outcome. The check runs once per process, after a short delay, under its
// own deadline, and cancels with the runtime context at shutdown.
func (manager *Manager) StartBootSelfCheck(ctx context.Context) {
	delay := manager.bootSelfCheckDelay
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(ctx, "daemon.selfcheck.panic",
					"component", "daemon",
					"subcomponent", "selfcheck",
					"err", fmt.Errorf("panic: %v", recovered),
				)
			}
		}()

		timer := time.NewTimer(delay)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
		}

		checkCtx := correlation.WithContext(ctx, correlation.New("").WithIdentityAttributes(
			correlation.IdentityAttribute{Key: "origin", Value: "boot-selfcheck"},
		))
		boundedCtx, cancel := context.WithTimeout(checkCtx, bootSelfCheckTimeout)
		defer cancel()
		_, _ = manager.runBootSelfCheck(boundedCtx)
	}()
}

// runBootSelfCheck proves the search path end to end once, by running a real
// one-row query against a collection the registry says is indexed. That single
// call embeds the query text through the configured endpoint and reads the
// collection back out of the store, so it catches a restore that left the
// daemon connected to a store whose data can no longer answer a query, which a
// component-by-component liveness ping cannot see.
//
// It is read-only and idempotent: a search writes nothing, deletes nothing, and
// triggers no indexing, so it is safe on every boot.
//
// The outcome feeds the same shared-dependency health record that job and search
// outcomes feed, so a failure raises the existing status banner rather than
// introducing a second notion of health, and a pass clears a boot-time banner
// the way a real user query would.
func (manager *Manager) runBootSelfCheck(ctx context.Context) (bootSelfCheckOutcome, error) {
	if manager.semantic == nil {
		slog.InfoContext(ctx, "daemon.selfcheck.skipped",
			"component", "daemon",
			"subcomponent", "selfcheck",
			"reason", "no vector backend configured",
		)
		return bootSelfCheckSkipped, nil
	}
	if !manager.semantic.Available() {
		manager.noteDependencyFailure(semantic.ErrUnavailable)
		slog.ErrorContext(ctx, "daemon.selfcheck.failed",
			"component", "daemon",
			"subcomponent", "selfcheck",
			"reason", "vector store is not connected",
			"err", semantic.ErrUnavailable,
		)
		return bootSelfCheckFailed, semantic.ErrUnavailable
	}

	codebase, found := manager.bootSelfCheckTarget()
	if !found {
		slog.InfoContext(ctx, "daemon.selfcheck.skipped",
			"component", "daemon",
			"subcomponent", "selfcheck",
			"reason", "no indexed codebase to query",
		)
		return bootSelfCheckSkipped, nil
	}

	startedAt := clock.Now()
	_, err := manager.semantic.Search(ctx, codebase.CanonicalPath, bootSelfCheckQuery, bootSelfCheckLimit, nil, "")
	durationMS := clock.Now().Sub(startedAt).Milliseconds()
	if err != nil {
		manager.noteDependencyFailure(err)
		checkErr := fmt.Errorf("boot self-check query against %s: %w", codebase.CanonicalPath, err)
		slog.ErrorContext(ctx, "daemon.selfcheck.failed",
			"component", "daemon",
			"subcomponent", "selfcheck",
			"codebase_id", codebase.ID,
			"path", codebase.CanonicalPath,
			"duration_ms", durationMS,
			"err", checkErr,
		)
		return bootSelfCheckFailed, checkErr
	}

	// The query embedded through the configured endpoint and the store answered
	// the search, which is the whole pipeline, so this clears any mode the boot
	// dial or a prior outage left on the record.
	manager.noteDependencyHealthy()
	slog.InfoContext(ctx, "daemon.selfcheck.passed",
		"component", "daemon",
		"subcomponent", "selfcheck",
		"codebase_id", codebase.ID,
		"path", codebase.CanonicalPath,
		"duration_ms", durationMS,
	)
	return bootSelfCheckPassed, nil
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
