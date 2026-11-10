// Package logcleanup runs the daemon's off-hot-path log retention sweep.
//
// Rotation is a write-path concern owned by gklog: active files rotate when
// they exceed their size cap. This package owns the separate retention concern.
// A background walker started at daemon boot deletes rotated backups that
// exceed the retention byte budget, oldest first, and never touches an active
// log file. Deletion happens only on the walker goroutine, never on the
// log-write path, mirroring clyde's rotation-vs-cleanup separation.
package logcleanup

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"goodkind.io/lm-semantic-search/internal/clock"
)

const (
	// completedEvent is emitted after every sweep pass. The concern router keys
	// on the first dot-segment, so this record lands in the cleanup concern file.
	completedEvent = "cleanup.completed"
	// startedEvent marks the beginning of a sweep pass.
	startedEvent = "cleanup.started"
	// auditEvent reports what a disabled sweep would delete without deleting it.
	auditEvent = "cleanup.audit"
	// skippedEvent marks a pass that did no work because the root was unusable.
	skippedEvent = "cleanup.skipped"
	// component tags every event so log consumers can filter this walker.
	component = "logcleanup"
	// defaultInterval bounds the sweep cadence when the caller passes a
	// non-positive interval, so a misconfigured knob cannot spin the walker.
	defaultInterval = 5 * time.Minute
)

// lumberjackBackupPattern matches the timestamp lumberjack inserts before the
// extension of a rotated file (backupTimeFormat "2006-01-02T15-04-05.000"). A
// name that carries this stamp, or a compressed ".gz" name, is a rotated backup
// eligible for deletion; every other log file is treated as active and kept.
var lumberjackBackupPattern = regexp.MustCompile(`-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{3}`)

// Policy is the resolved input for one retention sweep.
type Policy struct {
	// Root is the log directory walked for rotated backups.
	Root string
	// RetentionBytes caps the total size of rotated backups kept. Backups past
	// this budget are deleted oldest first. Zero or below keeps everything.
	RetentionBytes int64
	// Enabled deletes candidates when true; when false the sweep audits only.
	Enabled bool
}

// Result mirrors the cleanup.completed event one-for-one so callers can
// aggregate outcomes without reparsing the slog stream.
type Result struct {
	// Scanned counts every log file the walk visited, active and rotated.
	Scanned int `json:"scanned"`
	// Candidates counts rotated backups past the retention budget.
	Candidates int `json:"candidates"`
	// Deleted counts backups the sweep actually removed. Zero when disabled.
	Deleted int `json:"deleted"`
	// BytesDeleted sums the sizes of removed backups in bytes.
	BytesDeleted int64 `json:"bytes_deleted"`
	// Errors lists "<path>: <err>" strings for walk and remove failures.
	Errors []string `json:"errors"`
	// DurationMS is the wall-clock duration of the pass in milliseconds.
	DurationMS int64 `json:"duration_ms"`
}

type logFile struct {
	path    string
	size    int64
	modTime time.Time
}

// Start launches the background retention sweep. It runs one pass immediately,
// then one pass per interval until ctx is cancelled. Every pass runs on this
// goroutine, never on the log-write path, so a write can never trigger a
// delete. A non-positive interval falls back to defaultInterval.
func Start(ctx context.Context, policy Policy, interval time.Duration) {
	if interval <= 0 {
		interval = defaultInterval
	}
	go func() {
		// Top-level backstop recover: the per-pass recover in runOnceRecovered
		// already keeps the ticking loop alive across a panicking sweep, so this
		// only ever catches a panic from the loop scaffolding itself.
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.ErrorContext(ctx, "cleanup.walker.panicked",
					"component", component,
					"err", fmt.Errorf("panic: %v", recovered),
				)
			}
		}()
		runOnceRecovered(ctx, policy)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				runOnceRecovered(ctx, policy)
			}
		}
	}()
}

// runOnceRecovered runs one sweep pass and recovers from a panic so a single
// failed pass never kills the walker. Retention keeps running on the next tick
// rather than stopping until a daemon restart, which would let logs grow
// unbounded again, the exact failure this package exists to prevent.
func runOnceRecovered(ctx context.Context, policy Policy) {
	defer func() {
		if recovered := recover(); recovered != nil {
			slog.ErrorContext(ctx, "cleanup.walker.panicked",
				"component", component,
				"err", fmt.Errorf("panic: %v", recovered),
			)
		}
	}()
	RunOnce(policy)
}

// RunOnce performs a single retention sweep and emits cleanup.completed. It is
// the unit Start calls each interval: Start owns scheduling, RunOnce owns one
// pass. It deletes only rotated backups past the budget and never an active
// log file. A disabled policy audits the candidates and deletes nothing.
func RunOnce(policy Policy) Result {
	startedAt := clock.Now()
	result := Result{Scanned: 0, Candidates: 0, Deleted: 0, BytesDeleted: 0, Errors: []string{}, DurationMS: 0}

	root := strings.TrimSpace(policy.Root)
	if root == "" {
		slog.Debug(skippedEvent, "component", component, "reason", "empty_root")
		result.DurationMS = elapsedMS(startedAt)
		return result
	}
	if _, statErr := os.Stat(root); errors.Is(statErr, fs.ErrNotExist) {
		slog.Debug(skippedEvent, "component", component, "reason", "root_missing", "root", root)
		result.DurationMS = elapsedMS(startedAt)
		return result
	}

	slog.Info(startedEvent,
		"component", component,
		"root", root,
		"enabled", policy.Enabled,
		"retention_bytes", policy.RetentionBytes,
	)

	all, backups, walkErrs := collectLogFiles(root)
	result.Errors = append(result.Errors, walkErrs...)
	result.Scanned = len(all)

	candidates := overflowBackups(backups, policy.RetentionBytes)
	result.Candidates = len(candidates)

	if policy.Enabled {
		deleteCandidates(candidates, &result)
	} else {
		auditCandidates(root, candidates)
	}

	result.DurationMS = elapsedMS(startedAt)
	slog.Info(completedEvent,
		"component", component,
		"root", root,
		"enabled", policy.Enabled,
		"scanned", result.Scanned,
		"candidates", result.Candidates,
		"deleted", result.Deleted,
		"bytes_deleted", result.BytesDeleted,
		"errors", result.Errors,
		"duration_ms", result.DurationMS,
	)
	return result
}

func deleteCandidates(candidates []logFile, result *Result) {
	for _, candidate := range candidates {
		if err := os.Remove(candidate.path); err != nil {
			if os.IsNotExist(err) {
				continue
			}
			slog.Warn("cleanup.remove_failed", "component", component, "path", candidate.path, "err", err)
			result.Errors = append(result.Errors, fmt.Sprintf("%s: %v", candidate.path, err))
			continue
		}
		result.Deleted++
		result.BytesDeleted += candidate.size
	}
}

func auditCandidates(root string, candidates []logFile) {
	var wouldDeleteBytes int64
	for _, candidate := range candidates {
		wouldDeleteBytes += candidate.size
	}
	slog.Info(auditEvent,
		"component", component,
		"root", root,
		"candidates", len(candidates),
		"would_delete_bytes", wouldDeleteBytes,
	)
}

// collectLogFiles walks root and returns every log file and the rotated-backup
// subset. Walk and stat failures are collected as strings so one bad entry
// cannot abort the pass or block a delete of the healthy candidates.
func collectLogFiles(root string) (all []logFile, backups []logFile, errs []string) {
	all = make([]logFile, 0)
	backups = make([]logFile, 0)
	errs = make([]string, 0)
	walkErr := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			slog.Warn("cleanup.walk_entry_failed", "component", component, "path", path, "err", err)
			errs = append(errs, fmt.Sprintf("%s: %v", path, err))
			return nil
		}
		if entry.IsDir() {
			return nil
		}
		if !isLogFile(path) {
			return nil
		}
		info, infoErr := entry.Info()
		if infoErr != nil {
			slog.Warn("cleanup.stat_failed", "component", component, "path", path, "err", infoErr)
			errs = append(errs, fmt.Sprintf("%s: %v", path, infoErr))
			return nil
		}
		file := logFile{path: path, size: info.Size(), modTime: info.ModTime()}
		all = append(all, file)
		if isRotatedBackup(filepath.Base(path)) {
			backups = append(backups, file)
		}
		return nil
	})
	if walkErr != nil {
		slog.Warn("cleanup.walk_failed", "component", component, "root", root, "err", walkErr)
		errs = append(errs, fmt.Sprintf("%s: %v", root, walkErr))
	}
	return all, backups, errs
}

// overflowBackups returns the rotated backups whose cumulative size, newest
// first, exceeds budget. The newest backups up to the budget are kept; the
// oldest beyond it are returned for deletion, so retention is a total-bytes
// budget over rotated files with oldest expired first.
func overflowBackups(backups []logFile, budget int64) []logFile {
	if budget <= 0 {
		return nil
	}
	sorted := append([]logFile(nil), backups...)
	sort.Slice(sorted, func(i int, j int) bool {
		return sorted[i].modTime.After(sorted[j].modTime)
	})
	var total int64
	overflow := make([]logFile, 0)
	for _, file := range sorted {
		total += file.size
		if total > budget {
			overflow = append(overflow, file)
		}
	}
	return overflow
}

// isLogFile reports whether path names a log surface the sweep considers. Lock
// sidecars are excluded so the multi-process write lock is never removed.
func isLogFile(path string) bool {
	base := strings.ToLower(filepath.Base(path))
	if base == "" || strings.HasSuffix(base, ".lock") || base == "lock" {
		return false
	}
	return strings.HasSuffix(base, ".log") ||
		strings.HasSuffix(base, ".log.gz") ||
		strings.Contains(base, ".jsonl")
}

// isRotatedBackup reports whether name is a rotated backup rather than an
// active file. A compressed name is always a backup; an uncompressed name is a
// backup only when it carries lumberjack's rotation timestamp, so the active
// file lumberjack still writes is never a deletion candidate.
func isRotatedBackup(name string) bool {
	if strings.HasSuffix(strings.ToLower(name), ".gz") {
		return true
	}
	return lumberjackBackupPattern.MatchString(name)
}

func elapsedMS(startedAt time.Time) int64 {
	return max(clock.Now().Sub(startedAt), time.Millisecond).Milliseconds()
}
