package daemon

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
)

// Maintenance reports the operator's maintenance mode.
func (manager *Manager) Maintenance() model.MaintenanceState {
	manager.mu.Lock()
	defer manager.mu.Unlock()
	return manager.maintenance
}

// maintenanceRefusal returns the typed refusal for a request the daemon does
// not serve during maintenance, or nil while the mode is off. A search checks
// it before the path is even resolved, so no store call and no collection
// load happens on the caller's behalf while the mode is on.
func (manager *Manager) maintenanceRefusal() error {
	state := manager.Maintenance()
	if !state.Enabled {
		return nil
	}
	return adapterr.NewMaintenance(state.Reason)
}

// skipForMaintenance reports whether daemon-driven work named by activity must
// not start now, logging the skip so an operator reading the log sees why the
// daemon went quiet. The periodic sweep, the store-maintenance sweep, the
// repair pass, the automatic builds and syncs, the boot resume, and the boot
// self-check all ask it, because each of them starts work against the store.
// A watcher batch that arrives while the mode is on is dropped rather than
// requeued: requeueing would replay it on every debounce for as long as the
// mode lasts, and the first sweep after the mode ends compares the merkle
// snapshot against the tree and syncs every change the batch carried.
func (manager *Manager) skipForMaintenance(ctx context.Context, activity string) bool {
	state := manager.Maintenance()
	if !state.Enabled {
		return false
	}
	slog.InfoContext(ctx, "daemon.maintenance.skipped",
		"component", "daemon",
		"subcomponent", "maintenance",
		"activity", activity,
		"reason", state.Reason,
	)
	return true
}

// SetMaintenance turns maintenance mode on or off. Turning it on persists the
// mode, closes the backend's load gate, and leaves every running job alone;
// the returned count says how many jobs were still running so the operator can
// wait for them or cancel them before touching the store. Turning it off
// persists the change and reopens the gate; the next periodic sweep resumes
// repair, retries, and syncs on its own. Setting the mode it already holds
// refreshes the reason and is otherwise a no-op.
func (manager *Manager) SetMaintenance(
	ctx context.Context,
	enabled bool,
	reason string,
) (model.MaintenanceState, int, error) {
	reason = strings.TrimSpace(reason)
	manager.mu.Lock()
	previous := manager.maintenance
	next := model.MaintenanceState{Enabled: false, Reason: "", Since: time.Time{}}
	if enabled {
		since := previous.Since
		if !previous.Enabled {
			since = clock.Now()
		}
		next = model.MaintenanceState{Enabled: true, Reason: reason, Since: since}
	}
	if err := store.WriteMaintenance(manager.maintenancePath(), next); err != nil {
		manager.mu.Unlock()
		slog.ErrorContext(ctx, "persist maintenance mode failed", "path", manager.maintenancePath(), "err", err)
		return previous, 0, fmt.Errorf("persist maintenance mode: %w", err)
	}
	manager.maintenance = next
	activeJobs := 0
	for _, job := range manager.jobs {
		if !isTerminalJobState(job.State) {
			activeJobs++
		}
	}
	manager.mu.Unlock()

	manager.applyMaintenanceGate(next.Enabled)
	if next.Enabled != previous.Enabled {
		slog.WarnContext(ctx, "daemon.maintenance.changed",
			"component", "daemon",
			"subcomponent", "maintenance",
			"enabled", next.Enabled,
			"reason", next.Reason,
			"active_jobs", activeJobs,
		)
	}
	return next, activeJobs, nil
}

// loadMaintenance restores the persisted mode at startup and reapplies it to
// the backend, so a daemon that restarts during a store restore comes back
// still paused rather than resuming loads into a half-restored store.
func (manager *Manager) loadMaintenance(ctx context.Context) error {
	state, err := store.ReadMaintenance(manager.maintenancePath())
	if err != nil {
		slog.ErrorContext(ctx, "read maintenance mode failed", "path", manager.maintenancePath(), "err", err)
		return fmt.Errorf("read maintenance mode: %w", err)
	}
	manager.mu.Lock()
	manager.maintenance = state
	manager.mu.Unlock()
	manager.applyMaintenanceGate(state.Enabled)
	if state.Enabled {
		slog.WarnContext(ctx, "daemon.maintenance.restored",
			"component", "daemon",
			"subcomponent", "maintenance",
			"reason", state.Reason,
			"since", state.Since,
		)
	}
	return nil
}

// applyMaintenanceGate mirrors the mode onto the backend's load gate.
func (manager *Manager) applyMaintenanceGate(enabled bool) {
	if manager.semantic == nil {
		return
	}
	manager.semantic.SetMaintenance(enabled)
}

func (manager *Manager) maintenancePath() string {
	return store.MaintenancePath(manager.config.RegistryPath)
}
