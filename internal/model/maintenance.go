package model

import "time"

// MaintenanceState is the daemon's operator-set maintenance mode. While Enabled
// the daemon starts no background sweep, repair pass, automatic rebuild, or
// collection load, refuses index and conversation writes, and fails searches
// fast, so an operator can back up or restore the vector store with no daemon
// traffic against it. It is persisted beside the registry so a daemon restart
// during maintenance comes back still paused.
type MaintenanceState struct {
	Enabled bool `json:"enabled"`
	// Reason is the operator's note, shown on every status surface while the
	// mode is on. Empty while off.
	Reason string `json:"reason,omitempty"`
	// Since is when the current mode began, in UTC. Zero while off.
	Since time.Time `json:"since"`
}
