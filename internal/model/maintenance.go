package model

import "time"

type MaintenanceState struct {
	Enabled bool `json:"enabled"`
	// Reason is the operator's note, shown on every status surface while the
	// mode is on. Empty while off.
	Reason string `json:"reason,omitempty"`
	// Since is when the current mode began, in UTC. Zero while off.
	Since time.Time `json:"since"`
}
