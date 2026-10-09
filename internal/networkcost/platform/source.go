// Package platform reads the default network path's cost classification
// from the operating system.
package platform

import (
	"context"
	"time"

	"goodkind.io/lm-semantic-search/internal/networkcost"
)

// ClassifyTimeout limits each read. Classify returns unknown on timeout.
const ClassifyTimeout = 2 * time.Second

// Source reads the current classification of the default network path.
// Classify returns unknown when the context is canceled or the read times out.
// Classify uses an earlier context deadline instead of ClassifyTimeout.
type Source interface {
	Classify(context.Context) networkcost.Classification
}
