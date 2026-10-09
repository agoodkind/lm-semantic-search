//go:build darwin && cgo

package platform

/*
#cgo LDFLAGS: -framework Network
#include "network_cost_bridge_darwin.h"
*/
import "C"

import (
	"context"

	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/networkcost"
)

type darwinSource struct{}

// New returns a source that reads nw_path_is_expensive and
// nw_path_is_constrained from Network.framework.
// The source returns unknown when the default network path is not satisfied.
func New() Source {
	return darwinSource{}
}

func (darwinSource) Classify(ctx context.Context) networkcost.Classification {
	timeout := ClassifyTimeout
	deadline, hasDeadline := ctx.Deadline()
	if hasDeadline {
		timeout = min(timeout, clock.Until(deadline))
	}
	if ctx.Err() != nil || timeout <= 0 {
		return networkcost.ClassificationUnknown
	}

	result := C.lms_network_cost_read(C.int64_t(timeout.Nanoseconds()))
	if result.available == 0 {
		return networkcost.ClassificationUnknown
	}
	expensive := result.expensive != 0
	constrained := result.constrained != 0
	if expensive && constrained {
		return networkcost.ClassificationExpensiveAndConstrained
	}
	if expensive {
		return networkcost.ClassificationExpensive
	}
	if constrained {
		return networkcost.ClassificationConstrained
	}
	return networkcost.ClassificationNormal
}
