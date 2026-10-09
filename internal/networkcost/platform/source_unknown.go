//go:build !darwin || !cgo

package platform

import (
	"context"

	"goodkind.io/lm-semantic-search/internal/networkcost"
)

type unknownSource struct{}

func (unknownSource) Classify(context.Context) networkcost.Classification {
	return networkcost.ClassificationUnknown
}
