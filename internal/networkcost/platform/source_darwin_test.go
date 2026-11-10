//go:build darwin && cgo

package platform_test

import (
	"context"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/networkcost"
	"goodkind.io/lm-semantic-search/internal/networkcost/platform"
)

const classifyReturnMargin = time.Second

func TestPlatformSourceReturnsDefinedClassificationWithinTimeout(t *testing.T) {
	source := platform.New()

	startedAt := time.Now()
	classification := source.Classify(context.Background())
	elapsed := time.Since(startedAt)

	switch classification {
	case networkcost.ClassificationUnknown,
		networkcost.ClassificationNormal,
		networkcost.ClassificationExpensive,
		networkcost.ClassificationConstrained,
		networkcost.ClassificationExpensiveAndConstrained:
	default:
		t.Fatalf("Classify = %q, want a defined classification", classification)
	}
	if elapsed > platform.ClassifyTimeout+classifyReturnMargin {
		t.Fatalf(
			"Classify took %s, want at most %s",
			elapsed,
			platform.ClassifyTimeout+classifyReturnMargin,
		)
	}
	t.Logf("classification=%s elapsed=%s", classification, elapsed)
}

func TestPlatformSourceReturnsUnknownForCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	classification := platform.New().Classify(ctx)
	if classification != networkcost.ClassificationUnknown {
		t.Fatalf(
			"Classify = %q want %q",
			classification,
			networkcost.ClassificationUnknown,
		)
	}
}
