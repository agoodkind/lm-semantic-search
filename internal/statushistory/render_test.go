package statushistory

import (
	"strings"
	"testing"
)

func TestHumanSeparatesAggregateLatencyFromExclusiveTime(t *testing.T) {
	total := int64(120)
	calls := int64(3)
	report := Report{
		EmbeddingLatency: &Duration{Name: "embed_latency", TotalMS: &total, Calls: &calls},
		TimeBreakdown:    []Duration{{Name: "semantic.embedBatch", TotalMS: &total, Calls: &calls}},
	}

	output := Human(report)
	aggregate := strings.Index(output, "Aggregate latency\nembed_latency")
	exclusive := strings.Index(output, "Exclusive time\nsemantic.embedBatch")
	if aggregate < 0 || exclusive < 0 || aggregate > exclusive {
		t.Fatalf("output did not separate latency sections:\n%s", output)
	}
}
