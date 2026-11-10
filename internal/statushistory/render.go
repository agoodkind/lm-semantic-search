package statushistory

import (
	"fmt"
	"strconv"
	"strings"
)

// Human renders the bounded report for an operator. JSON callers use Report
// directly so null remains distinct from a measured zero.
func Human(report Report) string {
	lines := []string{
		"Historical status",
		"window " + report.Window.Start.UTC().Format("2006-01-02T15:04:05Z") + " to " + report.Window.End.UTC().Format("2006-01-02T15:04:05Z"),
		"coverage " + coverageText(report.Coverage.Complete),
		"",
		"Metrics",
	}
	for _, metric := range report.Metrics {
		switch metric.Kind {
		case metricKindCounter:
			lines = append(lines, fmt.Sprintf("%s current=%s delta=%s rate=%s/s %s", metric.Name, integerText(metric.Current), integerText(metric.Delta), floatText(metric.RatePerSecond), metric.Unit))
		case metricKindGauge:
			lines = append(lines, fmt.Sprintf("%s current=%s min=%s mean=%s max=%s %s", metric.Name, integerText(metric.Current), integerText(metric.Min), floatText(metric.Mean), integerText(metric.Max), metric.Unit))
		}
	}
	if report.EmbeddingLatency != nil {
		lines = append(lines, "", "Aggregate latency", durationText(*report.EmbeddingLatency))
	}
	lines = append(lines, "", "Exclusive time")
	for _, duration := range report.TimeBreakdown {
		lines = append(lines, durationText(duration))
	}
	lines = append(lines, "unattributed_duration_ms "+integerText(report.UnattributedDurationMS))
	if len(report.Warnings) > 0 {
		lines = append(lines, "", "Warnings")
		lines = append(lines, report.Warnings...)
	}
	return strings.Join(lines, "\n")
}

func durationText(duration Duration) string {
	return fmt.Sprintf("%s total=%s calls=%s mean=%s p50=%s p95=%s max=%s ms", duration.Name, integerText(duration.TotalMS), integerText(duration.Calls), floatText(duration.MeanMS), integerText(duration.P50MS), integerText(duration.P95MS), integerText(duration.MaxMS))
}

func coverageText(complete bool) string {
	if complete {
		return "complete"
	}
	return "incomplete"
}

func integerText(value *int64) string {
	if value == nil {
		return "n/a"
	}
	return strconv.FormatInt(*value, 10)
}

func floatText(value *float64) string {
	if value == nil {
		return "n/a"
	}
	return fmt.Sprintf("%.2f", *value)
}
