package render

import (
	"strconv"
	"strings"
	"testing"
	"unicode"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

// lineFields indexes the rendered text by record name so a test asserts on one
// record without depending on the order of the others.
func lineFields(text string) map[string]string {
	fields := map[string]string{}
	for _, line := range strings.Split(strings.TrimSpace(text), "\n") {
		parts := strings.SplitN(line, " ", 2)
		if len(parts) == 2 {
			fields[parts[0]] = parts[1]
		}
	}
	return fields
}

// TestStatusMetricsPrintsOneRecordPerLine proves the piped form is one
// whitespace-separated record per line, so grep, awk, and cut work on it, and
// that a value carries its unit as a third field.
func TestStatusMetricsPrintsOneRecordPerLine(t *testing.T) {
	t.Parallel()

	response := &pb.GetStatusResponse{
		Metrics: []*pb.Metric{
			{
				Group: "embed", Name: "embed_vectors_total", Unit: "vectors",
				Value: &pb.Metric_IntValue{IntValue: 3946},
			},
			{
				Group: "dependency_health", Name: "dependency_health.degraded",
				Value: &pb.Metric_BoolValue{BoolValue: false},
			},
			{
				Group: "dependency_health", Name: "dependency_health.mode",
				Value: &pb.Metric_StringValue{StringValue: ""},
			},
			{Group: "dependency_health", Name: "dependency_health.since"},
		},
	}

	got := lineFields(StatusMetrics(response))
	cases := map[string]string{
		"embed_vectors_total":        "3946 vectors",
		"dependency_health.degraded": "false",
		"dependency_health.mode":     `""`,
		"dependency_health.since":    "null",
	}
	for name, want := range cases {
		if got[name] != want {
			t.Fatalf("%s line = %q, want %q", name, got[name], want)
		}
	}
}

// TestStatusMetricsPrintsRawDigits proves the piped form does not group digits.
// That output is parsed, so a separator would break every consumer.
func TestStatusMetricsPrintsRawDigits(t *testing.T) {
	t.Parallel()

	response := &pb.GetStatusResponse{
		Metrics: []*pb.Metric{
			{
				Group: "runtime", Name: "heap_alloc_bytes", Unit: "bytes",
				Value: &pb.Metric_IntValue{IntValue: 249278160},
			},
		},
	}
	text := StatusMetrics(response)
	if !strings.Contains(text, "heap_alloc_bytes 249278160 bytes") {
		t.Fatalf("piped output grouped digits or dropped the unit:\n%s", text)
	}
}

// TestStatusMetricsIndexesActivityRows proves an activity field is addressable
// by its row position, so the flat form keeps the rows apart without carrying
// an index field protojson would drop at zero.
func TestStatusMetricsIndexesActivityRows(t *testing.T) {
	t.Parallel()

	response := &pb.GetStatusResponse{
		Activity: []*pb.ActivityRow{
			{Metrics: []*pb.Metric{
				{Name: "job_id", Value: &pb.Metric_StringValue{StringValue: "job_a"}},
			}},
			{Metrics: []*pb.Metric{
				{Name: "job_id"},
				{Name: "pending_paths", Unit: "paths", Value: &pb.Metric_IntValue{IntValue: 8}},
			}},
		},
	}
	text := StatusMetrics(response)
	for _, want := range []string{
		"activity.0.job_id job_a",
		"activity.1.job_id null",
		"activity.1.pending_paths 8 paths",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("missing %q in:\n%s", want, text)
		}
	}
}

// TestMetricValueTextRendersEachOneofMember proves every value kind has a
// rendering, so a new member cannot silently fall through to null.
func TestMetricValueTextRendersEachOneofMember(t *testing.T) {
	t.Parallel()

	cases := map[string]*pb.Metric{
		"3946":  {Value: &pb.Metric_IntValue{IntValue: 3946}},
		"40.8":  {Value: &pb.Metric_DoubleValue{DoubleValue: 40.8}},
		"true":  {Value: &pb.Metric_BoolValue{BoolValue: true}},
		"index": {Value: &pb.Metric_StringValue{StringValue: "index"}},
		`""`:    {Value: &pb.Metric_StringValue{StringValue: ""}},
		"null":  {},
	}
	for want, metric := range cases {
		if got := MetricValueText(metric); got != want {
			t.Fatalf("MetricValueText(%+v) = %q, want %q", metric.GetValue(), got, want)
		}
	}
}

// TestStatusMetricsKeepsEveryValueOneField proves a value carrying whitespace
// stays exactly one whitespace-separated field, so reading a value by field
// position works with awk, cut, and read no matter what the value holds.
func TestStatusMetricsKeepsEveryValueOneField(t *testing.T) {
	t.Parallel()

	response := &pb.GetStatusResponse{
		Activity: []*pb.ActivityRow{
			{Metrics: []*pb.Metric{
				{Name: "phase", Value: &pb.Metric_StringValue{StringValue: "Reindexing changed files..."}},
				{Name: "canonical_path", Value: &pb.Metric_StringValue{StringValue: "/Users/a/my project"}},
				{Name: "operation", Value: &pb.Metric_StringValue{StringValue: "index"}},
			}},
		},
	}

	text := StatusMetrics(response)
	lines := strings.Split(strings.TrimSpace(text), "\n")
	if len(lines) != 3 {
		t.Fatalf("records = %d, want 3; a value spanned a line:\n%s", len(lines), text)
	}
	for _, line := range lines {
		if fields := strings.Fields(line); len(fields) != 2 {
			t.Fatalf("record %q has %d fields, want exactly name and value", line, len(fields))
		}
	}

	got := lineFields(text)
	cases := map[string]string{
		"activity.0.phase":          `"Reindexing\x20changed\x20files..."`,
		"activity.0.canonical_path": `"/Users/a/my\x20project"`,
		"activity.0.operation":      "index",
	}
	for name, want := range cases {
		if got[name] != want {
			t.Fatalf("%s = %s, want %s", name, got[name], want)
		}
	}
}

// TestStatusMetricsValuesRoundTrip proves an escaped value recovers its exact
// original bytes, so escaping costs a consumer nothing but one unquote call.
func TestStatusMetricsValuesRoundTrip(t *testing.T) {
	t.Parallel()

	originals := []string{
		"Reindexing changed files...",
		"/Users/a/my project",
		"/tmp/a\nembed_vectors_total 999999",
		"tab\there",
		`quote"inside`,
		`back\slash`,
		"",
	}
	for _, original := range originals {
		rendered := MetricValueText(&pb.Metric{
			Value: &pb.Metric_StringValue{StringValue: original},
		})
		if strings.ContainsFunc(rendered, unicode.IsSpace) {
			t.Fatalf("rendered %q carries whitespace, so it is not one field", rendered)
		}
		recovered, err := strconv.Unquote(rendered)
		if err != nil {
			t.Fatalf("Unquote(%q) returned error: %v", rendered, err)
		}
		if recovered != original {
			t.Fatalf("round trip of %q produced %q", original, recovered)
		}
	}
}

// TestStatusMetricsKeepsCountersAsOneBareField proves a counter value stays a
// single unquoted field, so reading a number with awk or cut still works. Only
// string values can carry whitespace, and no counter is a string.
func TestStatusMetricsKeepsCountersAsOneBareField(t *testing.T) {
	t.Parallel()

	response := &pb.GetStatusResponse{
		Metrics: []*pb.Metric{
			{
				Group: "embed", Name: "embed_vectors_total", Unit: "vectors",
				Value: &pb.Metric_IntValue{IntValue: 3946},
			},
			{
				Group: "activity", Name: "overall_percent", Unit: "%",
				Value: &pb.Metric_DoubleValue{DoubleValue: 40.8},
			},
			{
				Group: "dependency_health", Name: "dependency_health.degraded",
				Value: &pb.Metric_BoolValue{BoolValue: false},
			},
		},
	}

	for _, line := range strings.Split(strings.TrimSpace(StatusMetrics(response)), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 || len(fields) > 3 {
			t.Fatalf("record %q has %d fields, want name, value, and optional unit", line, len(fields))
		}
		if strings.HasPrefix(fields[1], `"`) {
			t.Fatalf("record %q quoted a numeric value, breaking awk and cut", line)
		}
	}
}

// TestStatusMetricsNewlineValueCannotForgeARecord proves a value holding a
// newline stays inside its own record. Unquoted, its tail would read as a
// separate name and value that no counter ever reported.
func TestStatusMetricsNewlineValueCannotForgeARecord(t *testing.T) {
	t.Parallel()

	response := &pb.GetStatusResponse{
		Metrics: []*pb.Metric{
			{
				Group: "daemon", Name: "socket_path",
				Value: &pb.Metric_StringValue{StringValue: "/tmp/a\nembed_vectors_total 999999"},
			},
		},
	}

	text := StatusMetrics(response)
	if lines := strings.Split(strings.TrimSpace(text), "\n"); len(lines) != 1 {
		t.Fatalf("records = %d, want 1; the newline forged a record:\n%s", len(lines), text)
	}
	if strings.Contains(text, "\nembed_vectors_total") {
		t.Fatalf("a forged embed_vectors_total record appears:\n%s", text)
	}
}

// TestStatusMetricsEscapesControlRunes proves an unprintable rune never reaches
// the output bare. A codebase path is operator-supplied and this text goes to a
// terminal, so a raw escape character would emit a control sequence that clears
// the screen or moves the cursor.
func TestStatusMetricsEscapesControlRunes(t *testing.T) {
	t.Parallel()

	// A clear-screen sequence with no whitespace in it, so the earlier
	// whitespace-only predicate let it through untouched.
	hostile := "/Users/a/\x1b[2Jwiped"
	rendered := MetricValueText(&pb.Metric{
		Value: &pb.Metric_StringValue{StringValue: hostile},
	})

	if strings.ContainsRune(rendered, '\x1b') {
		t.Fatalf("a raw escape character survived into the output: %q", rendered)
	}
	for _, candidate := range rendered {
		if !unicode.IsPrint(candidate) {
			t.Fatalf("unprintable rune %q survived into %q", candidate, rendered)
		}
	}
	recovered, err := strconv.Unquote(rendered)
	if err != nil {
		t.Fatalf("Unquote(%q) returned error: %v", rendered, err)
	}
	if recovered != hostile {
		t.Fatalf("round trip produced %q, want the original bytes", recovered)
	}
}
