package status

import (
	"strconv"
	"strings"
	"testing"
	"unicode"
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

// testField builds one counter for a test.
func testField(group string, name string, unit string, value Value) Field {
	return Field{Group: group, Name: name, Unit: unit, Value: value, NoDelta: false}
}

// TestDumpPrintsOneRecordPerLine proves the piped form is one
// whitespace-separated record per line, so grep, awk, and cut work on it, and
// that a value is followed by its unit as a third field.
func TestDumpPrintsOneRecordPerLine(t *testing.T) {
	t.Parallel()

	snapshot := Snapshot{
		Counters: []Field{
			testField("embed", "embed_vectors_total", "vectors", Int(3946)),
			testField("dependency_health", "dependency_health.degraded", "", Bool(false)),
			testField("dependency_health", "dependency_health.mode", "", Text("")),
			testField("dependency_health", "dependency_health.since", "", Value{}),
		},
	}

	got := lineFields(Dump(snapshot))
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

// TestDumpPrintsRawDigits proves the piped form does not group digits.
// Consumers parse that output, and a separator would break every one of them.
func TestDumpPrintsRawDigits(t *testing.T) {
	t.Parallel()

	snapshot := Snapshot{
		Counters: []Field{testField("runtime", "heap_alloc_bytes", "bytes", Int(249278160))},
	}
	text := Dump(snapshot)
	if !strings.Contains(text, "heap_alloc_bytes 249278160 bytes") {
		t.Fatalf("piped output grouped digits or dropped the unit:\n%s", text)
	}
}

// TestDumpIndexesActivityRows proves an activity field is addressable
// by its row position, which keeps the rows apart without an index field.
func TestDumpIndexesActivityRows(t *testing.T) {
	t.Parallel()

	snapshot := Snapshot{
		Activity: [][]Field{
			{testField("", "job_id", "", Text("job_a"))},
			{
				testField("", "job_id", "", Value{}),
				testField("", "pending_paths", "paths", Int(8)),
			},
		},
	}
	text := Dump(snapshot)
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

// TestValueTextRendersEachKind proves every value kind has a rendering. A new
// kind cannot silently fall through to null.
func TestValueTextRendersEachKind(t *testing.T) {
	t.Parallel()

	cases := map[string]Value{
		"3946":  Int(3946),
		"40.8":  Float(40.8),
		"true":  Bool(true),
		"index": Text("index"),
		`""`:    Text(""),
		"null":  {},
	}
	for want, value := range cases {
		if got := valueText(value); got != want {
			t.Fatalf("valueText(%+v) = %q, want %q", value, got, want)
		}
	}
}

// TestDumpPrintsValuesAsWritten proves a value keeps its spaces. A phase name
// and a path read the way a person wrote them. This text is read rather than
// parsed; a machine consumer uses the JSON form.
func TestDumpPrintsValuesAsWritten(t *testing.T) {
	t.Parallel()

	snapshot := Snapshot{
		Activity: [][]Field{
			{
				testField("", "phase", "", Text("Reindexing changed files...")),
				testField("", "canonical_path", "", Text("/Users/a/some project")),
				testField("", "operation", "", Text("index")),
			},
		},
	}

	text := Dump(snapshot)
	lines := strings.Split(strings.TrimSpace(text), "\n")
	if len(lines) != 3 {
		t.Fatalf("records = %d, want 3; a value spanned a line:\n%s", len(lines), text)
	}
	got := lineFields(text)
	cases := map[string]string{
		"activity.0.phase":          "Reindexing changed files...",
		"activity.0.canonical_path": "/Users/a/some project",
		"activity.0.operation":      "index",
	}
	for name, want := range cases {
		if got[name] != want {
			t.Fatalf("%s = %s, want %s", name, got[name], want)
		}
	}
}

// TestDumpValuesRoundTrip proves an escaped value recovers its exact
// original bytes. Escaping costs a consumer one unquote call.
func TestDumpValuesRoundTrip(t *testing.T) {
	t.Parallel()

	originals := []string{
		"/tmp/a\nembed_vectors_total 999999",
		"tab\there",
		`quote"inside`,
		`back\slash`,
		"",
	}
	for _, original := range originals {
		rendered := valueText(Text(original))
		recovered, err := strconv.Unquote(rendered)
		if err != nil {
			t.Fatalf("Unquote(%q) returned error: %v", rendered, err)
		}
		if recovered != original {
			t.Fatalf("round trip of %q produced %q", original, recovered)
		}
	}
}

// TestDumpKeepsCountersAsOneBareField proves a counter value stays a
// single unquoted field, and reading a number with awk or cut still works. Only
// string values can carry whitespace, and no counter is a string.
func TestDumpKeepsCountersAsOneBareField(t *testing.T) {
	t.Parallel()

	snapshot := Snapshot{
		Counters: []Field{
			testField("embed", "embed_vectors_total", "vectors", Int(3946)),
			testField("activity", "overall_percent", "%", Float(40.8)),
			testField("dependency_health", "dependency_health.degraded", "", Bool(false)),
		},
	}

	for _, line := range strings.Split(strings.TrimSpace(Dump(snapshot)), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 || len(fields) > 3 {
			t.Fatalf("record %q has %d fields, want name, value, and optional unit", line, len(fields))
		}
		if strings.HasPrefix(fields[1], `"`) {
			t.Fatalf("record %q quoted a numeric value, breaking awk and cut", line)
		}
	}
}

// TestDumpNewlineValueCannotForgeARecord proves a value with a newline stays
// inside its own record. Unquoted, its tail would read as a separate name and
// value that no counter ever reported.
func TestDumpNewlineValueCannotForgeARecord(t *testing.T) {
	t.Parallel()

	snapshot := Snapshot{
		Counters: []Field{
			testField("daemon", "socket_path", "", Text("/tmp/a\nembed_vectors_total 999999")),
		},
	}

	text := Dump(snapshot)
	if lines := strings.Split(strings.TrimSpace(text), "\n"); len(lines) != 1 {
		t.Fatalf("records = %d, want 1; the newline forged a record:\n%s", len(lines), text)
	}
	if strings.Contains(text, "\nembed_vectors_total") {
		t.Fatalf("a forged embed_vectors_total record appears:\n%s", text)
	}
}

// TestDumpEscapesControlRunes proves an unprintable rune never appears in the
// output bare. A codebase path is operator-supplied and this text is printed to
// a terminal. A raw escape character would emit a control sequence that clears
// the screen or moves the cursor.
func TestDumpEscapesControlRunes(t *testing.T) {
	t.Parallel()

	// A clear-screen sequence with no whitespace in it, which an earlier
	// whitespace-only predicate let through untouched.
	hostile := "/Users/a/\x1b[2Jwiped"
	rendered := valueText(Text(hostile))

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

// TestDumpPrintsIdentityBeforeCounters proves the piped form says which process
// produced it and when. Without those records a captured snapshot cannot be
// told from another machine's, or from the same machine an hour earlier.
func TestDumpPrintsIdentityBeforeCounters(t *testing.T) {
	t.Parallel()

	snapshot := Snapshot{
		Identity: []Field{
			testField("", "version", "", Text("202607270542-fe-6e0a44c")),
			testField("", "commit", "", Text("6e0a44c")),
			testField("", "pid", "", Text("7342")),
			testField("", "socket", "", Text("/Users/a/state/daemon.sock")),
			testField("", "read_at", "", Text("2026-07-27T12:52:31Z")),
		},
		Counters: []Field{testField("embed", "embed_vectors_total", "vectors", Int(3946))},
	}

	got := lineFields(Dump(snapshot))
	cases := map[string]string{
		"version": "202607270542-fe-6e0a44c",
		"commit":  "6e0a44c",
		"pid":     "7342",
		"socket":  "/Users/a/state/daemon.sock",
		"read_at": "2026-07-27T12:52:31Z",
	}
	for name, want := range cases {
		if got[name] != want {
			t.Fatalf("%s = %q, want %q", name, got[name], want)
		}
	}
	if got["embed_vectors_total"] != "3946 vectors" {
		t.Fatalf("identity records displaced the counters: %q", got["embed_vectors_total"])
	}
}
