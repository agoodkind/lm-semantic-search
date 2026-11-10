package status_test

import (
	"bytes"
	"encoding/json"
	"math"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/status"
)

type encodedField struct {
	Value json.RawMessage `json:"value"`
	Unit  *string         `json:"unit"`
}

type encodedSnapshot struct {
	Notices  []string                  `json:"notices"`
	Identity map[string]encodedField   `json:"identity"`
	Counters map[string]encodedField   `json:"counters"`
	Activity []map[string]encodedField `json:"activity"`
}

func newField(name string, unit string, value status.Value) status.Field {
	return status.Field{Group: "", Name: name, Unit: unit, Value: value, NoDelta: false}
}

func encode(t *testing.T, snapshot status.Snapshot) []byte {
	t.Helper()
	var output bytes.Buffer
	if err := status.WriteJSON(&output, snapshot); err != nil {
		t.Fatalf("WriteJSON returned error: %v", err)
	}
	return output.Bytes()
}

func decode(t *testing.T, output []byte) encodedSnapshot {
	t.Helper()
	var decoded encodedSnapshot
	if err := json.Unmarshal(output, &decoded); err != nil {
		t.Fatalf("Unmarshal(%s) returned error: %v", output, err)
	}
	return decoded
}

func topLevel(t *testing.T, output []byte) map[string]json.RawMessage {
	t.Helper()
	var sections map[string]json.RawMessage
	if err := json.Unmarshal(output, &sections); err != nil {
		t.Fatalf("Unmarshal(%s) returned error: %v", output, err)
	}
	return sections
}

func objectKeys(t *testing.T, object json.RawMessage) []string {
	t.Helper()
	decoder := json.NewDecoder(bytes.NewReader(object))
	if _, err := decoder.Token(); err != nil {
		t.Fatalf("read opening token of %s: %v", object, err)
	}
	keys := []string{}
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			t.Fatalf("read key token of %s: %v", object, err)
		}
		key, ok := token.(string)
		if !ok {
			t.Fatalf("key token %v of %s is not a string", token, object)
		}
		keys = append(keys, key)
		var skipped json.RawMessage
		if err := decoder.Decode(&skipped); err != nil {
			t.Fatalf("skip value of %q in %s: %v", key, object, err)
		}
	}
	return keys
}

func TestWriteJSONEncodesKnownAndAbsentValues(t *testing.T) {
	t.Parallel()

	snapshot := status.Snapshot{
		Counters: []status.Field{
			newField("integer", "vectors", status.Int(3946)),
			newField("number", "", status.Float(40.8)),
			newField("flag", "", status.Bool(true)),
			newField("text", "", status.Text(`a <b> & "c"`+"\n")),
			newField("zero_integer", "", status.Int(0)),
			newField("zero_number", "", status.Float(0)),
			newField("false_flag", "", status.Bool(false)),
			newField("empty_text", "", status.Text("")),
			newField("absent", "", status.Value{}),
		},
	}

	output := encode(t, snapshot)
	counters := decode(t, output).Counters
	cases := map[string]string{
		"integer":      "3946",
		"number":       "40.8",
		"flag":         "true",
		"zero_integer": "0",
		"zero_number":  "0",
		"false_flag":   "false",
		"empty_text":   `""`,
		"absent":       "null",
	}
	for name, want := range cases {
		if got := string(counters[name].Value); got != want {
			t.Fatalf("%s value = %s, want %s", name, got, want)
		}
	}

	var text string
	if err := json.Unmarshal(counters["text"].Value, &text); err != nil {
		t.Fatalf("text value %s is not a JSON string: %v", counters["text"].Value, err)
	}
	if want := `a <b> & "c"` + "\n"; text != want {
		t.Fatalf("text value = %q, want %q", text, want)
	}
	if !strings.Contains(string(output), "<b> &") {
		t.Fatalf("output escaped HTML characters: %s", output)
	}

	if unit := counters["integer"].Unit; unit == nil || *unit != "vectors" {
		t.Fatalf("integer unit = %v, want vectors", unit)
	}
	if unit := counters["number"].Unit; unit != nil {
		t.Fatalf("number unit = %q, want no unit key", *unit)
	}
}

func TestWriteJSONKeepsSnapshotOrder(t *testing.T) {
	t.Parallel()

	snapshot := status.Snapshot{
		Notices: []string{"second notice sorts first", "a notice"},
		Identity: []status.Field{
			newField("version", "", status.Text("1")),
			newField("commit", "", status.Text("6e0a44c")),
			newField("pid", "", status.Text("7342")),
		},
		Counters: []status.Field{
			newField("zeta", "", status.Int(1)),
			newField("alpha", "", status.Int(2)),
			newField("mid", "", status.Value{}),
		},
		Activity: [][]status.Field{
			{
				newField("phase", "", status.Text("index")),
				newField("job_id", "", status.Text("job_a")),
			},
		},
	}

	output := encode(t, snapshot)
	wantSections := []string{"notices", "identity", "counters", "activity"}
	if got := objectKeys(t, output); !slices.Equal(got, wantSections) {
		t.Fatalf("top-level keys = %v, want %v", got, wantSections)
	}

	sections := topLevel(t, output)
	if got := objectKeys(t, sections["identity"]); !slices.Equal(got, []string{"version", "commit", "pid"}) {
		t.Fatalf("identity keys = %v", got)
	}
	if got := objectKeys(t, sections["counters"]); !slices.Equal(got, []string{"zeta", "alpha", "mid"}) {
		t.Fatalf("counters keys = %v", got)
	}
	var rows []json.RawMessage
	if err := json.Unmarshal(sections["activity"], &rows); err != nil {
		t.Fatalf("activity is not an array: %v", err)
	}
	if got := objectKeys(t, rows[0]); !slices.Equal(got, []string{"phase", "job_id"}) {
		t.Fatalf("activity row keys = %v", got)
	}
	if got := decode(t, output).Notices; !slices.Equal(got, snapshot.Notices) {
		t.Fatalf("notices = %v, want %v", got, snapshot.Notices)
	}
}

func TestWriteJSONEncodesTimeAndDuration(t *testing.T) {
	t.Parallel()

	readAt := time.Date(2026, time.July, 27, 12, 52, 31, 500, time.UTC)
	elapsed := 1500 * time.Millisecond
	snapshot := status.Snapshot{
		Identity: []status.Field{
			newField("read_at", "", status.Text(readAt.Format(time.RFC3339Nano))),
		},
		Counters: []status.Field{
			newField("elapsed", "seconds", status.Float(elapsed.Seconds())),
			newField("uptime", "ms", status.Int(elapsed.Milliseconds())),
			newField("last_sync", "", status.Value{}),
		},
	}

	decoded := decode(t, encode(t, snapshot))

	var gotReadAt time.Time
	if err := json.Unmarshal(decoded.Identity["read_at"].Value, &gotReadAt); err != nil {
		t.Fatalf("read_at %s is not an RFC 3339 string: %v", decoded.Identity["read_at"].Value, err)
	}
	if !gotReadAt.Equal(readAt) {
		t.Fatalf("read_at = %s, want %s", gotReadAt, readAt)
	}

	var seconds float64
	if err := json.Unmarshal(decoded.Counters["elapsed"].Value, &seconds); err != nil {
		t.Fatalf("elapsed %s is not a number: %v", decoded.Counters["elapsed"].Value, err)
	}
	if got := time.Duration(seconds * float64(time.Second)); got != elapsed {
		t.Fatalf("elapsed = %s, want %s", got, elapsed)
	}
	if unit := decoded.Counters["elapsed"].Unit; unit == nil || *unit != "seconds" {
		t.Fatalf("elapsed unit = %v, want seconds", unit)
	}

	var milliseconds int64
	if err := json.Unmarshal(decoded.Counters["uptime"].Value, &milliseconds); err != nil {
		t.Fatalf("uptime %s is not an integer: %v", decoded.Counters["uptime"].Value, err)
	}
	if got := time.Duration(milliseconds) * time.Millisecond; got != elapsed {
		t.Fatalf("uptime = %s, want %s", got, elapsed)
	}
	if unit := decoded.Counters["uptime"].Unit; unit == nil || *unit != "ms" {
		t.Fatalf("uptime unit = %v, want ms", unit)
	}

	if got := string(decoded.Counters["last_sync"].Value); got != "null" {
		t.Fatalf("last_sync = %s, want null", got)
	}
}

func TestWriteJSONEncodesActivityRows(t *testing.T) {
	t.Parallel()

	snapshot := status.Snapshot{
		Activity: [][]status.Field{
			{newField("job_id", "", status.Text("job_a"))},
			{},
			{
				newField("job_id", "", status.Value{}),
				newField("pending_paths", "paths", status.Int(8)),
			},
		},
	}

	activity := decode(t, encode(t, snapshot)).Activity
	if len(activity) != 3 {
		t.Fatalf("activity rows = %d, want 3", len(activity))
	}
	if got := string(activity[0]["job_id"].Value); got != `"job_a"` {
		t.Fatalf("row 0 job_id = %s, want \"job_a\"", got)
	}
	if activity[1] == nil || len(activity[1]) != 0 {
		t.Fatalf("row 1 = %v, want an empty object", activity[1])
	}
	if got := string(activity[2]["job_id"].Value); got != "null" {
		t.Fatalf("row 2 job_id = %s, want null", got)
	}
	if got := string(activity[2]["pending_paths"].Value); got != "8" {
		t.Fatalf("row 2 pending_paths = %s, want 8", got)
	}
	if unit := activity[2]["pending_paths"].Unit; unit == nil || *unit != "paths" {
		t.Fatalf("row 2 pending_paths unit = %v, want paths", unit)
	}
}

func TestWriteJSONEncodesEmptySnapshot(t *testing.T) {
	t.Parallel()

	output := encode(t, status.Snapshot{})
	if !bytes.HasSuffix(output, []byte("\n")) || bytes.Count(output, []byte("\n")) != 1 {
		t.Fatalf("output is not one line ending in a newline: %q", output)
	}
	sections := topLevel(t, output)
	cases := map[string]string{
		"notices":  "[]",
		"identity": "{}",
		"counters": "{}",
		"activity": "[]",
	}
	if len(sections) != len(cases) {
		t.Fatalf("top-level keys = %d, want %d: %s", len(sections), len(cases), output)
	}
	for name, want := range cases {
		if got := string(sections[name]); got != want {
			t.Fatalf("%s = %s, want %s", name, got, want)
		}
	}
}

func TestWriteJSONRejectsNonFiniteNumber(t *testing.T) {
	t.Parallel()

	snapshot := status.Snapshot{
		Counters: []status.Field{newField("rate", "", status.Float(math.NaN()))},
	}
	var output bytes.Buffer
	if err := status.WriteJSON(&output, snapshot); err == nil {
		t.Fatalf("WriteJSON returned no error for NaN; output: %s", output.Bytes())
	}
	if output.Len() != 0 {
		t.Fatalf("WriteJSON wrote %q before returning the error", output.Bytes())
	}
}

func TestWriteJSONAndDumpHaveTheSameKeys(t *testing.T) {
	t.Parallel()

	snapshot := status.Snapshot{
		Notices: []string{"maintenance window open", "dependency degraded"},
		Identity: []status.Field{
			newField("version", "", status.Text("202607270542-fe-6e0a44c")),
			newField("pid", "", status.Text("7342")),
			newField("read_at", "", status.Text("2026-07-27T12:52:31Z")),
		},
		Counters: []status.Field{
			newField("embed_vectors_total", "vectors", status.Int(3946)),
			newField("overall_percent", "%", status.Float(40.8)),
			newField("dependency_health.degraded", "", status.Bool(false)),
			newField("dependency_health.mode", "", status.Text("")),
			newField("last_sync", "", status.Value{}),
		},
		Activity: [][]status.Field{
			{newField("job_id", "", status.Text("job_a"))},
			{
				newField("job_id", "", status.Value{}),
				newField("pending_paths", "paths", status.Int(8)),
			},
		},
	}

	dumpLines := strings.Split(status.Dump(snapshot), "\n")
	noticeCount := len(snapshot.Notices)
	dumpNotices := dumpLines[:noticeCount]
	dumpKeys := []string{}
	for _, line := range dumpLines[noticeCount:] {
		name, _, _ := strings.Cut(line, " ")
		dumpKeys = append(dumpKeys, name)
	}

	output := encode(t, snapshot)
	sections := topLevel(t, output)
	jsonKeys := objectKeys(t, sections["identity"])
	jsonKeys = append(jsonKeys, objectKeys(t, sections["counters"])...)
	var rows []json.RawMessage
	if err := json.Unmarshal(sections["activity"], &rows); err != nil {
		t.Fatalf("activity is not an array: %v", err)
	}
	for index, row := range rows {
		prefix := "activity." + strconv.Itoa(index) + "."
		for _, name := range objectKeys(t, row) {
			jsonKeys = append(jsonKeys, prefix+name)
		}
	}

	if !slices.Equal(jsonKeys, dumpKeys) {
		t.Fatalf("JSON keys = %v, Dump keys = %v", jsonKeys, dumpKeys)
	}
	if got := decode(t, output).Notices; !slices.Equal(got, dumpNotices) {
		t.Fatalf("JSON notices = %v, Dump notices = %v", got, dumpNotices)
	}
}
