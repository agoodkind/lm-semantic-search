package status

import (
	"strconv"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// unusedSource stands in where a test never refreshes the screen.
func unusedSource() (Snapshot, error) {
	return Snapshot{}, nil
}

// TestGroupDigitsSeparatesThousands proves the terminal groups digits, which
// makes a value crossing a digit boundary visible without reading it.
func TestGroupDigitsSeparatesThousands(t *testing.T) {
	t.Parallel()

	cases := map[string]string{
		"0":         "0",
		"824":       "824",
		"1524":      "1,524",
		"249278160": "249,278,160",
		"-1":        "-1",
		"-12345":    "-12,345",
	}
	for input, want := range cases {
		if got := groupDigits(input); got != want {
			t.Fatalf("groupDigits(%q) = %q, want %q", input, got, want)
		}
	}
}

// TestDeltaTextSignsTheChange proves a delta always has a sign, and a reader
// never infers direction. An absent previous read reports nothing rather than a
// change of zero it did not observe.
func TestDeltaTextSignsTheChange(t *testing.T) {
	t.Parallel()

	cases := []struct {
		delta       int64
		hasPrevious bool
		want        string
	}{
		{72, true, "+72"},
		{-1, true, "-1"},
		{16104, true, "+16,104"},
		{0, true, "+0"},
		{0, false, ""},
		{72, false, ""},
	}
	for _, testCase := range cases {
		got := deltaText(testCase.delta, testCase.hasPrevious)
		if got != testCase.want {
			t.Fatalf("deltaText(%d, %t) = %q, want %q",
				testCase.delta, testCase.hasPrevious, got, testCase.want)
		}
	}
}

// TestStatusCounterBlockRendersAbsentAndGroupedValues proves the screen prints
// null for an absent value, and a missing fact never renders as a zero. It also
// proves an integer is digit-grouped.
func TestStatusCounterBlockRendersAbsentAndGroupedValues(t *testing.T) {
	t.Parallel()

	snapshot := Snapshot{
		Counters: []Field{
			testField("dependency_health", "dependency_health.since", "", Value{}),
			testField("dependency_health", "dependency_health.mode", "", Text("")),
			testField("embed", "embed_vectors_total", "vectors", Int(3946)),
		},
	}
	body := statusCounterBlock(snapshot, nil, 100)
	for _, want := range []string{"null", `""`, "3,946", "vectors"} {
		if !strings.Contains(body, want) {
			t.Fatalf("missing %q in:\n%s", want, body)
		}
	}
}

// TestStatusCounterBlockReportsTheChange proves a second read renders the
// signed change beside the value. The change tells a moving pipeline apart
// from a stalled one.
func TestStatusCounterBlockReportsTheChange(t *testing.T) {
	t.Parallel()

	snapshot := Snapshot{
		Counters: []Field{
			testField("embed", "embed_vectors_total", "vectors", Int(3946)),
			testField("runtime", "num_goroutine", "goroutines", Int(34)),
		},
	}
	previous := map[string]int64{"embed_vectors_total": 3874, "num_goroutine": 35}

	body := statusCounterBlock(snapshot, previous, 100)
	if !strings.Contains(body, "+72") {
		t.Fatalf("rising counter did not report its change:\n%s", body)
	}
	if !strings.Contains(body, "-1") {
		t.Fatalf("falling counter did not report its change:\n%s", body)
	}
}

// TestStatusCounterBlockOmitsADeltaForAnUnseenCounter proves a counter absent
// from the previous read reports no change, rather than a change measured
// against a zero that was never observed.
func TestStatusCounterBlockOmitsADeltaForAnUnseenCounter(t *testing.T) {
	t.Parallel()

	field := testField("embed", "embed_vectors_total", "vectors", Int(3946))
	if got := statusDeltaText(field, map[string]int64{"other": 1}); got != "" {
		t.Fatalf("delta for an unseen counter = %q, want empty", got)
	}
	if got := statusDeltaText(field, nil); got != "" {
		t.Fatalf("delta with no previous read = %q, want empty", got)
	}
}

// TestStatusActivityRowNamesTheAbsentJob proves a file-change row renders its
// missing job id as null. The screen states why the job commands cannot
// address that work.
func TestStatusActivityRowNamesTheAbsentJob(t *testing.T) {
	t.Parallel()

	row := []Field{
		testField("", "job_id", "", Value{}),
		testField("", "source", "", Text("watcher")),
		testField("", "pending_paths", "paths", Int(8)),
	}
	body := strings.Join(statusActivityRow(1, row, 100), "\n")
	for _, want := range []string{"[1] job_id=null", "source=watcher", "pending_paths=8"} {
		if !strings.Contains(body, want) {
			t.Fatalf("missing %q in:\n%s", want, body)
		}
	}
}

// TestIntegerValuesByNameKeepsOnlyIntegers proves the delta baseline keeps only
// values a difference is meaningful for. A timestamp or a percentage subtracted
// between two reads reports nothing an operator can act on.
func TestIntegerValuesByNameKeepsOnlyIntegers(t *testing.T) {
	t.Parallel()

	snapshot := Snapshot{
		Counters: []Field{
			testField("", "embed_vectors_total", "", Int(3946)),
			testField("", "overall_percent", "", Float(40.8)),
			testField("", "dependency_health.mode", "", Text("")),
			testField("", "dependency_health.since", "", Value{}),
		},
	}
	values := integerValuesByName(snapshot)
	if len(values) != 1 {
		t.Fatalf("baseline has %d values, want 1: %+v", len(values), values)
	}
	if values["embed_vectors_total"] != 3946 {
		t.Fatalf("embed_vectors_total = %d, want 3946", values["embed_vectors_total"])
	}
}

// vectorSnapshot builds a snapshot with one counter for a run named runID.
func vectorSnapshot(runID string, vectors int64) Snapshot {
	return Snapshot{
		Title: "test",
		RunID: runID,
		Counters: []Field{
			testField("embed", "embed_vectors_total", "vectors", Int(vectors)),
		},
	}
}

// TestApplyRefreshDropsTheBaselineAcrossARestart proves the model drops the
// baseline when the run changes. The first read after a restart shows no delta
// rather than a negative one the operator never observed.
func TestApplyRefreshDropsTheBaselineAcrossARestart(t *testing.T) {
	t.Parallel()

	model := newStatusModel(unusedSource, time.Second, time.Now, vectorSnapshot("7342@1", 3946))
	sameRun := model.applyRefresh(statusRefreshedMsg{snapshot: vectorSnapshot("7342@1", 4018)})
	if sameRun.previous["embed_vectors_total"] != 3946 {
		t.Fatalf("baseline within one run = %d, want 3946", sameRun.previous["embed_vectors_total"])
	}

	restarted := model.applyRefresh(statusRefreshedMsg{snapshot: vectorSnapshot("9001@1", 12)})
	if restarted.previous != nil {
		t.Fatalf("baseline survived a restart: %+v", restarted.previous)
	}

	unidentified := newStatusModel(unusedSource, time.Second, time.Now, vectorSnapshot("", 3946))
	missingRun := unidentified.applyRefresh(statusRefreshedMsg{snapshot: vectorSnapshot("", 4018)})
	if missingRun.previous != nil {
		t.Fatalf("baseline survived a snapshot with no run id: %+v", missingRun.previous)
	}
}

// TestApplyRefreshDropsTheBaselineAfterAResumedPause proves a resumed pause
// reports no delta on its first read, since the gap the operator chose is not
// the interval the header names.
func TestApplyRefreshDropsTheBaselineAfterAResumedPause(t *testing.T) {
	t.Parallel()

	model := newStatusModel(unusedSource, time.Second, time.Now, vectorSnapshot("7342@1", 3946))
	paused, _ := model.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'p'}})
	pausedModel, ok := paused.(statusModel)
	if !ok || !pausedModel.paused {
		t.Fatal("p did not pause the screen")
	}
	resumed, _ := pausedModel.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'p'}})
	resumedModel, ok := resumed.(statusModel)
	if !ok || resumedModel.paused {
		t.Fatal("p did not resume the screen")
	}

	afterResume := resumedModel.applyRefresh(statusRefreshedMsg{snapshot: vectorSnapshot("7342@1", 9999)})
	if afterResume.previous != nil {
		t.Fatalf("baseline survived a resumed pause: %+v", afterResume.previous)
	}
	// The read after that compares normally again.
	next := afterResume.applyRefresh(statusRefreshedMsg{snapshot: vectorSnapshot("7342@1", 10004)})
	if next.previous["embed_vectors_total"] != 9999 {
		t.Fatalf("baseline after one post-resume read = %d, want 9999", next.previous["embed_vectors_total"])
	}
}

// TestViewFitsTheTerminalHeight proves the frame never exceeds the terminal, and
// bubbletea cannot clip the pinned header off the top. The counter block alone
// is taller than a standard terminal, and the body scrolls for that reason.
func TestViewFitsTheTerminalHeight(t *testing.T) {
	t.Parallel()

	counters := make([]Field, 0, 40)
	for index := range 40 {
		counters = append(counters,
			testField("embed", "counter_"+strconv.Itoa(index), "things", Int(int64(index))))
	}
	snapshot := Snapshot{
		Title:    "lm-semantic-search  version=test  pid=7342",
		Counters: counters,
		Activity: [][]Field{{
			testField("", "job_id", "", Value{}),
			testField("", "state", "", Text("running")),
		}},
	}

	for _, height := range []int{10, 24, 40} {
		model := newStatusModel(unusedSource, time.Second, time.Now, snapshot)
		model.width = 100
		model.height = height
		lines := strings.Split(strings.TrimRight(model.View(), "\n"), "\n")
		if len(lines) > height {
			t.Fatalf("frame is %d lines on a %d-row terminal and the header is clipped",
				len(lines), height)
		}
		if !strings.Contains(lines[0], "lm-semantic-search") {
			t.Fatalf("first line is not the header at height %d: %q", height, lines[0])
		}
	}
}

// TestStatusActivityRowSurvivesAnEmptyRow proves the screen renders a row with
// no fields instead of panicking. A source can return an empty row, for example
// after a version mismatch with the program it reads, and that must not crash
// the command.
func TestStatusActivityRowSurvivesAnEmptyRow(t *testing.T) {
	t.Parallel()

	lines := statusActivityRow(0, nil, 100)
	if len(lines) != 1 {
		t.Fatalf("lines = %d, want 1 for a row with no fields: %v", len(lines), lines)
	}
	if !strings.Contains(lines[0], "[0]") {
		t.Fatalf("empty row lost its index: %q", lines[0])
	}
}

// TestStatusCounterBlockSurvivesAnEmptyReply proves the counter block renders
// nothing rather than panicking when a snapshot has no counters at all.
func TestStatusCounterBlockSurvivesAnEmptyReply(t *testing.T) {
	t.Parallel()

	if body := statusCounterBlock(Snapshot{}, nil, 100); body != "" {
		t.Fatalf("empty snapshot rendered %q, want empty", body)
	}
}

// TestViewSurvivesAnEmptyReply proves the whole frame renders on a snapshot
// with neither counters nor activity, the state of a program before it has done
// any work.
func TestViewSurvivesAnEmptyReply(t *testing.T) {
	t.Parallel()

	model := newStatusModel(unusedSource, time.Second, time.Now, Snapshot{
		Title: "lm-semantic-search  version=test  pid=7342",
	})
	model.width = 100
	model.height = 24

	frame := model.View()
	if !strings.Contains(frame, "lm-semantic-search") {
		t.Fatalf("frame lost its header on an empty snapshot:\n%s", frame)
	}
	if !strings.Contains(frame, "none running, none queued") {
		t.Fatalf("frame does not state nothing is running:\n%s", frame)
	}
}

// TestScrollDownStopsAtTheLastLine proves the offset cannot climb past the end
// of the body. Without the clamp on the keypress, holding Down would raise the
// offset without bound and the same number of Up presses would appear to do
// nothing before the screen finally moved.
func TestScrollDownStopsAtTheLastLine(t *testing.T) {
	t.Parallel()

	snapshot := Snapshot{
		Title: "lm-semantic-search  version=test  pid=7342",
		Counters: []Field{
			testField("embed", "embed_vectors_total", "vectors", Int(3946)),
		},
	}

	model := newStatusModel(unusedSource, time.Second, time.Now, snapshot)
	model.width = 100
	model.height = 24

	ceiling := model.maxOffset()
	current := tea.Model(model)
	for range 500 {
		next, _ := current.(statusModel).handleKey(tea.KeyMsg{Type: tea.KeyDown})
		current = next
	}
	if got := current.(statusModel).offset; got != ceiling {
		t.Fatalf("offset after 500 Down presses = %d, want the ceiling %d", got, ceiling)
	}

	// One Up press must move the view immediately, not burn against a backlog.
	after, _ := current.(statusModel).handleKey(tea.KeyMsg{Type: tea.KeyUp})
	if got := after.(statusModel).offset; got != max(ceiling-1, 0) {
		t.Fatalf("offset after one Up = %d, want %d", got, max(ceiling-1, 0))
	}
}

// TestNoDeltaFieldShowsNoDelta proves a value that cannot change has no delta
// column. A fixed value would otherwise print +0 beside it on every refresh,
// which adds noise to every line.
func TestNoDeltaFieldShowsNoDelta(t *testing.T) {
	t.Parallel()

	constant := testField("jobs", "index_slots_total", "slots", Int(4))
	constant.NoDelta = true
	changing := testField("jobs", "index_slots_in_use", "slots", Int(2))
	previous := map[string]int64{"index_slots_total": 4, "index_slots_in_use": 2}

	if got := statusDeltaText(constant, previous); got != "" {
		t.Fatalf("constant field delta = %q, want empty", got)
	}
	if got := statusDeltaText(changing, previous); got != "+0" {
		t.Fatalf("changing field delta = %q, want \"+0\"", got)
	}
}

// TestTimestampsRenderInTheHostZone proves the screen converts a UTC timestamp
// into the host's zone with an offset, and leaves anything that is not a
// timestamp alone. The program reports one zone; this is the only surface that
// converts, and it converts through the one shared lookup.
func TestTimestampsRenderInTheHostZone(t *testing.T) {
	t.Parallel()

	instant := "2026-07-27T12:52:31Z"
	rendered := displayTimeText(instant)

	parsed, err := time.Parse("2006-01-02T15:04:05-07:00", rendered)
	if err != nil {
		t.Fatalf("rendered %q does not parse as a timestamp with offset: %v", rendered, err)
	}
	expected, err := time.Parse(time.RFC3339Nano, instant)
	if err != nil {
		t.Fatalf("parse the source instant: %v", err)
	}
	if !parsed.Equal(expected) {
		t.Fatalf("rendered %q is a different instant from %q", rendered, instant)
	}

	// A path is not a timestamp and must survive untouched.
	if got := displayTimeText("/Users/a/code"); got != "/Users/a/code" {
		t.Fatalf("a path was rewritten to %q", got)
	}
	// So must a phase name.
	if got := displayTimeText("embedding"); got != "embedding" {
		t.Fatalf("a phase name was rewritten to %q", got)
	}
}
