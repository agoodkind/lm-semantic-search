package main

import (
	"strconv"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// TestGroupDigitsSeparatesThousands proves the terminal groups digits, which is
// what makes a value crossing a digit boundary visible without reading it.
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

// TestDeltaTextSignsTheChange proves a delta always carries its sign, so a
// reader never infers direction, and that an absent previous read reports
// nothing rather than a change of zero it did not observe.
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
// null for a metric with no value, so an absent fact never renders as a zero,
// and that an integer is digit-grouped.
func TestStatusCounterBlockRendersAbsentAndGroupedValues(t *testing.T) {
	t.Parallel()

	response := &pb.GetStatusResponse{
		Metrics: []*pb.Metric{
			{Group: "dependency_health", Name: "dependency_health.since"},
			{
				Group: "dependency_health", Name: "dependency_health.mode",
				Value: &pb.Metric_StringValue{StringValue: ""},
			},
			{
				Group: "embed", Name: "embed_vectors_total", Unit: "vectors",
				Value: &pb.Metric_IntValue{IntValue: 3946},
			},
		},
	}
	body := statusCounterBlock(response, nil, 100)
	for _, want := range []string{"null", `""`, "3,946", "vectors"} {
		if !strings.Contains(body, want) {
			t.Fatalf("missing %q in:\n%s", want, body)
		}
	}
}

// TestStatusCounterBlockReportsTheChange proves a second read renders the
// signed change beside the value, which is what tells a moving pipeline apart
// from a stalled one.
func TestStatusCounterBlockReportsTheChange(t *testing.T) {
	t.Parallel()

	response := &pb.GetStatusResponse{
		Metrics: []*pb.Metric{
			{
				Group: "embed", Name: "embed_vectors_total", Unit: "vectors",
				Value: &pb.Metric_IntValue{IntValue: 3946},
			},
			{
				Group: "runtime", Name: "num_goroutine", Unit: "goroutines",
				Value: &pb.Metric_IntValue{IntValue: 34},
			},
		},
	}
	previous := map[string]int64{"embed_vectors_total": 3874, "num_goroutine": 35}

	body := statusCounterBlock(response, previous, 100)
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

	metric := &pb.Metric{
		Group: "embed", Name: "embed_vectors_total", Unit: "vectors",
		Value: &pb.Metric_IntValue{IntValue: 3946},
	}
	if got := statusDeltaText(metric, map[string]int64{"other": 1}); got != "" {
		t.Fatalf("delta for an unseen counter = %q, want empty", got)
	}
	if got := statusDeltaText(metric, nil); got != "" {
		t.Fatalf("delta with no previous read = %q, want empty", got)
	}
}

// TestStatusActivityRowNamesTheAbsentJob proves a file-change row renders its
// missing job id as null, so the screen states why the job commands cannot
// address that work.
func TestStatusActivityRowNamesTheAbsentJob(t *testing.T) {
	t.Parallel()

	row := &pb.ActivityRow{Metrics: []*pb.Metric{
		{Name: "job_id"},
		{Name: "source", Value: &pb.Metric_StringValue{StringValue: "watcher"}},
		{Name: "pending_paths", Unit: "paths", Value: &pb.Metric_IntValue{IntValue: 8}},
	}}
	body := strings.Join(statusActivityRow(1, row, 100), "\n")
	for _, want := range []string{"[1] job_id=null", "source=watcher", "pending_paths=8"} {
		if !strings.Contains(body, want) {
			t.Fatalf("missing %q in:\n%s", want, body)
		}
	}
}

// TestIntegerValuesByNameKeepsOnlyIntegers proves the delta baseline holds only
// values a difference is meaningful for. A timestamp or a percentage subtracted
// between two reads reports nothing an operator can act on.
func TestIntegerValuesByNameKeepsOnlyIntegers(t *testing.T) {
	t.Parallel()

	response := &pb.GetStatusResponse{
		Metrics: []*pb.Metric{
			{Name: "embed_vectors_total", Value: &pb.Metric_IntValue{IntValue: 3946}},
			{Name: "overall_percent", Value: &pb.Metric_DoubleValue{DoubleValue: 40.8}},
			{Name: "dependency_health.mode", Value: &pb.Metric_StringValue{StringValue: ""}},
			{Name: "dependency_health.since"},
		},
	}
	values := integerValuesByName(response)
	if len(values) != 1 {
		t.Fatalf("baseline holds %d values, want 1: %+v", len(values), values)
	}
	if values["embed_vectors_total"] != 3946 {
		t.Fatalf("embed_vectors_total = %d, want 3946", values["embed_vectors_total"])
	}
}

// TestSameDaemonRunDetectsARestart proves the delta baseline is dropped when
// the two reads did not come from one continuous run. A restarted daemon zeroes
// every process counter, so subtracting across it would report large negative
// changes as if work were being undone.
func TestSameDaemonRunDetectsARestart(t *testing.T) {
	t.Parallel()

	first := time.Unix(1785156767, 0)
	second := time.Unix(1785160000, 0)

	reply := func(pid int32, startedAt time.Time) *pb.GetStatusResponse {
		return &pb.GetStatusResponse{
			Daemon: &pb.DaemonIdentity{Pid: pid, StartedAt: timestamppb.New(startedAt)},
		}
	}

	if !sameDaemonRun(reply(7342, first), reply(7342, first)) {
		t.Fatal("two reads of one run reported as different runs")
	}
	if sameDaemonRun(reply(7342, first), reply(9001, first)) {
		t.Fatal("a different pid reported as the same run")
	}
	if sameDaemonRun(reply(7342, first), reply(7342, second)) {
		t.Fatal("a different start time reported as the same run")
	}
	if sameDaemonRun(nil, reply(7342, first)) {
		t.Fatal("a missing previous reply reported as the same run")
	}
}

// TestApplyRefreshDropsTheBaselineAcrossARestart proves the model actually acts
// on that, so the first read after a restart shows no delta rather than a
// negative one the operator never observed.
func TestApplyRefreshDropsTheBaselineAcrossARestart(t *testing.T) {
	t.Parallel()

	reply := func(pid int32, vectors int64) *pb.GetStatusResponse {
		return &pb.GetStatusResponse{
			Daemon: &pb.DaemonIdentity{Pid: pid, StartedAt: timestamppb.New(time.Unix(1785156767, 0))},
			Metrics: []*pb.Metric{{
				Group: "embed", Name: "embed_vectors_total", Unit: "vectors",
				Value: &pb.Metric_IntValue{IntValue: vectors},
			}},
		}
	}

	model := newStatusModel(cliOptions{}, time.Second, reply(7342, 3946))
	sameRun := model.applyRefresh(statusRefreshedMsg{response: reply(7342, 4018)})
	if sameRun.previous["embed_vectors_total"] != 3946 {
		t.Fatalf("baseline within one run = %d, want 3946", sameRun.previous["embed_vectors_total"])
	}

	restarted := model.applyRefresh(statusRefreshedMsg{response: reply(9001, 12)})
	if restarted.previous != nil {
		t.Fatalf("baseline survived a restart: %+v", restarted.previous)
	}
}

// TestApplyRefreshDropsTheBaselineAfterAResumedPause proves a resumed pause
// reports no delta on its first read, since the gap the operator chose is not
// the interval the header names.
func TestApplyRefreshDropsTheBaselineAfterAResumedPause(t *testing.T) {
	t.Parallel()

	reply := func(vectors int64) *pb.GetStatusResponse {
		return &pb.GetStatusResponse{
			Daemon: &pb.DaemonIdentity{Pid: 7342, StartedAt: timestamppb.New(time.Unix(1785156767, 0))},
			Metrics: []*pb.Metric{{
				Group: "embed", Name: "embed_vectors_total", Unit: "vectors",
				Value: &pb.Metric_IntValue{IntValue: vectors},
			}},
		}
	}

	model := newStatusModel(cliOptions{}, time.Second, reply(3946))
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

	afterResume := resumedModel.applyRefresh(statusRefreshedMsg{response: reply(9999)})
	if afterResume.previous != nil {
		t.Fatalf("baseline survived a resumed pause: %+v", afterResume.previous)
	}
	// The read after that compares normally again.
	next := afterResume.applyRefresh(statusRefreshedMsg{response: reply(10004)})
	if next.previous["embed_vectors_total"] != 9999 {
		t.Fatalf("baseline after one post-resume read = %d, want 9999", next.previous["embed_vectors_total"])
	}
}

// TestViewFitsTheTerminalHeight proves the frame never exceeds the terminal, so
// bubbletea cannot clip the pinned header off the top. The counter block alone
// is taller than a standard terminal, which is why the body scrolls.
func TestViewFitsTheTerminalHeight(t *testing.T) {
	t.Parallel()

	metricsList := make([]*pb.Metric, 0, 40)
	for index := range 40 {
		metricsList = append(metricsList, &pb.Metric{
			Group: "embed", Name: "counter_" + strconv.Itoa(index), Unit: "things",
			Value: &pb.Metric_IntValue{IntValue: int64(index)},
		})
	}
	response := &pb.GetStatusResponse{
		Daemon:  &pb.DaemonIdentity{Pid: 7342, Version: "test"},
		Metrics: metricsList,
		Activity: []*pb.ActivityRow{{Metrics: []*pb.Metric{
			{Name: "job_id"},
			{Name: "state", Value: &pb.Metric_StringValue{StringValue: "running"}},
		}}},
	}

	for _, height := range []int{10, 24, 40} {
		model := newStatusModel(cliOptions{}, time.Second, response)
		model.width = 100
		model.height = height
		lines := strings.Split(strings.TrimRight(model.View(), "\n"), "\n")
		if len(lines) > height {
			t.Fatalf("frame is %d lines on a %d-row terminal, so the header is clipped",
				len(lines), height)
		}
		if !strings.Contains(lines[0], "lm-semantic-search") {
			t.Fatalf("first line is not the header at height %d: %q", height, lines[0])
		}
	}
}

// TestStatusActivityRowSurvivesAnEmptyRow proves the screen renders a row that
// carries no fields instead of panicking. Every row this daemon builds carries
// fields, but the reply arrives over a socket, so a daemon of another version
// or a truncated message must not crash the command.
func TestStatusActivityRowSurvivesAnEmptyRow(t *testing.T) {
	t.Parallel()

	lines := statusActivityRow(0, &pb.ActivityRow{}, 100)
	if len(lines) != 1 {
		t.Fatalf("lines = %d, want 1 for a row with no fields: %v", len(lines), lines)
	}
	if !strings.Contains(lines[0], "[0]") {
		t.Fatalf("empty row lost its index: %q", lines[0])
	}
}

// TestStatusCounterBlockSurvivesAnEmptyReply proves the counter block renders
// nothing rather than panicking when a reply carries no metrics at all.
func TestStatusCounterBlockSurvivesAnEmptyReply(t *testing.T) {
	t.Parallel()

	if body := statusCounterBlock(&pb.GetStatusResponse{}, nil, 100); body != "" {
		t.Fatalf("empty reply rendered %q, want empty", body)
	}
}

// TestViewSurvivesAnEmptyReply proves the whole frame renders on a reply that
// carries neither metrics nor activity, which is what a daemon reports before
// it has done any work.
func TestViewSurvivesAnEmptyReply(t *testing.T) {
	t.Parallel()

	model := newStatusModel(cliOptions{}, time.Second, &pb.GetStatusResponse{
		Daemon: &pb.DaemonIdentity{Pid: 7342, Version: "test"},
	})
	model.width = 100
	model.height = 24

	frame := model.View()
	if !strings.Contains(frame, "lm-semantic-search") {
		t.Fatalf("frame lost its header on an empty reply:\n%s", frame)
	}
	if !strings.Contains(frame, "none running, none queued") {
		t.Fatalf("frame does not state the daemon is idle:\n%s", frame)
	}
}

// TestScrollDownStopsAtTheLastLine proves the offset cannot climb past the end
// of the body. Without the clamp on the keypress, holding Down would raise the
// offset without bound and the same number of Up presses would appear to do
// nothing before the screen finally moved.
func TestScrollDownStopsAtTheLastLine(t *testing.T) {
	t.Parallel()

	response := &pb.GetStatusResponse{
		Daemon: &pb.DaemonIdentity{Pid: 7342, Version: "test"},
		Metrics: []*pb.Metric{
			{
				Group: "embed", Name: "embed_vectors_total", Unit: "vectors",
				Value: &pb.Metric_IntValue{IntValue: 3946},
			},
		},
	}

	model := newStatusModel(cliOptions{}, time.Second, response)
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

// TestConstantMetricShowsNoDelta proves a value that cannot change carries no
// delta column. index_slots_total is fixed for the life of the process, so a
// +0 beside it every refresh would be noise on every line.
func TestConstantMetricShowsNoDelta(t *testing.T) {
	t.Parallel()

	constant := &pb.Metric{
		Group: "jobs", Name: "index_slots_total", Unit: "slots",
		Value: &pb.Metric_IntValue{IntValue: 4},
	}
	changing := &pb.Metric{
		Group: "jobs", Name: "index_slots_in_use", Unit: "slots",
		Value: &pb.Metric_IntValue{IntValue: 2},
	}
	previous := map[string]int64{"index_slots_total": 4, "index_slots_in_use": 2}

	if got := statusDeltaText(constant, previous); got != "" {
		t.Fatalf("constant metric delta = %q, want empty", got)
	}
	if got := statusDeltaText(changing, previous); got != "+0" {
		t.Fatalf("changing metric delta = %q, want \"+0\"", got)
	}
}

// TestTimestampsRenderInTheHostZone proves the screen converts the daemon's
// UTC timestamp into the host's zone with an offset, and leaves anything that
// is not a timestamp alone. The daemon reports one zone; this is the only
// surface that converts, and it converts through the one shared lookup.
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
