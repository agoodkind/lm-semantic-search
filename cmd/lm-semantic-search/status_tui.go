package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	render "goodkind.io/lm-semantic-search/internal/render"
)

// statusValueGap and statusUnitGap separate the four columns. They are constants
// so the layout is identical between two renders of the same terminal width.
const (
	statusValueGap = 2
	statusUnitGap  = 2
	statusUnitWide = 10
)

// runStatusTUI drives the live status screen until the operator quits. It polls
// rather than subscribes: WatchJobs sends one message per requested job id and
// returns, so it is a snapshot rather than a subscription and cannot drive a
// refreshing screen.
func runStatusTUI(options cliOptions, interval time.Duration) error {
	first, err := fetchStatusResponse(options)
	if err != nil {
		return err
	}
	program := tea.NewProgram(newStatusModel(options, interval, first), tea.WithAltScreen())
	if _, runErr := program.Run(); runErr != nil {
		slog.Error("run status TUI failed", "err", runErr)
		return fmt.Errorf("run status screen: %w", runErr)
	}
	return nil
}

// statusModel is the bubbletea state for the live screen. It keeps the previous
// read's integer values so each refresh can report the change, and it keeps the
// last successful read time so a failed refresh never reads as a quiet system.
type statusModel struct {
	options  cliOptions
	interval time.Duration
	response *pb.GetStatusResponse
	previous map[string]int64
	// comparable records that the next read may be subtracted from the current
	// one. It is false after a resume, because the gap the operator chose is not
	// the interval the header names.
	comparable bool
	readAt     time.Time
	refreshErr error
	paused     bool
	refreshing bool
	width      int
	height     int
	offset     int
	quitting   bool
}

type statusRefreshedMsg struct {
	response *pb.GetStatusResponse
	err      error
}

type statusTickMsg struct{}

func newStatusModel(options cliOptions, interval time.Duration, first *pb.GetStatusResponse) statusModel {
	return statusModel{
		options:    options,
		interval:   interval,
		response:   first,
		previous:   nil,
		comparable: true,
		readAt:     time.Now(),
		refreshErr: nil,
		paused:     false,
		refreshing: false,
		width:      0,
		height:     0,
		offset:     0,
		quitting:   false,
	}
}

func (m statusModel) Init() tea.Cmd {
	return statusTick(m.interval)
}

func (m statusModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch typed := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = typed.Width
		m.height = typed.Height
		return m, nil
	case tea.KeyMsg:
		return m.handleKey(typed)
	case statusRefreshedMsg:
		return m.applyRefresh(typed), nil
	case statusTickMsg:
		cmds := []tea.Cmd{statusTick(m.interval)}
		if !m.paused && !m.refreshing {
			m.refreshing = true
			cmds = append(cmds, statusRefreshCmd(m.options))
		}
		return m, tea.Batch(cmds...)
	}
	return m, nil
}

func (m statusModel) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch {
	case keyMatches(msg, "ctrl+c", "q", "esc"):
		m.quitting = true
		return m, tea.Quit
	case keyMatches(msg, "p"):
		m.paused = !m.paused
		// Resuming drops the baseline, because the next read would otherwise
		// report a change spanning the whole pause while the header still names
		// the poll interval.
		if !m.paused {
			m.comparable = false
		}
		return m, nil
	case keyMatches(msg, "r"):
		if m.refreshing {
			return m, nil
		}
		m.refreshing = true
		return m, statusRefreshCmd(m.options)
	case keyMatches(msg, "down", "j"):
		m.offset++
		return m, nil
	case keyMatches(msg, "up", "k"):
		m.offset = max(m.offset-1, 0)
		return m, nil
	default:
		return m, nil
	}
}

// applyRefresh swaps in a fresh reply and keeps the prior integer values so the
// next render can report the change. A failed refresh keeps the previous reply
// and the previous read time, so the screen states it is stale rather than
// showing an empty one.
//
// The baseline is dropped whenever the two reads did not observe one continuous
// run of one process. A restarted daemon zeroes every counter, so subtracting
// across it reports large negative changes as if work were being undone, and a
// resumed pause would report a change spanning the whole pause under a header
// still claiming the poll interval.
func (m statusModel) applyRefresh(msg statusRefreshedMsg) statusModel {
	m.refreshing = false
	if msg.err != nil {
		m.refreshErr = msg.err
		return m
	}
	m.refreshErr = nil
	if sameDaemonRun(m.response, msg.response) && m.comparable {
		m.previous = integerValuesByName(m.response)
	} else {
		m.previous = nil
	}
	m.comparable = true
	m.response = msg.response
	m.readAt = time.Now()
	return m
}

// sameDaemonRun reports whether two replies came from one continuous run of one
// process. A different pid or a different start time means the counters restarted
// from zero, so a difference between the two is not a change anyone observed.
func sameDaemonRun(previous *pb.GetStatusResponse, current *pb.GetStatusResponse) bool {
	first := previous.GetDaemon()
	second := current.GetDaemon()
	if first == nil || second == nil {
		return false
	}
	if first.GetPid() != second.GetPid() {
		return false
	}
	return first.GetStartedAt().AsTime().Equal(second.GetStartedAt().AsTime())
}

// View composes the frame. The header and the key line are pinned; everything
// between them scrolls as one body, because the counter block alone is taller
// than a standard terminal and bubbletea keeps only the last height lines of a
// frame. Pinning the counters instead would push the header and the first
// groups off the top with no key that could bring them back.
func (m statusModel) View() string {
	if m.quitting {
		return ""
	}
	width := m.width
	if width <= 0 {
		width = defaultTermWidth
	}

	header := m.headerBlock()
	footer := faintStyle.Render(m.keyLine())

	body := append(
		strings.Split(statusCounterBlock(m.response, m.previous, width), "\n"),
		"",
	)
	body = append(body, strings.Split(m.activityBlock(width), "\n")...)

	visible := m.visibleBodyRows(len(strings.Split(header, "\n")))
	offset := min(m.offset, max(len(body)-visible, 0))
	end := min(offset+visible, len(body))

	var builder strings.Builder
	builder.WriteString(header)
	builder.WriteString("\n\n")
	builder.WriteString(strings.Join(body[offset:end], "\n"))
	builder.WriteString("\n")
	if hidden := len(body) - end; hidden > 0 {
		builder.WriteString(faintStyle.Render(fmt.Sprintf("  %d more lines below", hidden)))
		builder.WriteString("\n")
	}
	builder.WriteString(footer)
	builder.WriteString("\n")
	return builder.String()
}

// visibleBodyRows is how many body lines fit between the pinned header and the
// pinned key line. An unknown height renders everything, which is what a piped
// or freshly started screen wants.
func (m statusModel) visibleBodyRows(headerLines int) int {
	if m.height <= 0 {
		return int(^uint(0) >> 1)
	}
	// The blank line under the header, the more-lines notice, and the key line.
	const chrome = 3
	rows := m.height - headerLines - chrome
	if rows < 1 {
		return 1
	}
	return rows
}

// headerBlock names the process and states when the screen last read it. A
// failed refresh keeps the last successful timestamp and appends the reason, so
// a dead connection never reads as a quiet system.
func (m statusModel) headerBlock() string {
	daemon := m.response.GetDaemon()
	first := fmt.Sprintf("lm-semantic-search  version=%s  pid=%d",
		daemon.GetVersion(), daemon.GetPid())
	second := "socket=" + daemon.GetSocketPath()

	stamp := m.readAt.Format("15:04:05")
	third := fmt.Sprintf("read_at=%s  interval=%s", stamp, m.interval)
	if m.paused {
		third = fmt.Sprintf("paused_at=%s  interval=%s", stamp, m.interval)
	}
	if m.refreshErr != nil {
		third += fmt.Sprintf(`  refresh_error=%q`, m.refreshErr.Error())
	}
	return headerStyle.Render(first) + "\n" + faintStyle.Render(second) + "\n" + faintStyle.Render(third)
}

func (m statusModel) keyLine() string {
	keys := "up/down scroll   p pause   r refresh   q quit"
	if m.paused {
		keys = "up/down scroll   p resume   r refresh   q quit"
	}
	if m.refreshing {
		keys += "   reading"
	}
	return keys
}

// statusCounterBlock lays out the counters in four columns: the name, the
// digit-grouped value, the unit, and the change since the previous read. Column
// widths come from one pass over the metrics, so a value growing a digit never
// shifts the columns beside it.
func statusCounterBlock(response *pb.GetStatusResponse, previous map[string]int64, width int) string {
	metrics := response.GetMetrics()
	nameWidth := 0
	valueWidth := 0
	deltaWidth := 0
	for _, metric := range metrics {
		nameWidth = max(nameWidth, len(metric.GetName()))
		valueWidth = max(valueWidth, len(statusValueText(metric)))
		deltaWidth = max(deltaWidth, len(statusDeltaText(metric, previous)))
	}

	lines := make([]string, 0, len(metrics)+8)
	group := ""
	for _, metric := range metrics {
		if metric.GetGroup() != group && group != "" {
			lines = append(lines, "")
		}
		group = metric.GetGroup()
		lines = append(lines, statusCounterLine(metric, previous, nameWidth, valueWidth, deltaWidth, width))
	}
	return strings.Join(lines, "\n")
}

func statusCounterLine(
	metric *pb.Metric,
	previous map[string]int64,
	nameWidth int,
	valueWidth int,
	deltaWidth int,
	width int,
) string {
	name := padTo(metric.GetName(), nameWidth)
	value := padLeftTo(statusValueText(metric), valueWidth)
	unit := padTo(metric.GetUnit(), statusUnitWide)
	delta := padLeftTo(statusDeltaText(metric, previous), deltaWidth)

	line := name + strings.Repeat(" ", statusValueGap) + value +
		strings.Repeat(" ", statusUnitGap) + unit + delta
	line = strings.TrimRight(line, " ")
	if len(line) > width {
		line = fitTail(line, width)
	}
	if strings.TrimSpace(delta) == "" || strings.TrimSpace(delta) == "+0" {
		return faintStyle.Render(line)
	}
	return line
}

// statusValueText renders a value with digits grouped, which the piped and JSON
// forms deliberately do not do because both are parsed.
func statusValueText(metric *pb.Metric) string {
	text := render.MetricValueText(metric)
	if _, isInt := metric.GetValue().(*pb.Metric_IntValue); isInt {
		return groupDigits(text)
	}
	return text
}

// statusDeltaText reports the change since the previous read for an integer
// value. A value with no previous read, and any non-integer value, reports
// nothing rather than a change of zero it did not observe.
func statusDeltaText(metric *pb.Metric, previous map[string]int64) string {
	intValue, isInt := metric.GetValue().(*pb.Metric_IntValue)
	if !isInt || previous == nil {
		return ""
	}
	prior, seen := previous[metric.GetName()]
	if !seen {
		return ""
	}
	return deltaText(intValue.IntValue-prior, true)
}

// activityBlock renders every unit of work as an indented block of name=value
// pairs, using the same names the counters use. It renders all of them; View
// owns the scrolling, so the counters and the activity share one window rather
// than competing for the same rows.
func (m statusModel) activityBlock(width int) string {
	rows := m.response.GetActivity()
	header := headerStyle.Render(fmt.Sprintf("activity  rows=%d", len(rows)))
	if len(rows) == 0 {
		return header + "\n" + faintStyle.Render("  none running, none queued")
	}

	lines := []string{header}
	for index, row := range rows {
		lines = append(lines, statusActivityRow(index, row, width)...)
	}
	return strings.Join(lines, "\n")
}

// statusActivityRow renders one unit of work. Fields carry their unit in their
// name, so they take no unit column.
//
// A row carrying no fields still renders a line. Every row this daemon builds
// carries fields, but the reply comes off a socket, so a daemon of another
// version or a truncated message must not crash the screen.
func statusActivityRow(index int, row *pb.ActivityRow, width int) []string {
	pairs := make([]string, 0, len(row.GetMetrics()))
	for _, metric := range row.GetMetrics() {
		pairs = append(pairs, metric.GetName()+"="+statusValueText(metric))
	}
	if len(pairs) == 0 {
		return []string{fmt.Sprintf("  [%d] (no fields reported)", index)}
	}

	lines := []string{fmt.Sprintf("  [%d] %s", index, pairs[0])}
	current := "     "
	for _, pair := range pairs[1:] {
		if len(current)+len(pair)+2 > width {
			lines = append(lines, faintStyle.Render(strings.TrimRight(current, " ")))
			current = "     "
		}
		current += pair + "  "
	}
	if strings.TrimSpace(current) != "" {
		lines = append(lines, faintStyle.Render(strings.TrimRight(current, " ")))
	}
	return lines
}

// integerValuesByName indexes a reply's integer values so the next render can
// subtract them. Only integers carry a delta; a rate, a timestamp, and a string
// have no meaningful difference between two reads.
func integerValuesByName(response *pb.GetStatusResponse) map[string]int64 {
	values := make(map[string]int64, len(response.GetMetrics()))
	for _, metric := range response.GetMetrics() {
		if intValue, isInt := metric.GetValue().(*pb.Metric_IntValue); isInt {
			values[metric.GetName()] = intValue.IntValue
		}
	}
	return values
}

// groupDigits inserts a comma every three digits from the right, preserving a
// leading sign. The terminal groups digits so a value crossing a digit boundary
// is visible without reading it; the piped and JSON forms keep raw digits
// because both are parsed.
func groupDigits(digits string) string {
	sign := ""
	if strings.HasPrefix(digits, "-") {
		sign = "-"
		digits = digits[1:]
	}
	if len(digits) <= 3 {
		return sign + digits
	}
	parts := make([]string, 0, (len(digits)+2)/3)
	for len(digits) > 3 {
		parts = append([]string{digits[len(digits)-3:]}, parts...)
		digits = digits[:len(digits)-3]
	}
	parts = append([]string{digits}, parts...)
	return sign + strings.Join(parts, ",")
}

// deltaText renders the change since the previous read. It always carries a
// sign, so direction never has to be inferred, and it is empty when there is no
// previous read to compare against.
func deltaText(delta int64, hasPrevious bool) string {
	if !hasPrevious {
		return ""
	}
	if delta < 0 {
		return groupDigits(strconv.FormatInt(delta, 10))
	}
	return "+" + groupDigits(strconv.FormatInt(delta, 10))
}

func statusTick(interval time.Duration) tea.Cmd {
	return tea.Tick(interval, func(time.Time) tea.Msg {
		return statusTickMsg{}
	})
}

func statusRefreshCmd(options cliOptions) tea.Cmd {
	return func() tea.Msg {
		response, err := fetchStatusResponse(options)
		return statusRefreshedMsg{response: response, err: err}
	}
}

// fetchStatusResponse reads one status reply, rejecting an unexpected reply type
// rather than rendering a zero value as if the daemon had reported it.
func fetchStatusResponse(options cliOptions) (*pb.GetStatusResponse, error) {
	result, err := callDaemon(options, func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
		return client.GetStatus(ctx, &pb.GetStatusRequest{})
	})
	if err != nil {
		return nil, err
	}
	response, ok := result.(*pb.GetStatusResponse)
	if !ok {
		return nil, errors.New("unexpected response type from GetStatus")
	}
	return response, nil
}
