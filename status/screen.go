package status

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// statusValueGap and statusUnitGap separate the four columns. They are constants
// to keep the layout identical between two renders of the same terminal width.
const (
	statusValueGap = 2
	statusUnitGap  = 2
	statusUnitWide = 10
	// defaultTermWidth is the width used before the terminal reports its size.
	defaultTermWidth = 120
)

var (
	faintStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	headerStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("250")).Bold(true)
)

// statusModel is the bubbletea state for the live screen. It keeps the previous
// read's integer values so each refresh can report the change, and it keeps the
// last successful read time. A failed refresh therefore never reads as a quiet
// system.
type statusModel struct {
	source   Source
	interval time.Duration
	now      func() time.Time
	snapshot Snapshot
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
	snapshot Snapshot
	err      error
}

type statusTickMsg struct{}

func newStatusModel(source Source, interval time.Duration, now func() time.Time, first Snapshot) statusModel {
	return statusModel{
		source:     source,
		interval:   interval,
		now:        now,
		snapshot:   first,
		previous:   nil,
		comparable: true,
		readAt:     now(),
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
			cmds = append(cmds, statusRefreshCmd(m.source))
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
		return m, statusRefreshCmd(m.source)
	case keyMatches(msg, "down", "j"):
		// Clamped on the way down, not only when rendering. View clamps a local
		// copy, so without this m.offset would keep climbing past the last line
		// and the same number of Up presses would appear to do nothing before the
		// screen finally moved.
		m.offset = min(m.offset+1, m.maxOffset())
		return m, nil
	case keyMatches(msg, "up", "k"):
		m.offset = max(m.offset-1, 0)
		return m, nil
	default:
		return m, nil
	}
}

// applyRefresh swaps in a fresh snapshot and keeps the prior integer values for
// the next render to report the change. A failed refresh keeps the previous
// snapshot and the previous read time, and the screen states it is stale rather
// than showing an empty one.
//
// The baseline is dropped whenever the two reads did not observe one continuous
// run of one process. A restarted process zeroes every counter, so subtracting
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
	if sameRun(m.snapshot, msg.snapshot) && m.comparable {
		m.previous = integerValuesByName(m.snapshot)
	} else {
		m.previous = nil
	}
	m.comparable = true
	m.snapshot = msg.snapshot
	m.readAt = m.now()
	return m
}

// sameRun reports whether two snapshots came from one continuous run of one
// process. A different or missing RunID means the counters may have restarted
// from zero, and a difference between the two is not a change anyone observed.
func sameRun(previous Snapshot, current Snapshot) bool {
	if previous.RunID == "" || current.RunID == "" {
		return false
	}
	return previous.RunID == current.RunID
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

	body := m.bodyLines(width)

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

// bodyLines is everything that scrolls: the counter block, a blank separator,
// then the activity block. View and the scroll keys both read it. The last
// scrollable line has the same number in both.
func (m statusModel) bodyLines(width int) []string {
	lines := append(
		strings.Split(statusCounterBlock(m.snapshot, m.previous, width), "\n"),
		"",
	)
	return append(lines, strings.Split(m.activityBlock(width), "\n")...)
}

// maxOffset is the furthest the body can scroll before its last line sits at
// the bottom of the window. Scrolling past it would move nothing.
func (m statusModel) maxOffset() int {
	width := m.width
	if width <= 0 {
		width = defaultTermWidth
	}
	headerLines := len(strings.Split(m.headerBlock(), "\n"))
	return max(len(m.bodyLines(width))-m.visibleBodyRows(headerLines), 0)
}

// visibleBodyRows is how many body lines fit between the pinned header and the
// pinned key line. An unknown height renders everything, the right choice for a
// freshly started screen.
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

// headerBlock shows the snapshot's title and details and states when the screen
// last read the source. A failed refresh keeps the last successful timestamp and
// appends the reason. A dead connection therefore never reads as a quiet system.
func (m statusModel) headerBlock() string {
	stamp := inLocalZone(m.readAt).Format("15:04:05")
	last := fmt.Sprintf("read_at=%s  interval=%s", stamp, m.interval)
	if m.paused {
		last = fmt.Sprintf("paused_at=%s  interval=%s", stamp, m.interval)
	}
	if m.refreshErr != nil {
		last += fmt.Sprintf(`  refresh_error=%q`, m.refreshErr.Error())
	}

	lines := []string{headerStyle.Render(m.snapshot.Title)}
	for _, detail := range m.snapshot.Details {
		lines = append(lines, faintStyle.Render(detail))
	}
	lines = append(lines, faintStyle.Render(last))
	return strings.Join(lines, "\n")
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
// widths come from one pass over the counters, and a value growing a digit does
// not shift the columns beside it.
func statusCounterBlock(snapshot Snapshot, previous map[string]int64, width int) string {
	counters := snapshot.Counters
	nameWidth := 0
	valueWidth := 0
	deltaWidth := 0
	for _, counter := range counters {
		nameWidth = max(nameWidth, len(counter.Name))
		valueWidth = max(valueWidth, len(statusValueText(counter.Value)))
		deltaWidth = max(deltaWidth, len(statusDeltaText(counter, previous)))
	}

	lines := make([]string, 0, len(counters)+8)
	group := ""
	for _, counter := range counters {
		if counter.Group != group && group != "" {
			lines = append(lines, "")
		}
		group = counter.Group
		lines = append(lines, statusCounterLine(counter, previous, nameWidth, valueWidth, deltaWidth, width))
	}
	return strings.Join(lines, "\n")
}

func statusCounterLine(
	counter Field,
	previous map[string]int64,
	nameWidth int,
	valueWidth int,
	deltaWidth int,
	width int,
) string {
	name := padTo(counter.Name, nameWidth)
	value := padLeftTo(statusValueText(counter.Value), valueWidth)
	unit := padTo(counter.Unit, statusUnitWide)
	delta := padLeftTo(statusDeltaText(counter, previous), deltaWidth)

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

// statusValueText renders a value for the screen: digits grouped, and a
// timestamp shown in the operator's own zone.
//
// The dump keeps UTC, because it is parsed and a machine consumer wants one
// unambiguous zone. The screen is the only surface a person reads directly, so
// it is the only one that converts.
func statusValueText(value Value) string {
	text := valueText(value)
	if value.kind == kindInteger {
		return groupDigits(text)
	}
	return displayTimeText(text)
}

// displayTimeText converts an RFC3339 timestamp into the host's zone with its
// offset, and returns anything else unchanged. A counter, a phase name, and a
// path never parse as RFC3339, and the function returns each of them as given.
func displayTimeText(text string) string {
	parsed, err := time.Parse(time.RFC3339Nano, text)
	if err != nil {
		return text
	}
	return inLocalZone(parsed).Format("2006-01-02T15:04:05-07:00")
}

// inLocalZone returns value in the host's zone, or unchanged when that zone
// cannot be loaded.
//
// Loading the zone by name rather than reading the process-wide local zone is
// what keeps the gosmopolitan analyzer satisfied: the analyzer exists to catch
// an implicit machine locale, and a named lookup states the intent.
func inLocalZone(value time.Time) time.Time {
	location, err := time.LoadLocation("Local")
	if err != nil {
		return value
	}
	return value.In(location)
}

// statusDeltaText reports the change since the previous read for an integer
// value. A value with no previous read, a field marked NoDelta, and any
// non-integer value report an empty string rather than a change of zero.
func statusDeltaText(field Field, previous map[string]int64) string {
	if field.NoDelta {
		return ""
	}
	if field.Value.kind != kindInteger || previous == nil {
		return ""
	}
	prior, seen := previous[field.Name]
	if !seen {
		return ""
	}
	return deltaText(field.Value.integer-prior, true)
}

// activityBlock renders every unit of work as an indented block of name=value
// pairs, using the same labels as the counters. It renders all of them; View
// owns the scrolling, and the counters and the activity share one window rather
// than competing for the same rows.
func (m statusModel) activityBlock(width int) string {
	rows := m.snapshot.Activity
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

// statusActivityRow renders one unit of work. Fields include their unit in
// their name, so they take no unit column.
//
// A row with no fields still renders a line. A source can return an empty row,
// for example after a version mismatch with the program it reads, and that must
// not crash the screen.
func statusActivityRow(index int, row []Field, width int) []string {
	pairs := make([]string, 0, len(row))
	for _, field := range row {
		pairs = append(pairs, field.Name+"="+statusValueText(field.Value))
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

// integerValuesByName indexes a snapshot's integer values for the next render
// to subtract. Only integers have a delta; a rate, a timestamp, and a string
// have no meaningful difference between two reads.
func integerValuesByName(snapshot Snapshot) map[string]int64 {
	values := make(map[string]int64, len(snapshot.Counters))
	for _, counter := range snapshot.Counters {
		if counter.Value.kind == kindInteger {
			values[counter.Name] = counter.Value.integer
		}
	}
	return values
}

// groupDigits inserts a comma every three digits from the right, preserving a
// leading sign. The terminal groups digits so a value crossing a digit boundary
// is visible without reading it; the dump keeps raw digits because it is
// parsed.
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

// deltaText renders the change since the previous read. It always has a sign,
// and direction never has to be inferred. It is empty when there is no previous
// read to compare against.
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

func statusRefreshCmd(source Source) tea.Cmd {
	return func() tea.Msg {
		snapshot, err := source()
		return statusRefreshedMsg{snapshot: snapshot, err: err}
	}
}

// keyMatches reports whether the pressed key equals any of the given names,
// keeping key handling as plain comparisons rather than a switch on a bare
// string.
func keyMatches(msg tea.KeyMsg, keys ...string) bool {
	return slices.Contains(keys, msg.String())
}

// padTo pads text with trailing spaces to a display width of width, measured in
// runes. A multibyte ellipsis counts as one column. It assumes text already
// fits; fit it first with fitTail.
func padTo(text string, width int) string {
	gap := width - utf8.RuneCountInString(text)
	if gap <= 0 {
		return text
	}
	return text + strings.Repeat(" ", gap)
}

// padLeftTo right-aligns text within width by prepending spaces, measured in
// runes.
func padLeftTo(text string, width int) string {
	gap := width - utf8.RuneCountInString(text)
	if gap <= 0 {
		return text
	}
	return strings.Repeat(" ", gap) + text
}

// fitTail keeps the head of text and drops the tail with a trailing ellipsis
// when it overflows width. Width and slicing are rune-based, and the ellipsis
// is counted once.
func fitTail(text string, width int) string {
	if width <= 0 {
		return ""
	}
	runes := []rune(text)
	if len(runes) <= width {
		return text
	}
	if width == 1 {
		return "…"
	}
	return string(runes[:width-1]) + "…"
}
