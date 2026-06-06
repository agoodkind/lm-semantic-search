package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/charmbracelet/bubbles/table"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

// runCodebaseListTUI fetches the tracked codebases once and drives the
// interactive list. It runs only when stdout is a real terminal in human mode;
// the piped, JSON, and single-line paths stay on callAndPrint. An empty registry
// prints the daemon's plain text instead of launching an empty table.
func runCodebaseListTUI(options cliOptions) error {
	result, err := callDaemon(options, func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
		return client.ListIndexes(ctx, &pb.ListIndexesRequest{})
	})
	if err != nil {
		return err
	}
	listResponse, ok := result.(*pb.ListIndexesResponse)
	if !ok {
		return errors.New("unexpected response type from ListIndexes")
	}
	codebases := listResponse.GetIndexes()
	if len(codebases) == 0 {
		_, writeErr := fmt.Fprintln(os.Stdout, strings.TrimSpace(listResponse.GetDisplayText()))
		if writeErr != nil {
			slog.Error("write empty codebase list failed", "err", writeErr)
			return fmt.Errorf("write empty list output: %w", writeErr)
		}
		return nil
	}

	program := tea.NewProgram(newListModel(options, codebases), tea.WithAltScreen())
	if _, runErr := program.Run(); runErr != nil {
		slog.Error("run codebase list TUI failed", "err", runErr)
		return fmt.Errorf("run codebase list: %w", runErr)
	}
	return nil
}

// listModel is the bubbletea state for the interactive codebase list. It lays
// out daemon-provided records and, on enter, asks the daemon (GetIndex) for the
// current-state detail. It makes no status decision of its own.
type listModel struct {
	options       cliOptions
	table         table.Model
	codebases     []*pb.Codebase
	showingDetail bool
	detail        string
	loading       bool
	err           error
	quitting      bool
}

// detailLoadedMsg carries the result of the GetIndex call fired on enter.
type detailLoadedMsg struct {
	text string
	err  error
}

func newListModel(options cliOptions, codebases []*pb.Codebase) listModel {
	columns, rows := buildTableColumns(codebases)
	codebaseTable := table.New(
		table.WithColumns(columns),
		table.WithRows(rows),
		table.WithFocused(true),
		table.WithHeight(tableHeight(len(rows))),
	)
	styles := table.DefaultStyles()
	styles.Header = styles.Header.
		BorderStyle(lipgloss.NormalBorder()).
		BorderForeground(lipgloss.Color("240")).
		BorderBottom(true).
		Bold(true)
	styles.Selected = styles.Selected.
		Foreground(lipgloss.Color("231")).
		Background(lipgloss.Color("57")).
		Bold(true)
	codebaseTable.SetStyles(styles)

	return listModel{
		options:       options,
		table:         codebaseTable,
		codebases:     codebases,
		showingDetail: false,
		detail:        "",
		loading:       false,
		err:           nil,
		quitting:      false,
	}
}

func (m listModel) Init() tea.Cmd {
	return nil
}

func (m listModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch typed := msg.(type) {
	case tea.KeyMsg:
		return m.handleKey(typed)
	case detailLoadedMsg:
		m.loading = false
		if typed.err != nil {
			m.err = typed.err
			return m, nil
		}
		m.err = nil
		m.detail = typed.text
		m.showingDetail = true
		return m, nil
	}
	if m.showingDetail {
		return m, nil
	}
	var cmd tea.Cmd
	m.table, cmd = m.table.Update(msg)
	return m, cmd
}

func (m listModel) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if keyMatches(msg, "ctrl+c", "q") {
		m.quitting = true
		return m, tea.Quit
	}
	if m.showingDetail {
		return m.handleDetailKey(msg)
	}
	return m.handleListKey(msg)
}

func (m listModel) handleDetailKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if keyMatches(msg, "esc", "backspace", "left", "h") {
		m.showingDetail = false
		m.detail = ""
		m.err = nil
	}
	return m, nil
}

func (m listModel) handleListKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if keyMatches(msg, "esc") {
		m.quitting = true
		return m, tea.Quit
	}
	if keyMatches(msg, "enter", "right", "l") {
		if len(m.codebases) == 0 {
			return m, nil
		}
		id := m.codebases[m.table.Cursor()].GetId()
		m.loading = true
		m.err = nil
		return m, loadDetailCmd(m.options, id)
	}
	var cmd tea.Cmd
	m.table, cmd = m.table.Update(msg)
	return m, cmd
}

// keyMatches reports whether the pressed key equals any of the given names,
// keeping key handling as plain comparisons rather than a switch on a bare
// string.
func keyMatches(msg tea.KeyMsg, keys ...string) bool {
	return slices.Contains(keys, msg.String())
}

func (m listModel) View() string {
	if m.quitting {
		return ""
	}
	if m.showingDetail {
		return m.detailView()
	}
	return m.listView()
}

func (m listModel) listView() string {
	footer := faintStyle.Render("up/down move · enter open · q quit")
	if m.loading {
		footer = faintStyle.Render("loading")
	}
	if m.err != nil {
		footer = errorStyle.Render(m.err.Error())
	}
	return m.table.View() + "\n" + footer + "\n"
}

func (m listModel) detailView() string {
	body := m.detail
	if m.err != nil {
		body = errorStyle.Render(m.err.Error())
	}
	footer := faintStyle.Render("esc back · q quit")
	return strings.TrimRight(body, "\n") + "\n\n" + footer + "\n"
}

func loadDetailCmd(options cliOptions, id string) tea.Cmd {
	return func() tea.Msg {
		result, err := callDaemon(options, func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (protoMessage, error) {
			return client.GetIndex(ctx, &pb.GetIndexRequest{Path: id})
		})
		if err != nil {
			return detailLoadedMsg{text: "", err: err}
		}
		indexResponse, ok := result.(*pb.GetIndexResponse)
		if !ok {
			return detailLoadedMsg{text: "", err: errors.New("unexpected response type from GetIndex")}
		}
		return detailLoadedMsg{text: indexResponse.GetDisplayText(), err: nil}
	}
}

var (
	faintStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	errorStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("9"))
)

// buildTableColumns builds the column set and rows for the codebase table. The
// ID column is sized to the longest id so the full id stays visible and
// selectable, since callers copy it into other commands.
func buildTableColumns(codebases []*pb.Codebase) ([]table.Column, []table.Row) {
	idWidth := len("ID")
	pathWidth := len("PATH")
	for _, codebase := range codebases {
		if width := len(codebase.GetId()); width > idWidth {
			idWidth = width
		}
		if width := len(codebase.GetCanonicalPath()); width > pathWidth {
			pathWidth = width
		}
	}
	const maxPathWidth = 70
	pathWidth = min(pathWidth, maxPathWidth)
	columns := []table.Column{
		{Title: "ID", Width: idWidth},
		{Title: "STATUS", Width: 12},
		{Title: "FILES", Width: 6},
		{Title: "PATH", Width: pathWidth},
	}
	rows := make([]table.Row, 0, len(codebases))
	for _, codebase := range codebases {
		rows = append(rows, table.Row{
			codebase.GetId(),
			statusCell(codebase.GetStatus()),
			fileCountCell(codebase),
			codebase.GetCanonicalPath(),
		})
	}
	return columns, rows
}

// statusCell renders a color-coded dot and the status word so the five states
// are visually distinct. The dot carries the color; the word stays plain to keep
// the table column width predictable.
func statusCell(status string) string {
	dot := lipgloss.NewStyle().Foreground(statusColor(status)).Render("●")
	return dot + " " + status
}

// statusColors maps each codebase status to a distinct foreground color. An
// unknown status falls back to grayStatus.
var statusColors = map[string]lipgloss.Color{
	"indexed":     lipgloss.Color("10"),
	"indexing":    lipgloss.Color("11"),
	"stale":       lipgloss.Color("208"),
	"failed":      lipgloss.Color("9"),
	"not_indexed": grayStatus,
}

const grayStatus = lipgloss.Color("245")

// statusColor returns the color for a codebase status, defaulting to gray.
func statusColor(status string) lipgloss.Color {
	if color, ok := statusColors[status]; ok {
		return color
	}
	return grayStatus
}

// fileCountCell returns the indexed-file count from the last successful run, or
// a hyphen placeholder when the codebase has never completed a run.
func fileCountCell(codebase *pb.Codebase) string {
	run := codebase.GetLastSuccessfulRun()
	if run == nil {
		return "-"
	}
	return strconv.Itoa(int(run.GetIndexedFiles()))
}

// tableHeight caps the visible rows so a long registry scrolls inside the table
// rather than overflowing the screen, while a short list stays fully visible.
func tableHeight(rowCount int) int {
	const maxVisibleRows = 20
	if rowCount < maxVisibleRows {
		return rowCount + 1
	}
	return maxVisibleRows
}
