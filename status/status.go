// Package status shows a program's raw values as a live terminal screen or as
// a plain text dump.
//
// The package has no knowledge of what the values mean. The calling program
// supplies a Source that returns a fresh Snapshot of named values, and Run
// shows them: a screen that refreshes in place when stdout is a terminal, one
// dump otherwise.
package status

import (
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"golang.org/x/term"
)

const (
	// DefaultInterval is how often the live screen reads the Source when the
	// caller has no preference.
	DefaultInterval = 2 * time.Second
	// minimumInterval floors the cadence so several open screens cannot become
	// a busy loop against the program that owns the Source.
	minimumInterval = 500 * time.Millisecond
)

// Value is one raw value. The zero Value is absent and prints as null, which is
// how every surface says a fact is missing rather than zero or empty.
type Value struct {
	kind    valueKind
	integer int64
	number  float64
	flag    bool
	text    string
}

type valueKind int

const (
	kindAbsent valueKind = iota
	kindInteger
	kindNumber
	kindFlag
	kindText
)

// Int returns an integer Value. Only integers show a change since the previous
// read on the live screen.
func Int(value int64) Value {
	return Value{kind: kindInteger, integer: value, number: 0, flag: false, text: ""}
}

// Float returns a floating point Value.
func Float(value float64) Value {
	return Value{kind: kindNumber, integer: 0, number: value, flag: false, text: ""}
}

// Bool returns a boolean Value.
func Bool(value bool) Value {
	return Value{kind: kindFlag, integer: 0, number: 0, flag: value, text: ""}
}

// Text returns a string Value. The live screen shows a text that parses as an
// RFC 3339 timestamp in the host's time zone.
func Text(value string) Value {
	return Value{kind: kindText, integer: 0, number: 0, flag: false, text: value}
}

// Field is one named value.
type Field struct {
	// Group sets which fields belong together. The live screen leaves a blank
	// line where Group changes between two counters.
	Group string
	// Name is printed exactly as given.
	Name string
	// Unit is printed after the value and may be empty.
	Unit string
	// Value is the raw value.
	Value Value
	// NoDelta turns off the change column for a value that never changes.
	NoDelta bool
}

// Snapshot is everything one read of the program reports.
type Snapshot struct {
	// Title is the first line of the live screen header.
	Title string
	// Details are further header lines on the live screen.
	Details []string
	// Notices are lines the dump prints before its records.
	Notices []string
	// Identity identifies the process the values came from. The dump prints each
	// non-empty text field before the counters.
	Identity []Field
	// RunID identifies one continuous run of the observed process. The live
	// screen shows the change between two reads only when both reads carry the
	// same non-empty RunID, because a restarted process starts its counters over.
	RunID string
	// Counters are the named values, in display order.
	Counters []Field
	// Activity lists units of work, each as its own set of named values.
	Activity [][]Field
}

// Source returns a fresh Snapshot. Run reads it once for a dump and on every
// refresh of the live screen.
type Source func() (Snapshot, error)

// Options sets how Run shows the Source.
type Options struct {
	// Interval is the refresh cadence of the live screen. Run raises a value
	// below half a second to half a second.
	Interval time.Duration
	// Once prints one dump even when stdout is a terminal.
	Once bool
	// Now returns the current time. The live screen shows it as the time of
	// the last successful read.
	Now func() time.Time
}

// Run shows the Source as a live screen when stdout is a terminal and Once is
// false, and as one dump on stdout otherwise.
func Run(source Source, options Options) error {
	interval := max(options.Interval, minimumInterval)
	live := !options.Once && term.IsTerminal(int(os.Stdout.Fd()))
	if live {
		return runScreen(source, interval, options.Now)
	}
	snapshot, err := source()
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(os.Stdout, "%s\n", strings.TrimSpace(Dump(snapshot)))
	if err != nil {
		slog.Error("write status dump failed", "err", err)
		return fmt.Errorf("write status dump: %w", err)
	}
	return nil
}

// runScreen drives the live screen until the operator quits. It polls rather
// than subscribes, because a Source returns one snapshot per call.
func runScreen(source Source, interval time.Duration, now func() time.Time) error {
	first, err := source()
	if err != nil {
		return err
	}
	program := tea.NewProgram(newStatusModel(source, interval, now, first), tea.WithAltScreen())
	if _, runErr := program.Run(); runErr != nil {
		slog.Error("run status TUI failed", "err", runErr)
		return fmt.Errorf("run status screen: %w", runErr)
	}
	return nil
}
