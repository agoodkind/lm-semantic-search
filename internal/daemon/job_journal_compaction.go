package daemon

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"sort"
	"time"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
)

// jobJournalRetainedTerminalPerCodebase is how many terminal jobs one codebase
// keeps through a compaction.
const jobJournalRetainedTerminalPerCodebase = 5

// jobJournalRetainedTerminalTotal caps the retained terminal jobs across every
// codebase.
const jobJournalRetainedTerminalTotal = 5000

// selectRetainedJobEvents keeps crash recovery records and a bounded,
// deterministic terminal history for each codebase.
func selectRetainedJobEvents(
	latest map[string]model.JobEvent,
) []model.JobEvent {
	nonTerminal := make([]model.JobEvent, 0)
	terminal := make([]model.JobEvent, 0)
	for _, event := range latest {
		state := event.Job.State
		isTerminal := state == model.JobStateCompleted ||
			state == model.JobStateFailed ||
			state == model.JobStateCancelled
		if isTerminal {
			terminal = append(terminal, event)
		} else {
			nonTerminal = append(nonTerminal, event)
		}
	}

	sortJobEventsByRecency(terminal)
	terminalPerCodebase := make(map[string]int)
	retainedTerminal := make([]model.JobEvent, 0, len(terminal))
	for _, event := range terminal {
		codebaseID := event.Job.CodebaseID
		if terminalPerCodebase[codebaseID] >= jobJournalRetainedTerminalPerCodebase {
			continue
		}
		terminalPerCodebase[codebaseID]++
		retainedTerminal = append(retainedTerminal, event)
	}
	if len(retainedTerminal) > jobJournalRetainedTerminalTotal {
		retainedTerminal = retainedTerminal[:jobJournalRetainedTerminalTotal]
	}

	retained := make(
		[]model.JobEvent,
		0,
		len(nonTerminal)+len(retainedTerminal),
	)
	retained = append(retained, nonTerminal...)
	retained = append(retained, retainedTerminal...)
	sortJobEventsByRecency(retained)
	return retained
}

// compactJobJournal rewrites the journal only after every retained event and
// resulting byte count have been prepared successfully.
func compactJobJournal(path string) (kept int, dropped int, err error) {
	latest, err := store.ReadJobEventsLatest(path)
	if err != nil {
		slog.Error(
			"read jobs journal for compaction failed",
			"component",
			"daemon",
			"subcomponent",
			"journal",
			"path",
			path,
			"err",
			err,
		)
		return 0, 0, fmt.Errorf(
			"read jobs journal %s for compaction: %w",
			path,
			err,
		)
	}
	beforeInfo, err := os.Stat(path)
	if err != nil {
		slog.Error(
			"stat jobs journal before compaction failed",
			"component",
			"daemon",
			"subcomponent",
			"journal",
			"path",
			path,
			"err",
			err,
		)
		return 0, 0, fmt.Errorf(
			"stat jobs journal %s before compaction: %w",
			path,
			err,
		)
	}
	retained := selectRetainedJobEvents(latest)
	afterBytes, err := marshalJobEventsSize(retained)
	if err != nil {
		slog.Error(
			"measure compacted jobs journal failed",
			"component",
			"daemon",
			"subcomponent",
			"journal",
			"path",
			path,
			"err",
			err,
		)
		return 0, 0, fmt.Errorf(
			"measure compacted jobs journal %s: %w",
			path,
			err,
		)
	}
	if err := store.WriteJobEvents(path, retained); err != nil {
		slog.Error(
			"write compacted jobs journal failed",
			"component",
			"daemon",
			"subcomponent",
			"journal",
			"path",
			path,
			"err",
			err,
		)
		return 0, 0, fmt.Errorf(
			"write compacted jobs journal %s: %w",
			path,
			err,
		)
	}

	kept = len(retained)
	dropped = len(latest) - kept
	slog.Info(
		"compacted job journal",
		"component",
		"daemon",
		"subcomponent",
		"journal",
		"path",
		path,
		"kept",
		kept,
		"dropped",
		dropped,
		"bytes_before",
		beforeInfo.Size(),
		"bytes_after",
		afterBytes,
	)
	return kept, dropped, nil
}

// sortJobEventsByRecency makes rewrites stable while preserving newest records
// first for both recovery and operator history.
func sortJobEventsByRecency(events []model.JobEvent) {
	sort.Slice(events, func(i int, j int) bool {
		leftTime := jobEventRecency(events[i])
		rightTime := jobEventRecency(events[j])
		if leftTime.Equal(rightTime) {
			return events[i].Job.ID < events[j].Job.ID
		}
		return leftTime.After(rightTime)
	})
}

// jobEventRecency uses completion time when available because a terminal job's
// last update may precede its durable completion record.
func jobEventRecency(event model.JobEvent) time.Time {
	if event.Job.CompletedAt != nil {
		return *event.Job.CompletedAt
	}
	return event.Job.UpdatedAt
}

// marshalJobEventsSize measures the exact JSONL bytes before replacement so a
// measurement failure cannot follow a successful rewrite.
func marshalJobEventsSize(events []model.JobEvent) (int64, error) {
	var size int64
	for _, event := range events {
		encoded, err := json.Marshal(event)
		if err != nil {
			slog.Error(
				"marshal job event for journal size failed",
				"component",
				"daemon",
				"subcomponent",
				"journal",
				"job_id",
				event.Job.ID,
				"err",
				err,
			)
			return 0, fmt.Errorf("marshal job event %s: %w", event.Job.ID, err)
		}
		size += int64(len(encoded) + 1)
	}
	return size, nil
}
