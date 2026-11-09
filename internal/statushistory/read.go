package statushistory

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"goodkind.io/lm-semantic-search/internal/model"
)

const initialLogBufferSize = 64 * 1024

func readHistory(stateRoot string) (history, error) {
	var result history
	result.jobs = map[string]model.Job{}
	logsRoot := filepath.Join(stateRoot, "logs")
	entries, err := os.ReadDir(logsRoot)
	if err != nil && !os.IsNotExist(err) {
		slog.Error("read daemon logs failed", "path", logsRoot, "err", err)

		return history{}, fmt.Errorf("read daemon logs: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		if !strings.HasSuffix(entry.Name(), ".jsonl") &&
			!strings.HasSuffix(entry.Name(), ".jsonl.gz") {
			continue
		}

		if err := readLog(filepath.Join(logsRoot, entry.Name()), &result); err != nil {
			if strings.HasSuffix(entry.Name(), ".jsonl.gz") {
				result.warnings = append(result.warnings, "unreadable compressed daemon log")
				result.malformed = true

				continue
			}

			return history{}, err
		}
	}

	if err := readJobs(filepath.Join(stateRoot, "jobs.jsonl"), &result); err != nil {
		return history{}, err
	}

	return result, nil
}

func readLog(path string, result *history) error {
	file, err := os.Open(path)
	if err != nil {
		slog.Error("open daemon log failed", "path", path, "err", err)

		return fmt.Errorf("open daemon log %s: %w", path, err)
	}

	defer func() { _ = file.Close() }()
	var reader io.Reader = file
	if strings.HasSuffix(path, ".gz") {
		gzipReader, err := gzip.NewReader(file)
		if err != nil {
			slog.Error("open compressed daemon log failed", "path", path, "err", err)

			return fmt.Errorf("open compressed daemon log %s: %w", path, err)
		}

		defer func() { _ = gzipReader.Close() }()
		reader = gzipReader
	}

	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, initialLogBufferSize), maxStructuredRecordSize)

	for scanner.Scan() {
		var record map[string]json.RawMessage
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			result.warnings = append(result.warnings, "malformed structured log record")
			result.malformed = true

			continue
		}

		message := logMessage(stringValue(record["msg"]))
		switch message {
		case logMessagePerfCounters:
			readAt, timestampValid := timeValue(record["time"])
			if !timestampValid {
				result.warnings = append(
					result.warnings,
					"structured counter record without a timestamp",
				)
				result.malformed = true

				continue
			}

			result.snapshots = append(
				result.snapshots,
				snapshot{
					at:     readAt,
					values: numericFields(record),
					units:  recordUnits(record),
					live:   false,
				},
			)
		case logMessageSpanDone:
			readAt, timestampValid := timeValue(record["time"])
			if !timestampValid {
				result.warnings = append(
					result.warnings,
					"structured span record without a timestamp",
				)
				result.malformed = true

				continue
			}

			duration, durationValid := intValue(record["duration_ms"])
			if !durationValid {
				result.warnings = append(result.warnings, "span record without duration")
				result.malformed = true

				continue
			}

			result.spans = append(
				result.spans,
				span{
					at:       readAt,
					name:     stringValue(record["span"]),
					id:       stringValue(record["span_id"]),
					parent:   stringValue(record["parent_span_id"]),
					jobID:    stringValue(record["job_id"]),
					duration: duration,
				},
			)
		}
	}

	if err := scanner.Err(); err != nil {
		slog.Error("scan daemon log failed", "path", path, "err", err)
		result.warnings = append(result.warnings, "structured log scan stopped before end of file")
		result.malformed = true
	}

	return nil
}

func readJobs(path string, result *history) error {
	file, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil
	}

	if err != nil {
		slog.Error("open daemon jobs journal failed", "path", path, "err", err)

		return fmt.Errorf("open daemon jobs journal: %w", err)
	}

	defer func() { _ = file.Close() }()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, initialLogBufferSize), maxStructuredRecordSize)

	for scanner.Scan() {
		var event model.JobEvent
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			result.warnings = append(result.warnings, "malformed jobs journal record")
			result.malformed = true

			continue
		}

		if terminal(event.Job.State) {
			result.jobs[event.Job.ID] = event.Job
		}
	}

	if err := scanner.Err(); err != nil {
		slog.Error("scan daemon jobs journal failed", "path", path, "err", err)
		result.warnings = append(result.warnings, "job journal scan stopped before end of file")
		result.malformed = true
	}

	return nil
}

func stringValue(raw json.RawMessage) string {
	var value string
	_ = json.Unmarshal(raw, &value)

	return value
}

func timeValue(raw json.RawMessage) (time.Time, bool) {
	value := stringValue(raw)
	parsed, err := time.Parse(time.RFC3339Nano, value)

	return parsed, err == nil
}

func intValue(raw json.RawMessage) (int64, bool) {
	var value int64
	if err := json.Unmarshal(raw, &value); err == nil {
		return value, true
	}

	var decimal float64
	if err := json.Unmarshal(raw, &decimal); err != nil || decimal != math.Trunc(decimal) {
		return 0, false
	}

	return int64(decimal), true
}

func numericFields(record map[string]json.RawMessage) map[string]int64 {
	values := map[string]int64{}

	for name, raw := range record {
		if value, ok := intValue(raw); ok {
			values[name] = value
		}
	}

	return values
}

func recordUnits(record map[string]json.RawMessage) map[string]string {
	units := map[string]string{}
	raw, found := record["units"]
	if !found {
		return units
	}

	_ = json.Unmarshal(raw, &units)

	return units
}
