package daemon

import (
	"bytes"
	"log/slog"
	"strings"
	"sync"
	"testing"
)

// capturedLogs is a concurrency-safe sink for the process-global slog handler.
// A test manager keeps background workers alive, and any of them can log while
// the handler is installed, so an unsynchronized buffer would be a data race
// rather than a flake.
//
// Assertions read whole lines and match on every substring the caller supplies,
// which is how a test scopes itself to the codebase it set up rather than to
// whatever else logged during the window. A bare "no error line anywhere"
// assertion is not deterministic on a shared sink and must not be written.
type capturedLogs struct {
	mu     sync.Mutex
	buffer bytes.Buffer
}

func (logs *capturedLogs) Write(data []byte) (int, error) {
	logs.mu.Lock()
	defer logs.mu.Unlock()
	return logs.buffer.Write(data)
}

// text returns everything captured so far, for failure messages.
func (logs *capturedLogs) text() string {
	logs.mu.Lock()
	defer logs.mu.Unlock()
	return logs.buffer.String()
}

// linesContaining returns the captured lines that carry every substring in
// matches. An empty result means no single line carried all of them.
func (logs *capturedLogs) linesContaining(matches ...string) []string {
	found := make([]string, 0)
	for _, line := range strings.Split(logs.text(), "\n") {
		if lineContainsAll(line, matches) {
			found = append(found, line)
		}
	}
	return found
}

func lineContainsAll(line string, matches []string) bool {
	for _, match := range matches {
		if !strings.Contains(line, match) {
			return false
		}
	}
	return true
}

// captureLogs redirects the default logger into a race-safe buffer for the rest
// of the test, so assertions read the text an operator would see in the daemon
// log. The handler is process-global, so a test using this must not call
// t.Parallel.
func captureLogs(t *testing.T) *capturedLogs {
	t.Helper()

	logs := &capturedLogs{}
	previousLogger := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(previousLogger) })
	return logs
}
