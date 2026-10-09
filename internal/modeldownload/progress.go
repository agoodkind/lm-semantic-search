package modeldownload

import (
	"time"

	"goodkind.io/lm-semantic-search/internal/clock"
)

const progressInterval = 250 * time.Millisecond

type progressReporter struct {
	progress        ProgressFunc
	artifact        string
	downloadedBytes int64
	totalBytes      int64
	lastReport      time.Time
}

func newProgressReporter(progress ProgressFunc, artifact string) *progressReporter {
	return &progressReporter{
		progress:        progress,
		artifact:        artifact,
		downloadedBytes: 0,
		totalBytes:      0,
		lastReport:      time.Time{},
	}
}

func (reporter *progressReporter) begin(downloadedBytes int64, totalBytes int64) {
	reporter.downloadedBytes = downloadedBytes
	reporter.totalBytes = totalBytes
}

func (reporter *progressReporter) Write(chunk []byte) (int, error) {
	reporter.downloadedBytes += int64(len(chunk))
	if reporter.progress == nil {
		return len(chunk), nil
	}
	now := clock.Now()
	if now.Sub(reporter.lastReport) < progressInterval {
		return len(chunk), nil
	}
	reporter.lastReport = now
	reporter.progress(reporter.artifact, reporter.downloadedBytes, reporter.totalBytes)
	return len(chunk), nil
}

func (reporter *progressReporter) finish() {
	if reporter.progress == nil {
		return
	}
	reporter.progress(reporter.artifact, reporter.downloadedBytes, reporter.totalBytes)
}
