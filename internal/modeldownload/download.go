// Package modeldownload downloads files with pinned SHA-256 checksums.
package modeldownload

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	maximumAttempts      = 4
	retryBaseDelay       = time.Second
	retryDelayMultiplier = 2
	partialFileSuffix    = ".partial"
	partialFileMode      = 0o600
	artifactFileMode     = 0o644
	directoryMode        = 0o700
)

// ErrUnavailable indicates that Ensure could not complete the download.
// Ensure wraps ErrUnavailable for transport errors, HTTP error statuses,
// four failed attempts, context cancellation, or checksum mismatches.
var ErrUnavailable = errors.New("artifact unavailable")

// ProgressFunc reports disk bytes including earlier partial bytes.
// artifact is the destination file name. totalBytes is 0 for unknown lengths.
// The package calls ProgressFunc at most once per 250 milliseconds and once
// after the last byte.
type ProgressFunc func(artifact string, downloadedBytes int64, totalBytes int64)

// SleepFunc waits for the retry delay or returns when the context ends.
type SleepFunc func(ctx context.Context, delay time.Duration)

// Request requires SHA256 to contain the expected lowercase hexadecimal digest.
// A nil Progress disables progress reports. A nil Sleep selects a timer.
type Request struct {
	HTTPClient      *http.Client
	URL             string
	SHA256          string
	DestinationPath string
	Progress        ProgressFunc
	Sleep           SleepFunc
}

type attemptOutcome int

const (
	outcomeInstalled attemptOutcome = iota
	outcomeRetry
	outcomeRangeRejected
	outcomeFailed
)

// Ensure skips requests when the destination has the expected SHA-256 checksum.
// Ensure locks <DestinationPath>.lock before resuming <DestinationPath>.partial.
// Ensure makes at most four attempts.
// Ensure verifies SHA-256 before renaming the partial file to the destination.
func Ensure(ctx context.Context, request Request) error {
	installed, err := installedArtifactMatches(ctx, request)
	if err != nil || installed {
		return err
	}
	directory := filepath.Dir(request.DestinationPath)
	if err := os.MkdirAll(directory, directoryMode); err != nil {
		return fileError(
			ctx,
			"create cache directory",
			directory,
			err,
		)
	}
	lockFile, err := acquireLock(ctx, request)
	if err != nil {
		return err
	}
	defer func() {
		_ = lockFile.Close()
	}()
	installed, err = installedArtifactMatches(ctx, request)
	if err != nil || installed {
		return err
	}
	return download(ctx, request)
}

func installedArtifactMatches(ctx context.Context, request Request) (bool, error) {
	_, statErr := os.Stat(request.DestinationPath)
	if errors.Is(statErr, os.ErrNotExist) {
		return false, nil
	}
	if statErr != nil {
		return false, fileError(
			ctx,
			"inspect artifact",
			request.DestinationPath,
			statErr,
		)
	}
	hasher := sha256.New()
	if err := hashFile(ctx, request.DestinationPath, hasher); err != nil {
		return false, err
	}
	if hex.EncodeToString(hasher.Sum(nil)) == request.SHA256 {
		return true, nil
	}
	slog.WarnContext(
		ctx,
		"artifact checksum mismatch; downloading replacement",
		"path",
		request.DestinationPath,
	)
	return false, nil
}

func download(ctx context.Context, request Request) error {
	slog.InfoContext(
		ctx,
		"download artifact",
		"url",
		request.URL,
		"path",
		request.DestinationPath,
	)
	sleep := request.Sleep
	if sleep == nil {
		sleep = sleepContext
	}
	partialPath := request.DestinationPath + partialFileSuffix
	reporter := newProgressReporter(
		request.Progress,
		filepath.Base(request.DestinationPath),
	)
	delay := retryBaseDelay
	rangeRestarted := false
	attempt := 1
	for {
		outcome, err := runAttempt(ctx, request, partialPath, reporter)
		switch outcome {
		case outcomeInstalled:
			return nil
		case outcomeFailed:
			return err
		case outcomeRangeRejected:
			if rangeRestarted {
				return err
			}
			rangeRestarted = true
			continue
		case outcomeRetry:
		}
		if attempt >= maximumAttempts {
			return err
		}
		slog.WarnContext(
			ctx,
			"retry artifact download",
			"url",
			request.URL,
			"attempt",
			attempt,
			"maximum_attempts",
			maximumAttempts,
			"delay",
			delay,
			"err",
			err,
		)
		sleep(ctx, delay)
		if cancelErr := ctx.Err(); cancelErr != nil {
			return unavailable(ctx, request.URL, cancelErr)
		}
		attempt++
		delay *= retryDelayMultiplier
	}
}

func sleepContext(ctx context.Context, delay time.Duration) {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
	case <-timer.C:
	}
}

func unavailable(ctx context.Context, rawURL string, cause error) error {
	slog.ErrorContext(
		ctx,
		"download artifact failed",
		"url",
		rawURL,
		"err",
		cause,
	)
	return fmt.Errorf(
		"download artifact %s: %w: %w",
		rawURL,
		ErrUnavailable,
		cause,
	)
}

func fileError(
	ctx context.Context,
	operation string,
	path string,
	cause error,
) error {
	slog.ErrorContext(ctx, operation+" failed", "path", path, "err", cause)
	return fmt.Errorf("%s %s: %w", operation, path, cause)
}

func hashFile(ctx context.Context, path string, destination io.Writer) error {
	file, err := os.Open(path)
	if err != nil {
		return fileError(ctx, "open artifact", path, err)
	}
	_, copyErr := io.Copy(destination, file)
	closeErr := file.Close()
	if copyErr != nil {
		return fileError(ctx, "hash artifact", path, copyErr)
	}
	if closeErr != nil {
		return fileError(ctx, "close artifact", path, closeErr)
	}
	return nil
}
