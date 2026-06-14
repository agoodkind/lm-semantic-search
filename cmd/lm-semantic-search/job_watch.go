package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	daemonclient "goodkind.io/lm-semantic-search/client"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

// watchPollInterval is how often watchJob asks the daemon for job state. It
// matches the MCP adapter's wait poll cadence so both surfaces load the
// daemon identically.
const watchPollInterval = 1500 * time.Millisecond

// watchJob follows one daemon job and renders progress lines until the job
// reaches a terminal state, the timeout expires, or the user interrupts. It
// polls GetJob on a fixed interval; the daemon's WatchJobs RPC reports a
// one-shot snapshot rather than a subscription, so polling is the reliable
// follow primitive, exactly as the MCP adapter's wait path does. Timeout and
// interrupt both detach without cancelling the job: the daemon owns the job,
// so the command reports it as sent to the background.
func watchJob(options cliOptions, jobID string, timeout time.Duration) error {
	if jobID == "" {
		// A deduplicated or already-indexed registration has no job to watch.
		return nil
	}
	signalCtx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	ctx, cancel := context.WithTimeout(signalCtx, timeout)
	defer cancel()

	connection, client, err := daemonclient.DialDaemon(ctx, options.socketPath)
	if err != nil {
		slog.Error("dial daemon for job watch failed", "socket_path", options.socketPath, "err", err)
		return fmt.Errorf("dial daemon: %w", err)
	}
	defer connection.Close()

	ticker := time.NewTicker(watchPollInterval)
	defer ticker.Stop()

	for {
		current, getErr := client.GetJob(ctx, &pb.GetJobRequest{JobId: jobID})
		if getErr != nil {
			if ctx.Err() != nil {
				printSentToBackground(jobID)
				return nil
			}
			return formatCallError(getErr)
		}
		if done, exitErr := renderJobUpdate(current.GetJob()); done {
			return exitErr
		}

		select {
		case <-ctx.Done():
			printSentToBackground(jobID)
			return nil
		case <-ticker.C:
		}
	}
}

// printSentToBackground is the single detach line for both the timeout and
// the interrupt: the job keeps running in the daemon either way.
func printSentToBackground(jobID string) {
	fmt.Fprintf(os.Stderr, "\nsent to background: job %s keeps running in the daemon; check it with `lm-semantic-search job get %s`\n", jobID, jobID)
}

// jobOutcome is the daemon-resolved terminal result carried on the wire's
// outcome field. The CLI never derives terminality from the raw state field
// (the display guard enforces this); these constants mirror the tokens
// internal/pbconv stamps.
type jobOutcome string

const (
	outcomeLive      jobOutcome = ""
	outcomeSucceeded jobOutcome = "succeeded"
	outcomeFailed    jobOutcome = "failed"
	outcomeCanceled  jobOutcome = "canceled"
)

// renderJobUpdate prints one progress line and reports whether the job
// reached a terminal state, with the error the command should exit with.
func renderJobUpdate(job *pb.Job) (bool, error) {
	if job == nil {
		return false, nil
	}
	progress := job.GetProgress()
	unit := progress.GetUnit()
	if unit == "" {
		unit = "file"
	}
	fmt.Fprintf(os.Stderr, "\r\033[K%s %.1f%% (%d/%d %ss)",
		progress.GetPhase(), progress.GetOverallPercent(),
		progress.GetFilesProcessed(), progress.GetFilesTotal(), unit)
	if reused := progress.GetChunksReused(); reused > 0 {
		fmt.Fprintf(os.Stderr, " · %d reused", reused)
	}

	switch jobOutcome(job.GetOutcome()) {
	case outcomeSucceeded:
		fmt.Fprintf(os.Stderr, "\njob %s completed\n", job.GetId())
		return true, nil
	case outcomeFailed:
		fmt.Fprintln(os.Stderr)
		message := job.GetDisplayError()
		if message == "" {
			message = "job failed; see `lm-semantic-search job get " + job.GetId() + "`"
		}
		return true, errors.New(message)
	case outcomeCanceled:
		fmt.Fprintln(os.Stderr)
		return true, errors.New("job " + job.GetId() + " was cancelled")
	case outcomeLive:
		return false, nil
	default:
		return false, nil
	}
}
