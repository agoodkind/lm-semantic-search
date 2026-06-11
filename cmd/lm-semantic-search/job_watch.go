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
	"goodkind.io/lm-semantic-search/internal/model"
)

// watchJob attaches to one daemon job and renders progress lines until the
// job reaches a terminal state, the timeout expires, or the user interrupts.
// Timeout and interrupt both detach without cancelling the job: the daemon
// owns the job, so the command reports it as sent to the background.
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

	stream, err := client.WatchJobs(ctx, &pb.WatchJobsRequest{JobIds: []string{jobID}})
	if err != nil {
		return formatCallError(err)
	}

	// The stream pushes transitions; fetch the current state once so a job
	// that finished before the attach does not wait out the whole timeout.
	if current, getErr := client.GetJob(ctx, &pb.GetJobRequest{JobId: jobID}); getErr == nil {
		if done, exitErr := renderJobUpdate(current.GetJob()); done {
			return exitErr
		}
	}

	for {
		update, recvErr := stream.Recv()
		if recvErr != nil {
			if ctx.Err() != nil {
				fmt.Fprintf(os.Stderr, "\nsent to background: job %s keeps running in the daemon; check it with `lm-semantic-search job get %s`\n", jobID, jobID)
				return nil
			}
			return formatCallError(recvErr)
		}
		if done, exitErr := renderJobUpdate(update.GetJob()); done {
			return exitErr
		}
	}
}

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

	switch model.JobState(job.GetState()) {
	case model.JobStateCompleted:
		fmt.Fprintf(os.Stderr, "\njob %s completed\n", job.GetId())
		return true, nil
	case model.JobStateFailed:
		fmt.Fprintln(os.Stderr)
		message := job.GetDisplayError()
		if message == "" {
			message = "job failed; see `lm-semantic-search job get " + job.GetId() + "`"
		}
		return true, errors.New(message)
	case model.JobStateCancelled:
		fmt.Fprintln(os.Stderr)
		return true, errors.New("job " + job.GetId() + " was cancelled")
	case model.JobStateQueued, model.JobStateRunning, model.JobStateCancelling:
		return false, nil
	default:
		return false, nil
	}
}
