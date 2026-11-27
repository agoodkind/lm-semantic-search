// Package scheduling admits background embedding batches when the host is idle.
package scheduling

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"sync/atomic"

	"goodkind.io/lm-semantic-search/internal/jobscheduler"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/platformactivity"
)

// Scheduler admits one quiet embedding batch at a time. It does not interrupt
// a batch when host activity resumes. Call Close when the owner shuts down.
type Scheduler struct {
	scheduler *jobscheduler.Scheduler
	done      <-chan struct{}
	onClose   func(func()) func() bool
	cancel    context.CancelFunc
	sequence  atomic.Uint64
}

// NewQuiet uses the LMS quiet policy with normal priority and its default idle
// threshold. Canceling ctx cancels pending acquisitions and stops sampling.
func NewQuiet(ctx context.Context) *Scheduler {
	return newQuiet(ctx, platformactivity.New(ctx))
}

func newQuiet(ctx context.Context, activity platformactivity.Source) *Scheduler {
	lifetime, cancel := context.WithCancel(ctx)
	scheduler := jobscheduler.New(lifetime, 1, activity)
	context.AfterFunc(lifetime, scheduler.Close)

	return &Scheduler{
		scheduler: scheduler,
		done:      lifetime.Done(),
		onClose: func(callback func()) func() bool {
			return context.AfterFunc(lifetime, callback)
		},
		cancel:   cancel,
		sequence: atomic.Uint64{},
	}
}

// Acquire waits until the quiet policy admits one batch. The returned function
// releases the slot and must be called after the batch finishes.
func (s *Scheduler) Acquire(ctx context.Context) (func(), error) {
	select {
	case <-s.done:
		return nil, context.Canceled
	default:
	}

	batchCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	stop := s.onClose(cancel)
	defer stop()
	sequence := s.sequence.Add(1)
	policy := model.DefaultSchedulingPolicy()
	policy.Quiet = true
	lease, err := s.scheduler.Acquire(
		batchCtx,
		jobscheduler.Entry{
			JobID:          strconv.FormatUint(sequence, 10),
			Policy:         policy,
			QueueSequence:  sequence,
			State:          jobscheduler.EntryWaiting,
			Reason:         model.SchedulingReasonUnspecified,
			PauseRequested: false,
		},
	)
	if err != nil {
		if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			slog.WarnContext(ctx, "quiet embedding batch admission failed", "err", err)
		}

		return nil, fmt.Errorf("admit quiet embedding batch: %w", err)
	}

	select {
	case <-s.done:
		lease.Release()

		return nil, context.Canceled
	default:
		return lease.Release, nil
	}
}

// WaitingReason returns the current quiet policy reason for a pending batch, or
// an empty string when the policy does not block any pending batch.
func (s *Scheduler) WaitingReason() string {
	snapshot := s.scheduler.Snapshot()
	if snapshot.Queued[model.JobPriorityNormal] == 0 {
		return ""
	}

	for _, reason := range snapshot.Reasons {
		if reason != model.SchedulingReasonUnspecified {
			return string(reason)
		}
	}

	return ""
}

// Close cancels pending acquisitions and stops activity sampling.
//
// Callers own the context and release function for each batch that already
// started. Close does not wait for those batches.
//
//	defer scheduler.Close()
func (s *Scheduler) Close() {
	s.cancel()
	s.scheduler.Close()
}
