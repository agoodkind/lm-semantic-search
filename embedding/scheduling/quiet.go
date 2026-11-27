// Package scheduling admits background embedding batches when the host is idle.
package scheduling

import (
	"context"
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
	ctx       context.Context
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
		ctx:       lifetime,
		cancel:    cancel,
		sequence:  atomic.Uint64{},
	}
}

// Acquire waits until the quiet policy admits one batch. The returned function
// releases the slot and must be called after the batch finishes.
func (s *Scheduler) Acquire(ctx context.Context) (func(), error) {
	if err := s.ctx.Err(); err != nil {
		return nil, err
	}

	batchCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	stop := context.AfterFunc(s.ctx, cancel)
	defer stop()
	sequence := s.sequence.Add(1)
	policy := model.DefaultSchedulingPolicy()
	policy.Quiet = true
	lease, err := s.scheduler.Acquire(batchCtx, jobscheduler.Entry{
		JobID:         strconv.FormatUint(sequence, 10),
		Policy:        policy,
		QueueSequence: sequence,
	})
	if err != nil {
		return nil, err
	}

	if err := s.ctx.Err(); err != nil {
		lease.Release()
		return nil, err
	}

	return lease.Release, nil
}

// WaitingReason returns the current quiet policy reason for a pending batch, or
// an empty string when the policy does not block any pending batch.
func (s *Scheduler) WaitingReason() string {
	for _, reason := range s.scheduler.Snapshot().Reasons {
		if reason != model.SchedulingReasonUnspecified {
			return string(reason)
		}
	}

	return ""
}

// Close cancels pending acquisitions and stops activity sampling. Callers own
// the context and release function for each batch that already started.
func (s *Scheduler) Close() {
	s.cancel()
	s.scheduler.Close()
}
