package daemon

import (
	"fmt"
	"log/slog"
	"sync"

	"goodkind.io/lm-semantic-search/internal/model"
)

const jobJournalQueueCapacity = 256

type jobJournalWriteRequest struct {
	event  model.JobEvent
	result chan error
}

type jobJournalWriter struct {
	path           string
	appendJobEvent appendJobEventFunc
	queue          chan jobJournalWriteRequest
	done           chan bool
	closeOnce      sync.Once
}

func newJobJournalWriter(
	path string,
	appendJobEvent appendJobEventFunc,
	queueCapacity int,
) *jobJournalWriter {
	writer := &jobJournalWriter{
		path:           path,
		appendJobEvent: appendJobEvent,
		queue:          make(chan jobJournalWriteRequest, queueCapacity),
		done:           make(chan bool, 1),
		closeOnce:      sync.Once{},
	}
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.Error(
					"job journal writer panic",
					"path",
					writer.path,
					"err",
					fmt.Errorf("panic: %v", recovered),
				)
			}
			writer.done <- true
		}()
		writer.run()
	}()
	return writer
}

func (writer *jobJournalWriter) enqueue(event model.JobEvent) error {
	request := jobJournalWriteRequest{event: event, result: nil}
	select {
	case writer.queue <- request:
		return nil
	default:
	}

	request.result = make(chan error, 1)
	slog.Warn(
		"job journal queue full; falling back to synchronous ordered write",
		"path",
		writer.path,
		"job_id",
		event.Job.ID,
		"event",
		event.Event,
		"queue_capacity",
		cap(writer.queue),
	)
	writer.queue <- request
	return <-request.result
}

func (writer *jobJournalWriter) run() {
	for request := range writer.queue {
		err := writer.write(request.event)
		if request.result != nil {
			request.result <- err
		}
	}
}

func (writer *jobJournalWriter) write(event model.JobEvent) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("append jobs journal %s: panic: %v", writer.path, recovered)
			slog.Error(
				"append jobs journal failed",
				"path",
				writer.path,
				"job_id",
				event.Job.ID,
				"event",
				event.Event,
				"err",
				err,
			)
		}
	}()
	return appendJobJournalEvent(writer.path, writer.appendJobEvent, event)
}

func (writer *jobJournalWriter) close() {
	writer.closeOnce.Do(func() {
		close(writer.queue)
		<-writer.done
	})
}

func appendJobJournalEvent(
	path string,
	appendJobEvent appendJobEventFunc,
	event model.JobEvent,
) error {
	if err := appendJobEvent(path, event); err != nil {
		wrappedErr := fmt.Errorf("append jobs journal %s: %w", path, err)
		slog.Error(
			"append jobs journal failed",
			"path",
			path,
			"job_id",
			event.Job.ID,
			"event",
			event.Event,
			"err",
			wrappedErr,
		)
		return wrappedErr
	}
	return nil
}

func (manager *Manager) closeJobJournal() {
	manager.mu.Lock()
	defer manager.mu.Unlock()

	if manager.jobJournal == nil {
		return
	}
	manager.jobJournal.close()
	manager.jobJournal = nil
}
