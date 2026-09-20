package semantic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"goodkind.io/lm-semantic-search/internal/clock"
)

const (
	// collectionLoadBackoffInitial is the first pause after Milvus reports memory
	// exhaustion. Thirty seconds is long enough for the query node to release the
	// segments of a failed load before the daemon asks for another.
	collectionLoadBackoffInitial = 30 * time.Second
	// collectionLoadBackoffMax bounds the pause however many loads fail in a row,
	// so a Milvus that stays short of memory is retried every five minutes
	// rather than never.
	collectionLoadBackoffMax = 5 * time.Minute
	// collectionLoadBackoffFactor is the growth between consecutive pauses.
	collectionLoadBackoffFactor = 2
)

// errCollectionLoadUnrecovered marks a load that exhausted both polls and the
// single re-issued request without the collection becoming queryable. It is
// wrapped alongside ErrCollectionNotReady so the backoff can tell this
// terminal outcome from the other not-ready errors the load path returns.
var errCollectionLoadUnrecovered = errors.New("collection load did not recover after a re-issued load")

// milvusMemoryExhaustionMessages are the lowercased texts the Milvus Go client
// surfaces when a load is refused for memory. The client (merr.Error in
// github.com/milvus-io/milvus/pkg/v2/util/merr) rebuilds the server status
// Reason as the error message, and the leaf messages come from merr/errors.go:
// "memory limit exceeded" is ErrServiceMemoryLimitExceeded (code 3, legacy
// ErrorCode_InsufficientMemoryToLoad and ErrorCode_MemoryQuotaExhausted) and
// "service resource insufficient" is ErrServiceResourceInsufficient (code 12).
// merr's wrapFields appends "[key=value]" fields, including the query node's
// "resourceType=Memory" field. "OOM if load" is the query node segment
// loader's own message. Text is matched rather than the typed sentinel
// because importing merr breaks the gate's govulncheck, as store_errors.go
// records for the not-loaded message.
var milvusMemoryExhaustionMessages = []string{
	"memory limit exceeded",
	"service resource insufficient",
	"resourcetype=memory",
	"oom if load",
	"insufficient memory",
}

// milvusMemoryExhaustion reports whether a load error carries one of the Milvus
// memory-exhaustion messages.
func milvusMemoryExhaustion(err error) bool {
	if err == nil {
		return false
	}
	lowered := strings.ToLower(err.Error())
	for _, message := range milvusMemoryExhaustionMessages {
		if strings.Contains(lowered, message) {
			return true
		}
	}
	return false
}

// collectionLoadMemorySignal reports whether a load outcome means Milvus could
// not fit the collection. A direct memory error is one signal. A load that
// never finished after a re-issued request is the other, and it is the only
// one the daemon actually observed when Milvus ran out of memory: the query
// node retries the failed segment loads itself, so the client saw no error
// text, only a collection that stayed loading past every bound. A caller's own
// wait timeout, a single stuck poll that then recovered, a cancelled context,
// and a transport outage are not signals: each has its own cause and its own
// handling, and pausing loads on them would stall a healthy store.
func collectionLoadMemorySignal(err error) bool {
	if err == nil {
		return false
	}
	return errors.Is(err, errCollectionLoadUnrecovered) || milvusMemoryExhaustion(err)
}

// collectionLoadBackoff pauses new collection loads after Milvus reports
// memory exhaustion. Milvus reports no free-memory figure the daemon could
// consult, and a refused load costs Milvus the segments it already read, so
// the only safe reaction is to stop asking for a bounded interval. Loads
// already in flight finish on their own. The pause grows on every further
// signal up to collectionLoadBackoffMax and resets once any load succeeds.
type collectionLoadBackoff struct {
	now      func() time.Time
	mutex    sync.Mutex
	active   bool
	since    time.Time
	until    time.Time
	interval time.Duration
	signals  int
}

func newCollectionLoadBackoff() *collectionLoadBackoff {
	return &collectionLoadBackoff{
		now:      clock.Now,
		mutex:    sync.Mutex{},
		active:   false,
		since:    time.Time{},
		until:    time.Time{},
		interval: 0,
		signals:  0,
	}
}

// admit reports whether a load of collectionName may start now. During a pause
// it returns ErrCollectionLoadDeferred at once, so the caller fails fast
// instead of queueing behind a store that cannot take the load. The first
// admission after the pause elapses ends the backoff and logs that transition
// once.
func (backoff *collectionLoadBackoff) admit(ctx context.Context, collectionName string) error {
	backoff.mutex.Lock()
	defer backoff.mutex.Unlock()
	if !backoff.active {
		return nil
	}
	now := backoff.now()
	if !now.Before(backoff.until) {
		backoff.endLocked(ctx, "pause_elapsed")
		return nil
	}
	remaining := backoff.until.Sub(now).Round(time.Second)
	err := fmt.Errorf(
		"load of collection %s deferred for %s: %w",
		collectionName,
		remaining,
		ErrCollectionLoadDeferred,
	)
	slog.WarnContext(ctx, "semantic.collection_load_deferred",
		"component", "semantic",
		"subcomponent", "load",
		"collection", collectionName,
		"remaining_ms", remaining.Milliseconds(),
		"signals", backoff.signals,
		"err", err,
	)
	return err
}

// noteLoadOutcome records how one load transition ended. A success ends any
// pause and resets the interval. A memory signal starts a pause, or extends
// the current one with the next larger interval. Every other failure leaves
// the backoff untouched. Entering a pause is logged once; a further signal
// during the pause only extends it.
func (backoff *collectionLoadBackoff) noteLoadOutcome(
	ctx context.Context,
	collectionName string,
	err error,
) {
	backoff.mutex.Lock()
	defer backoff.mutex.Unlock()
	if err == nil {
		if backoff.active {
			backoff.endLocked(ctx, "load_succeeded")
		}
		backoff.interval = 0
		return
	}
	if !collectionLoadMemorySignal(err) {
		return
	}
	now := backoff.now()
	backoff.interval = nextCollectionLoadBackoffInterval(backoff.interval)
	backoff.until = now.Add(backoff.interval)
	backoff.signals++
	if backoff.active {
		slog.DebugContext(ctx, "semantic.collection_load_backoff_extended",
			"component", "semantic",
			"subcomponent", "load",
			"collection", collectionName,
			"pause_ms", backoff.interval.Milliseconds(),
			"signals", backoff.signals,
			"err", err,
		)
		return
	}
	backoff.active = true
	backoff.since = now
	slog.WarnContext(ctx, "semantic.collection_load_backoff_started",
		"component", "semantic",
		"subcomponent", "load",
		"collection", collectionName,
		"pause_ms", backoff.interval.Milliseconds(),
		"signals", backoff.signals,
		"err", err,
	)
}

// endLocked leaves the pause and logs that transition once. The caller holds
// the mutex.
func (backoff *collectionLoadBackoff) endLocked(ctx context.Context, reason string) {
	backoff.active = false
	slog.InfoContext(ctx, "semantic.collection_load_backoff_ended",
		"component", "semantic",
		"subcomponent", "load",
		"reason", reason,
		"paused_ms", backoff.now().Sub(backoff.since).Milliseconds(),
		"signals", backoff.signals,
	)
	backoff.signals = 0
}

// nextCollectionLoadBackoffInterval grows the pause geometrically from the
// initial value up to the cap.
func nextCollectionLoadBackoffInterval(current time.Duration) time.Duration {
	if current <= 0 {
		return collectionLoadBackoffInitial
	}
	next := current * collectionLoadBackoffFactor
	if next > collectionLoadBackoffMax {
		return collectionLoadBackoffMax
	}
	return next
}
