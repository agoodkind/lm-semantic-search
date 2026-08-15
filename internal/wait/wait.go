// Package wait blocks on a codebase becoming searchable. It polls the daemon's
// read-only status RPC on behalf of the CLI wait command and the MCP
// wait_for_indexing tool, and never starts or mutates an index.
package wait

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os/signal"
	"syscall"
	"time"

	daemonclient "goodkind.io/lm-semantic-search/client"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/response"
)

var watchPollInterval = 1500 * time.Millisecond

// readinessBuilding and readinessLoading are the two collection_readiness
// values that mean a build is still underway, so a not-searchable answer
// carrying either one is provisional rather than final.
const (
	readinessBuilding = "building"
	readinessLoading  = "loading"
)

// ErrWaitTimeout reports that the timeout expired before the codebase settled.
// The caller still receives the last status the daemon answered.
var ErrWaitTimeout = errors.New("wait timeout expired")

// ErrWaitCancelled reports that the context was cancelled, by an interrupt
// signal or by the caller, before the codebase settled.
var ErrWaitCancelled = errors.New("wait cancelled")

// ForIndexStatus polls GetIndex until the codebase becomes searchable,
// reaches a terminal nonready state, the context is cancelled, or the timeout
// expires. It returns the final GetIndexResponse and any error. The wait
// operation is read-only and never calls StartIndex.
//
// Terminal nonready means the daemon answered a definite not-searchable
// verdict with no active indexing job and collection_readiness outside the
// in-progress states (building, loading).
func ForIndexStatus(ctx context.Context, socketPath string, path string, timeout time.Duration) (*pb.GetIndexResponse, error) {
	clientInfo, err := response.CurrentClientInfo()
	if err != nil {
		slog.ErrorContext(ctx, "resolve client info failed", "path", path, "err", err)
		return nil, fmt.Errorf("resolve client info: %w", err)
	}
	return ForIndexStatusWithClientInfo(ctx, socketPath, path, timeout, clientInfo)
}

// ForIndexStatusWithClientInfo polls GetIndex using the supplied caller metadata.
func ForIndexStatusWithClientInfo(ctx context.Context, socketPath string, path string, timeout time.Duration, clientInfo *pb.ClientInfo) (*pb.GetIndexResponse, error) {
	signalCtx, stop := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(signalCtx, timeout)
		defer cancel()
	} else {
		ctx = signalCtx
	}

	connection, client, err := daemonclient.DialDaemon(ctx, socketPath)
	if err != nil {
		slog.ErrorContext(ctx, "dial daemon failed", "socket_path", socketPath, "err", err)
		return nil, fmt.Errorf("dial daemon: %w", err)
	}
	defer func() { _ = connection.Close() }()

	return forIndexStatusWithClient(ctx, client, path, clientInfo)
}

// forIndexStatusWithClient is the internal polling logic that can be used with
// a mocked client for testing. Every exit path returns the most recent status
// the daemon answered, so a timeout or an interrupt still shows the caller how
// far the build got.
func forIndexStatusWithClient(ctx context.Context, client pb.SemanticSearchDaemonServiceClient, path string, clientInfo *pb.ClientInfo) (*pb.GetIndexResponse, error) {
	ticker := time.NewTicker(watchPollInterval)
	defer ticker.Stop()
	var lastResponse *pb.GetIndexResponse

	for {
		current, getErr := client.GetIndex(ctx, &pb.GetIndexRequest{
			Path:   path,
			Client: clientInfo,
		})
		if current != nil {
			lastResponse = current
		}
		if getErr != nil {
			// A cancelled or expired context is the reason the RPC failed, so it
			// owns the error the caller sees rather than the transport message.
			if stopErr := contextStopError(ctx); stopErr != nil {
				return lastResponse, stopErr
			}
			slog.ErrorContext(ctx, "GetIndex failed", "path", path, "err", getErr)
			return nil, fmt.Errorf("GetIndex: %w", getErr)
		}
		if indexStatusIsFinal(current) {
			return current, nil
		}

		select {
		case <-ctx.Done():
			return lastResponse, contextStopError(ctx)
		case <-ticker.C:
		}
	}
}

// contextStopError maps a stopped context onto the sentinel the caller sees,
// and returns nil while the context is still live.
func contextStopError(ctx context.Context) error {
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return ErrWaitTimeout
	}
	if errors.Is(ctx.Err(), context.Canceled) {
		return ErrWaitCancelled
	}
	return nil
}

// indexStatusIsFinal reports whether this status is the answer the wait was
// after: the codebase is searchable, or the daemon returned a definite
// not-searchable verdict with no build underway.
//
// The searchable field is checked for presence, not just truthiness. An absent
// field means the daemon reached no verdict, which is not a false and is not
// terminal, so the wait keeps polling.
func indexStatusIsFinal(current *pb.GetIndexResponse) bool {
	if current == nil {
		return false
	}
	if current.GetSearchable() {
		return true
	}
	if current.Searchable == nil {
		return false
	}
	readiness := current.GetCollectionReadiness()
	if readiness == readinessBuilding || readiness == readinessLoading {
		return false
	}
	return current.GetActiveJob() == nil
}
