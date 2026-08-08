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

// ForIndexStatus polls GetIndex until the codebase becomes searchable,
// reaches a terminal nonready state, the context is cancelled, or the timeout
// expires. It returns the final GetIndexResponse and any error. The wait
// operation is read-only and never calls StartIndex.
//
// Terminal nonready means the codebase has failed permanently (no searchable
// response) with no active indexing job and collection_readiness outside the
// in-progress states (building, loading).
func ForIndexStatus(ctx context.Context, socketPath string, path string, timeout time.Duration) (*pb.GetIndexResponse, error) {
	clientInfo, err := response.CurrentClientInfo()
	if err != nil {
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
		slog.Error("dial daemon failed", "err", err)
		return nil, fmt.Errorf("dial daemon: %w", err)
	}
	defer func() { _ = connection.Close() }()

	return forIndexStatusWithClient(ctx, client, path, clientInfo)
}

// forIndexStatusWithClient is the internal polling logic that can be used with
// a mocked client for testing.
func forIndexStatusWithClient(ctx context.Context, client pb.SemanticSearchDaemonServiceClient, path string, clientInfo *pb.ClientInfo) (*pb.GetIndexResponse, error) {
	ticker := time.NewTicker(watchPollInterval)
	defer ticker.Stop()
	var lastResponse *pb.GetIndexResponse

	for {
		current, getErr := client.GetIndex(ctx, &pb.GetIndexRequest{
			Path:   path,
			Client: clientInfo,
		})
		if getErr != nil {
			if errors.Is(ctx.Err(), context.Canceled) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
				if current != nil {
					lastResponse = current
				}
				if lastResponse != nil {
					if errors.Is(ctx.Err(), context.DeadlineExceeded) {
						return lastResponse, fmt.Errorf("wait timeout expired")
					}
					return lastResponse, fmt.Errorf("wait cancelled")
				}
				if errors.Is(ctx.Err(), context.DeadlineExceeded) {
					return nil, fmt.Errorf("wait timeout expired")
				}
				return nil, fmt.Errorf("wait cancelled")
			}
			slog.Error("GetIndex failed", "err", getErr)
			return nil, fmt.Errorf("GetIndex: %w", getErr)
		}
		if current != nil {
			lastResponse = current
		}

		// Check if searchable
		if current != nil && current.Searchable != nil && *current.Searchable {
			return current, nil
		}

		// Check for terminal nonready: no active job and readiness not in progress
		readiness := current.GetCollectionReadiness()
		isInProgress := readiness == "building" || readiness == "loading"
		hasActiveJob := current.GetActiveJob() != nil

		if current != nil && current.Searchable != nil && !isInProgress && !hasActiveJob {
			// Terminal nonready state
			return current, nil
		}

		select {
		case <-ctx.Done():
			if errors.Is(ctx.Err(), context.DeadlineExceeded) {
				return lastResponse, fmt.Errorf("wait timeout expired")
			}
			return lastResponse, fmt.Errorf("wait cancelled")
		case <-ticker.C:
		}
	}
}
