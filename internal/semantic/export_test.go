package semantic

import (
	"context"
	"time"
)

// Test hooks for the external reconnect test, which lives in package
// semantic_test so the production package never imports google.golang.org/grpc
// directly (the grpc-handler lint heuristic keys on that import). Each setter
// returns a restore function.

func SetBootDialTimeoutForTest(timeout time.Duration) func() {
	previous := bootDialTimeout
	bootDialTimeout = timeout
	return func() { bootDialTimeout = previous }
}

func SetReconnectSleepForTest(sleep func(context.Context, time.Duration) bool) func() {
	previous := reconnectSleep
	reconnectSleep = sleep
	return func() { reconnectSleep = previous }
}

func SetReconnectJitterForTest(jitter func(time.Duration) time.Duration) func() {
	previous := reconnectJitter
	reconnectJitter = jitter
	return func() { reconnectJitter = previous }
}

// WrapStoreErrorForTest exposes wrapStoreError to the external semantic_test
// package, where constructing a real gRPC transport error is allowed, so the
// store-outage classification of the write/index path can be tested without the
// production package importing google.golang.org/grpc.
func WrapStoreErrorForTest(ctx context.Context, err error, operation string) error {
	return wrapStoreError(ctx, err, operation)
}
