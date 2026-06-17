package adapterr

import (
	"errors"
	"fmt"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// IsGRPCUnavailable classifies the transport-level outages (a fully down backend
// answers Unavailable or DeadlineExceeded before any server status) and leaves
// every other status code and every non-gRPC error alone.
func TestIsGRPCUnavailable(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"unavailable", status.Error(codes.Unavailable, "connection refused"), true},
		{"deadline exceeded", status.Error(codes.DeadlineExceeded, "context deadline exceeded"), true},
		{"wrapped unavailable", fmt.Errorf("dial: %w", status.Error(codes.Unavailable, "down")), true},
		{"not found", status.Error(codes.NotFound, "missing"), false},
		{"internal", status.Error(codes.Internal, "boom"), false},
		{"plain error", errors.New("not a status"), false},
	}
	for _, testCase := range cases {
		if got := IsGRPCUnavailable(testCase.err); got != testCase.want {
			t.Fatalf("%s: IsGRPCUnavailable = %v, want %v", testCase.name, got, testCase.want)
		}
	}
}
