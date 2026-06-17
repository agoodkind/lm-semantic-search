package adapterr

import (
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// IsGRPCUnavailable reports whether err carries a gRPC status of Unavailable or
// DeadlineExceeded: the transport-level signals that a backend is unreachable or
// did not answer in time. A milvus that is fully down returns one of these from
// the transport before any server status is produced, so a typed milvus error
// (merr) does not classify it.
//
// This helper lives in adapterr because adapterr already depends on
// google.golang.org/grpc. Packages that must stay free of a direct grpc import,
// such as internal/semantic (where a direct grpc import makes the grpc-handler
// lint heuristic treat every *Service method as a handler), classify
// transport-level outages through this helper instead of importing grpc/status
// themselves.
func IsGRPCUnavailable(err error) bool {
	if err == nil {
		return false
	}
	grpcStatus, ok := status.FromError(err)
	if !ok {
		return false
	}
	code := grpcStatus.Code()
	return code == codes.Unavailable || code == codes.DeadlineExceeded
}
