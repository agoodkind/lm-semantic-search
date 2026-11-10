package main

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestUpdateStatusTreatsMissingStateAsEmpty(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("CLAUDE_CONTEXTD_CONFIG_ROOT", t.TempDir())
	t.Setenv("CLAUDE_CONTEXTD_STATE_ROOT", t.TempDir())

	cmd := newUpdateStatusCmd()
	var stdout bytes.Buffer
	cmd.SetOut(&stdout)

	if err := cmd.Execute(); err != nil {
		t.Fatalf("update status returned error: %v", err)
	}

	output := stdout.String()
	if !strings.Contains(output, "current version:") {
		t.Fatalf("update status output = %q, want current version", output)
	}
	if !strings.Contains(output, "current commit:") {
		t.Fatalf("update status output = %q, want current commit", output)
	}
	if !strings.Contains(output, "current buildHash:") {
		t.Fatalf("update status output = %q, want current buildHash", output)
	}
	if strings.Contains(output, "last check:") {
		t.Fatalf("update status output = %q, want no last check for empty state", output)
	}
	if strings.Contains(output, "applied tag:") {
		t.Fatalf("update status output = %q, want no applied tag for empty state", output)
	}
}

func TestRequestDaemonShutdownTreatsUnavailableAsNotRunning(t *testing.T) {
	restoreDialDaemon := replaceDialDaemon(func(ctx context.Context, socketPath string) (*grpc.ClientConn, pb.SemanticSearchDaemonServiceClient, error) {
		_ = ctx
		_ = socketPath
		return nil, shutdownClient{err: status.Error(codes.Unavailable, "daemon not running")}, nil
	})
	t.Cleanup(restoreDialDaemon)

	restarted, err := requestDaemonShutdown(context.Background(), "/tmp/missing.sock")
	if err != nil {
		t.Fatalf("requestDaemonShutdown returned error: %v", err)
	}
	if restarted {
		t.Fatalf("requestDaemonShutdown restarted = true, want false")
	}
}

func TestRequestDaemonShutdownPropagatesShutdownErrors(t *testing.T) {
	restoreDialDaemon := replaceDialDaemon(func(ctx context.Context, socketPath string) (*grpc.ClientConn, pb.SemanticSearchDaemonServiceClient, error) {
		_ = ctx
		_ = socketPath
		return nil, shutdownClient{err: status.Error(codes.PermissionDenied, "permission denied")}, nil
	})
	t.Cleanup(restoreDialDaemon)

	restarted, err := requestDaemonShutdown(context.Background(), "/tmp/protected.sock")
	if err == nil {
		t.Fatalf("requestDaemonShutdown returned nil error")
	}
	if restarted {
		t.Fatalf("requestDaemonShutdown restarted = true, want false")
	}
	if !strings.Contains(err.Error(), "shutdown daemon") {
		t.Fatalf("requestDaemonShutdown error = %q, want shutdown daemon context", err.Error())
	}
}

func TestRequestDaemonShutdownPropagatesDialErrors(t *testing.T) {
	dialErr := errors.New("bad socket path")
	restoreDialDaemon := replaceDialDaemon(func(ctx context.Context, socketPath string) (*grpc.ClientConn, pb.SemanticSearchDaemonServiceClient, error) {
		_ = ctx
		_ = socketPath
		return nil, nil, dialErr
	})
	t.Cleanup(restoreDialDaemon)

	restarted, err := requestDaemonShutdown(context.Background(), "/tmp/bad.sock")
	if !errors.Is(err, dialErr) {
		t.Fatalf("requestDaemonShutdown error = %v, want dial error", err)
	}
	if restarted {
		t.Fatalf("requestDaemonShutdown restarted = true, want false")
	}
}

func replaceDialDaemon(dial func(context.Context, string) (*grpc.ClientConn, pb.SemanticSearchDaemonServiceClient, error)) func() {
	originalDialDaemon := dialDaemon
	dialDaemon = dial
	return func() {
		dialDaemon = originalDialDaemon
	}
}

type shutdownClient struct {
	pb.SemanticSearchDaemonServiceClient
	err error
}

func (client shutdownClient) Shutdown(ctx context.Context, request *pb.ShutdownRequest, options ...grpc.CallOption) (*pb.ShutdownResponse, error) {
	_ = ctx
	_ = request
	_ = options
	return nil, client.err
}
