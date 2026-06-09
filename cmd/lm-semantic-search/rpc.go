package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"os/signal"
	"syscall"

	daemonclient "goodkind.io/lm-semantic-search/client"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/response"
	grpcstatus "google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

type protoMessage = proto.Message

type rpcCall func(context.Context, pb.SemanticSearchDaemonServiceClient) (protoMessage, error)

func currentClientInfo() (*pb.ClientInfo, error) {
	pid := os.Getpid()
	if pid < 0 || pid > math.MaxInt32 {
		return nil, fmt.Errorf("process id %d does not fit in int32", pid)
	}
	return &pb.ClientInfo{
		Name: "cli",
		Pid:  int32(pid),
	}, nil
}

// callDaemon dials the daemon, runs one RPC, and returns the raw proto reply.
// It is the shared seam under callAndPrint and the interactive list view, so the
// TUI can fetch records without the print step double-emitting output.
func callDaemon(options cliOptions, call rpcCall) (protoMessage, error) {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	connection, client, err := daemonclient.DialDaemon(ctx, options.socketPath)
	if err != nil {
		slog.Error("dial daemon failed", "socket_path", options.socketPath, "err", err)
		return nil, fmt.Errorf("dial daemon: %w", err)
	}
	defer connection.Close()

	result, err := call(ctx, client)
	if err != nil {
		slog.Error("daemon RPC failed", "socket_path", options.socketPath, "err", err)
		return nil, formatCallError(err)
	}
	return result, nil
}

func callAndPrint(options cliOptions, call rpcCall) error {
	result, err := callDaemon(options, call)
	if err != nil {
		return err
	}

	formatted, err := response.FormatProto(options.outputMode, result)
	if err != nil {
		slog.Error("format response failed", "err", err)
		return fmt.Errorf("format response: %w", err)
	}
	_, err = fmt.Fprintf(os.Stdout, "%s\n", formatted)
	if err != nil {
		return fmt.Errorf("write response output: %w", err)
	}
	return nil
}

func formatCallError(err error) error {
	grpcErr, ok := grpcstatus.FromError(err)
	if ok {
		return errors.New(grpcErr.Message())
	}
	return errors.New(err.Error())
}

func safeSearchLimit(limit int) (int32, error) {
	if limit < 0 {
		return 0, fmt.Errorf("limit must be non-negative")
	}
	if limit > math.MaxInt32 {
		return 0, fmt.Errorf("limit %d exceeds int32", limit)
	}
	return int32(limit), nil
}
