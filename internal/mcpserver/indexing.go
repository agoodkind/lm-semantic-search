package mcpserver

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/mark3labs/mcp-go/v2/mcp"
	"github.com/mark3labs/mcp-go/v2/server"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/response"
	"goodkind.io/lm-semantic-search/internal/wait"
	"google.golang.org/protobuf/proto"
)

func registerIndexTool(mcpServer *server.MCPServer, socketPath string, outputMode response.Mode) {
	mcpServer.AddTool(
		indexToolDefinition(),
		wrapTool(
			"index_codebase",
			func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
				absolutePath, errResult, ok := requireNonEmptyArg(req, "absolutePath")
				if !ok {
					return errResult, nil
				}

				schedulingPolicy, err := indexSchedulingPolicyPatch(ctx, req)
				if err != nil {
					return toolErrorResult(err.Error()), nil
				}

				startRequest := &pb.StartIndexRequest{
					Path:              absolutePath,
					Force:             req.GetBool("force", false),
					IgnorePatterns:    req.GetStringSlice("ignorePatterns", []string{}),
					IncludeSubmodules: req.GetStringSlice("includeSubmodules", []string{}),
					Splitter:          &pb.SplitterConfig{Type: req.GetString("splitter", "")},
					Client:            mcpClientInfo(),
					SchedulingPolicy:  schedulingPolicy,
					SourcePath:        req.GetString("sourcePath", ""),
					StartingCommit:    req.GetString("startingCommit", ""),
				}
				if !req.GetBool("wait", false) {
					return callDaemonTool(
						ctx,
						socketPath,
						outputMode,
						func(ctx context.Context, client pb.SemanticSearchDaemonServiceClient) (
							proto.Message,
							error,
						) {
							return client.StartIndex(ctx, startRequest)
						},
					)
				}

				timeoutSeconds := req.GetInt("wait_timeout_seconds", defaultIndexWaitSeconds)
				if timeoutSeconds <= 0 {
					timeoutSeconds = defaultIndexWaitSeconds
				}

				return callDaemonIndexAndWait(
					ctx,
					socketPath,
					outputMode,
					startRequest,
					time.Duration(timeoutSeconds)*time.Second,
				)
			},
		),
	)
}

func registerWaitForIndexingTool(
	mcpServer *server.MCPServer,
	socketPath string,
	outputMode response.Mode,
) {
	mcpServer.AddTool(
		mcp.NewTool(
			"wait_for_indexing",
			mcp.WithDescription("Wait for a codebase to become searchable through the daemon"),
			mcp.WithString(
				"absolutePath",
				mcp.Required(),
				mcp.Description("absolute path to the codebase directory"),
			),
			mcp.WithNumber(
				"wait_timeout_seconds",
				mcp.Description(
					"max seconds to wait; on timeout the tool returns the current progress (default 300)",
				),
			),
		),
		wrapTool(
			"wait_for_indexing",
			func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
				absolutePath, errResult, ok := requireNonEmptyArg(req, "absolutePath")
				if !ok {
					return errResult, nil
				}

				timeoutSeconds := req.GetInt("wait_timeout_seconds", defaultIndexWaitSeconds)
				if timeoutSeconds <= 0 {
					timeoutSeconds = defaultIndexWaitSeconds
				}

				timeout := time.Duration(timeoutSeconds) * time.Second
				result, waitErr := wait.ForIndexStatusWithClientInfo(
					ctx,
					socketPath,
					absolutePath,
					timeout,
					mcpClientInfo(),
				)
				// A timeout or an interrupt still carries the last status the daemon
				// answered, and that progress is the useful reply, so it renders like
				// any other status. Only a wait that produced no status at all is
				// reported as a tool error.
				if result == nil {
					message := "wait for indexing returned no status"
					if waitErr != nil {
						message = "wait for indexing: " + waitErr.Error()
					}

					slog.ErrorContext(
						ctx,
						"wait for indexing returned no status",
						"path",
						absolutePath,
						"err",
						waitErr,
					)

					return toolErrorResult(message), nil
				}

				return renderToolResponse(outputMode, result)
			},
		),
	)
}

// callDaemonIndexAndWait starts an index job through StartIndex and polls
// GetJob until the job reaches a terminal state (completed, failed,
// canceled) or the supplied timeout elapses. When the timeout trips, the
// daemon job keeps running and the tool returns the latest GetIndex view so
// the caller can decide to poll again or move on. The poll cadence is short
// (~1.5s) so terminal events propagate to the caller with minimal delay.
//
// Concurrent waiters dedupe at the daemon (see Manager.dedupAgainstActiveJob)
// because StartIndex returns the existing job id when one is in flight, so
// every waiter ends up polling the same job and observing the same terminal
// event.
func callDaemonIndexAndWait(
	ctx context.Context,
	socketPath string,
	outputMode response.Mode,
	startRequest *pb.StartIndexRequest,
	timeout time.Duration,
) (*mcp.CallToolResult, error) {
	connection, client, err := grpcutil.DialDaemon(ctx, socketPath)
	if err != nil {
		slog.ErrorContext(ctx, "dial daemon for wait failed", "socket_path", socketPath, "err", err)

		return nil, fmt.Errorf("dial daemon: %w", err)
	}
	defer connection.Close()

	outgoingCtx := grpcutil.WithCorrelation(ctx)
	startResponse, err := client.StartIndex(outgoingCtx, startRequest)
	if err != nil {
		slog.ErrorContext(
			ctx,
			"start index for wait failed",
			"path",
			startRequest.GetPath(),
			"err",
			err,
		)

		return toolErrorResult(rpcErrorText(err)), nil
	}

	jobID := startResponse.GetJobId()
	if jobID == "" {
		return renderToolResponse(outputMode, startResponse)
	}

	waitCtx, cancel := context.WithTimeout(outgoingCtx, timeout)
	defer cancel()
	ticker := time.NewTicker(indexWaitPollInterval)
	defer ticker.Stop()

	for {
		jobResponse, err := client.GetJob(waitCtx, &pb.GetJobRequest{JobId: jobID})
		if err == nil && isTerminalJobState(jobResponse.GetJob().GetState()) {
			indexResponse, err := client.GetIndex(
				outgoingCtx,
				&pb.GetIndexRequest{Path: startRequest.GetPath(), Client: mcpClientInfo()},
			)
			if err != nil {
				slog.ErrorContext(
					ctx,
					"get index after wait failed",
					"path",
					startRequest.GetPath(),
					"err",
					err,
				)

				return renderToolResponse(outputMode, jobResponse)
			}

			return renderToolResponse(outputMode, indexResponse)
		}

		select {
		case <-waitCtx.Done():
			indexResponse, err := client.GetIndex(
				outgoingCtx,
				&pb.GetIndexRequest{Path: startRequest.GetPath(), Client: mcpClientInfo()},
			)
			if err != nil {
				slog.ErrorContext(
					ctx,
					"get index after wait timeout failed",
					"path",
					startRequest.GetPath(),
					"err",
					err,
				)

				return renderToolResponse(outputMode, startResponse)
			}

			return renderToolResponse(outputMode, indexResponse)
		case <-ticker.C:
		}
	}
}

// isTerminalJobState reports whether a JobState value will not change again.
// The daemon emits JobState through the proto's `state` string field; the
// comparison goes through model.JobState so the switch stays on typed
// constants rather than bare string literals.
func isTerminalJobState(state string) bool {
	switch model.JobState(state) {
	case model.JobStateCompleted, model.JobStateFailed, model.JobStateCancelled:
		return true
	case model.JobStateQueued,
		model.JobStateRunning,
		model.JobStatePaused,
		model.JobStateCancelling:
		return false
	default:
		return false
	}
}

func indexToolDefinition() mcp.Tool {
	return mcp.NewTool(
		"index_codebase",
		mcp.WithDescription("Index a codebase directory for semantic search through the daemon"),
		mcp.WithString(
			"absolutePath",
			mcp.Required(),
			mcp.Description("absolute path to the codebase directory"),
		),
		mcp.WithBoolean("force", mcp.Description("force reindex even if already indexed")),
		mcp.WithString("splitter", mcp.Description("splitter type, typically ast")),
		mcp.WithArray(
			"ignorePatterns",
			mcp.Description("extra ignore patterns to exclude"),
			mcp.WithStringItems(),
		),
		mcp.WithArray(
			"includeSubmodules",
			mcp.Description("submodule names or paths to include"),
			mcp.WithStringItems(),
		),
		mcp.WithString(
			"sourcePath",
			mcp.Description(
				"Record the source worktree path or codebase identifier in the same Git repository. "+
					"Omit the input to preserve the stored source.",
			),
		),
		mcp.WithString(
			"startingCommit",
			mcp.Description(
				"Record the known starting commit. "+
					"The daemon rejects the request when the target has no stored source path "+
					"and the request supplies no source path. Omit the input to preserve the stored value.",
			),
		),
		mcp.WithString(
			"priority",
			mcp.Description("scheduling priority for this index request"),
			mcp.Enum("high", "normal", "low"),
		),
		mcp.WithBoolean("quiet", mcp.Description("wait for host idle for this index request")),
		mcp.WithNumber(
			"idle_after_seconds",
			mcp.Description("host idle threshold in whole seconds for this index request"),
		),
		mcp.WithBoolean(
			"wait",
			mcp.Description(
				"block this tool call until the indexing job reaches a terminal state (completed, failed, or canceled)",
			),
		),
		mcp.WithNumber(
			"wait_timeout_seconds",
			mcp.Description(
				"max seconds to wait when wait=true; on timeout the daemon job keeps running "+
					"and the tool returns the current progress (default 300)",
			),
		),
	)
}
