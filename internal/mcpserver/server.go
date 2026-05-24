// Package mcpserver exposes the daemon over the MCP stdio tool surface.
package mcpserver

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	pb "github.com/zilliztech/claude-context-go/gen/go/claudecontext/v1"
	"github.com/zilliztech/claude-context-go/internal/config"
	"github.com/zilliztech/claude-context-go/internal/grpcutil"
	"github.com/zilliztech/claude-context-go/internal/version"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

// Run starts the MCP stdio server and blocks until the client disconnects.
func Run(ctx context.Context) error {
	slog.InfoContext(ctx, "start MCP server")

	cfg, err := config.Default()
	if err != nil {
		slog.ErrorContext(ctx, "load MCP config failed", "err", err)
		return fmt.Errorf("load MCP config: %w", err)
	}

	server := mcp.NewServer(&mcp.Implementation{
		Name:    "claude-context",
		Version: version.Version,
	}, nil)

	registerIndexTool(server, cfg.SocketPath)
	registerClearTool(server, cfg.SocketPath)
	registerStatusTool(server, cfg.SocketPath)
	registerListIndexesTool(server, cfg.SocketPath)
	registerListJobsTool(server, cfg.SocketPath)
	registerGetJobTool(server, cfg.SocketPath)
	registerDoctorTool(server, cfg.SocketPath)
	registerSearchTool(server, cfg.SocketPath)

	if err := server.Run(ctx, &mcp.StdioTransport{}); err != nil {
		slog.ErrorContext(ctx, "run MCP server failed", "err", err)
		return fmt.Errorf("run MCP server: %w", err)
	}
	return nil
}

type emptyInput struct{}

type pathInput struct {
	Path string `json:"path" jsonschema:"absolute path to the codebase directory"`
}

type indexInput struct {
	Path             string   `json:"path" jsonschema:"absolute path to the codebase directory"`
	Force            bool     `json:"force,omitempty" jsonschema:"force reindex even if already indexed"`
	Splitter         string   `json:"splitter,omitempty" jsonschema:"splitter type, typically ast"`
	CustomExtensions []string `json:"customExtensions,omitempty" jsonschema:"extra file extensions to include"`
	IgnorePatterns   []string `json:"ignorePatterns,omitempty" jsonschema:"extra ignore patterns to exclude"`
}

type searchInput struct {
	Path            string   `json:"path" jsonschema:"absolute path to the codebase directory"`
	Query           string   `json:"query" jsonschema:"natural language code search query"`
	Limit           int32    `json:"limit,omitempty" jsonschema:"maximum number of results"`
	ExtensionFilter []string `json:"extensionFilter,omitempty" jsonschema:"optional file extensions filter"`
}

type listJobsInput struct {
	CodebaseID string `json:"codebase_id,omitempty" jsonschema:"optional codebase id to filter jobs"`
}

type getJobInput struct {
	JobID string `json:"job_id" jsonschema:"job id to inspect"`
}

type toolOutput struct {
	Message string `json:"message" jsonschema:"human-readable tool result"`
}

type daemonTextCall func(context.Context, pb.ClaudeContextDaemonServiceClient) (string, error)

func registerIndexTool(server *mcp.Server, socketPath string) {
	mcp.AddTool(server, indexToolDefinition(), func(ctx context.Context, req *mcp.CallToolRequest, input indexInput) (*mcp.CallToolResult, toolOutput, error) {
		output, err := callDaemon(ctx, socketPath, func(ctx context.Context, client pb.ClaudeContextDaemonServiceClient) (string, error) {
			response, err := client.StartIndex(ctx, &pb.StartIndexRequest{
				Path:             input.Path,
				Force:            input.Force,
				CustomExtensions: append([]string{}, input.CustomExtensions...),
				IgnorePatterns:   append([]string{}, input.IgnorePatterns...),
				Splitter:         &pb.SplitterConfig{Type: input.Splitter},
				Client:           &pb.ClientInfo{Name: "mcp"},
			})
			if err != nil {
				return "", fmt.Errorf("start index RPC: %w", err)
			}
			return "Started indexing job " + response.GetJobId() + " for " + response.GetCanonicalPath(), nil
		})
		if err != nil {
			slog.ErrorContext(ctx, "index tool failed", "err", err)
			return nil, toolOutput{}, err
		}
		return textResult(output), toolOutput{Message: output}, nil
	})
}

func registerClearTool(server *mcp.Server, socketPath string) {
	mcp.AddTool(server, clearToolDefinition(), func(ctx context.Context, req *mcp.CallToolRequest, input pathInput) (*mcp.CallToolResult, toolOutput, error) {
		output, err := callDaemon(ctx, socketPath, func(ctx context.Context, client pb.ClaudeContextDaemonServiceClient) (string, error) {
			response, err := client.ClearIndex(ctx, &pb.ClearIndexRequest{Path: input.Path, Client: &pb.ClientInfo{Name: "mcp"}})
			if err != nil {
				return "", fmt.Errorf("clear index RPC: %w", err)
			}
			return "Cleared codebase " + response.GetCodebaseId(), nil
		})
		if err != nil {
			slog.ErrorContext(ctx, "clear tool failed", "err", err)
			return nil, toolOutput{}, err
		}
		return textResult(output), toolOutput{Message: output}, nil
	})
}

func registerStatusTool(server *mcp.Server, socketPath string) {
	mcp.AddTool(server, statusToolDefinition(), func(ctx context.Context, req *mcp.CallToolRequest, input pathInput) (*mcp.CallToolResult, toolOutput, error) {
		output, err := callDaemon(ctx, socketPath, func(ctx context.Context, client pb.ClaudeContextDaemonServiceClient) (string, error) {
			response, err := client.GetIndex(ctx, &pb.GetIndexRequest{Path: input.Path})
			if err != nil {
				return "", fmt.Errorf("get index RPC: %w", err)
			}
			return renderProto(response), nil
		})
		if err != nil {
			slog.ErrorContext(ctx, "status tool failed", "err", err)
			return nil, toolOutput{}, err
		}
		return textResult(output), toolOutput{Message: output}, nil
	})
}

func registerListIndexesTool(server *mcp.Server, socketPath string) {
	mcp.AddTool(server, listIndexesToolDefinition(), func(ctx context.Context, req *mcp.CallToolRequest, input emptyInput) (*mcp.CallToolResult, toolOutput, error) {
		output, err := callDaemon(ctx, socketPath, func(ctx context.Context, client pb.ClaudeContextDaemonServiceClient) (string, error) {
			response, err := client.ListIndexes(ctx, &pb.ListIndexesRequest{})
			if err != nil {
				return "", fmt.Errorf("list indexes RPC: %w", err)
			}
			return renderProto(response), nil
		})
		if err != nil {
			slog.ErrorContext(ctx, "list indexes tool failed", "err", err)
			return nil, toolOutput{}, err
		}
		return textResult(output), toolOutput{Message: output}, nil
	})
}

func registerListJobsTool(server *mcp.Server, socketPath string) {
	mcp.AddTool(server, listJobsToolDefinition(), func(ctx context.Context, req *mcp.CallToolRequest, input listJobsInput) (*mcp.CallToolResult, toolOutput, error) {
		output, err := callDaemon(ctx, socketPath, func(ctx context.Context, client pb.ClaudeContextDaemonServiceClient) (string, error) {
			response, err := client.ListJobs(ctx, &pb.ListJobsRequest{CodebaseId: input.CodebaseID})
			if err != nil {
				return "", fmt.Errorf("list jobs RPC: %w", err)
			}
			return renderProto(response), nil
		})
		if err != nil {
			slog.ErrorContext(ctx, "list jobs tool failed", "err", err)
			return nil, toolOutput{}, err
		}
		return textResult(output), toolOutput{Message: output}, nil
	})
}

func registerGetJobTool(server *mcp.Server, socketPath string) {
	mcp.AddTool(server, getJobToolDefinition(), func(ctx context.Context, req *mcp.CallToolRequest, input getJobInput) (*mcp.CallToolResult, toolOutput, error) {
		output, err := callDaemon(ctx, socketPath, func(ctx context.Context, client pb.ClaudeContextDaemonServiceClient) (string, error) {
			response, err := client.GetJob(ctx, &pb.GetJobRequest{JobId: input.JobID})
			if err != nil {
				return "", fmt.Errorf("get job RPC: %w", err)
			}
			return renderProto(response), nil
		})
		if err != nil {
			slog.ErrorContext(ctx, "get job tool failed", "err", err)
			return nil, toolOutput{}, err
		}
		return textResult(output), toolOutput{Message: output}, nil
	})
}

func registerDoctorTool(server *mcp.Server, socketPath string) {
	mcp.AddTool(server, doctorToolDefinition(), func(ctx context.Context, req *mcp.CallToolRequest, input emptyInput) (*mcp.CallToolResult, toolOutput, error) {
		output, err := callDaemon(ctx, socketPath, func(ctx context.Context, client pb.ClaudeContextDaemonServiceClient) (string, error) {
			response, err := client.Doctor(ctx, &pb.DoctorRequest{})
			if err != nil {
				return "", fmt.Errorf("doctor RPC: %w", err)
			}
			return renderProto(response), nil
		})
		if err != nil {
			slog.ErrorContext(ctx, "doctor tool failed", "err", err)
			return nil, toolOutput{}, err
		}
		return textResult(output), toolOutput{Message: output}, nil
	})
}

func registerSearchTool(server *mcp.Server, socketPath string) {
	mcp.AddTool(server, searchToolDefinition(), func(ctx context.Context, req *mcp.CallToolRequest, input searchInput) (*mcp.CallToolResult, toolOutput, error) {
		output, err := callDaemon(ctx, socketPath, func(ctx context.Context, client pb.ClaudeContextDaemonServiceClient) (string, error) {
			response, err := client.SearchCode(ctx, &pb.SearchCodeRequest{
				Path:            input.Path,
				Query:           input.Query,
				Limit:           input.Limit,
				ExtensionFilter: append([]string{}, input.ExtensionFilter...),
			})
			if err != nil {
				return "", fmt.Errorf("search code RPC: %w", err)
			}
			return renderProto(response), nil
		})
		if err != nil {
			slog.ErrorContext(ctx, "search tool failed", "err", err)
			return nil, toolOutput{}, err
		}
		return textResult(output), toolOutput{Message: output}, nil
	})
}

func textResult(message string) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: message},
		},
	}
}

func callDaemon(ctx context.Context, socketPath string, call daemonTextCall) (string, error) {
	connection, client, err := grpcutil.DialDaemon(ctx, socketPath)
	if err != nil {
		slog.ErrorContext(ctx, "dial daemon failed", "socket_path", socketPath, "err", err)
		return "", fmt.Errorf("dial daemon: %w", err)
	}
	defer connection.Close()

	result, err := call(ctx, client)
	if err != nil {
		slog.ErrorContext(ctx, "daemon RPC failed", "socket_path", socketPath, "err", err)
		return "", fmt.Errorf("daemon RPC: %w", err)
	}
	return result, nil
}

func renderProto(message proto.Message) string {
	marshaler := protojson.MarshalOptions{
		Indent:    "  ",
		Multiline: true,
	}
	data, err := marshaler.Marshal(message)
	if err != nil {
		slog.Error("marshal daemon response failed", "err", err)
		return fmt.Sprintf("%v", message)
	}
	return string(data)
}

func indexToolDefinition() *mcp.Tool {
	return &mcp.Tool{
		Name:        "index_codebase",
		Description: "Index a codebase directory for semantic search through the daemon",
	}
}

func clearToolDefinition() *mcp.Tool {
	return &mcp.Tool{
		Name:        "clear_index",
		Description: "Clear a tracked codebase index through the daemon",
	}
}

func statusToolDefinition() *mcp.Tool {
	return &mcp.Tool{
		Name:        "get_indexing_status",
		Description: "Get the current indexing status of one codebase path",
	}
}

func listIndexesToolDefinition() *mcp.Tool {
	return &mcp.Tool{
		Name:        "list_indexing_statuses",
		Description: "List every tracked codebase and its current indexing status",
	}
}

func listJobsToolDefinition() *mcp.Tool {
	return &mcp.Tool{
		Name:        "list_indexing_jobs",
		Description: "List active and historical indexing jobs",
	}
}

func getJobToolDefinition() *mcp.Tool {
	return &mcp.Tool{
		Name:        "get_indexing_job",
		Description: "Get one indexing job by id",
	}
}

func doctorToolDefinition() *mcp.Tool {
	return &mcp.Tool{
		Name:        "doctor_indexing",
		Description: "Inspect local daemon indexing health and diagnostics",
	}
}

func searchToolDefinition() *mcp.Tool {
	return &mcp.Tool{
		Name:        "search_code",
		Description: "Search indexed code in the daemon",
	}
}
