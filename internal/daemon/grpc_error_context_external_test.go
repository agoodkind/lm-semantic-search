package daemon_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/sandbox"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestSearchCodeContextPreservesGRPCDetails(t *testing.T) {
	for _, variable := range sandbox.Env(t.TempDir()) {
		t.Setenv(variable.Name, variable.Value)
	}
	t.Setenv("CLAUDE_CONTEXT_PROFILE", config.ProfileStandard)
	t.Setenv("MILVUS_ADDRESS", "")
	t.Setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
	t.Setenv("OPENAI_API_KEY", "local-test-placeholder")
	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("load sandbox config: %v", err)
	}
	for _, directory := range sandbox.Directories(cfg) {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			t.Fatalf("create sandbox directory: %v", err)
		}
	}
	client, _ := startUnavailableStoreDaemon(t, cfg)
	for _, test := range []struct {
		name    string
		path    string
		context string
		detail  string
		code    codes.Code
		reason  string
	}{
		{
			name: "relative path", path: "relative-source",
			context: "resolve search request path:", detail: "pass an absolute path",
			code: codes.InvalidArgument, reason: "invalid_path",
		},
		{
			name: "URI path", path: "https://example.invalid/source",
			context: "search codebase:", detail: "looks like a URI",
			code: codes.InvalidArgument, reason: "invalid_path",
		},
		{
			name: "unindexed directory", path: t.TempDir(),
			context: "search codebase:", detail: "run the index_codebase tool",
			code: codes.NotFound, reason: "not_indexed",
		},
		{
			name: "unknown codebase ID", path: "cb_missing_context",
			context: "search codebase:", detail: "list tracked codebases",
			code: codes.NotFound, reason: "unknown_codebase_id",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_, err := client.SearchCode(ctx, &pb.SearchCodeRequest{Path: test.path, Query: "find source"})
			response := status.Convert(err)
			if response.Code() != test.code {
				t.Fatalf("SearchCode status = %s, want %s: %v", response.Code(), test.code, err)
			}
			if !strings.HasPrefix(response.Message(), test.context) || !strings.Contains(response.Message(), test.detail) {
				t.Fatalf("SearchCode message = %q, want context %q and detail %q", response.Message(), test.context, test.detail)
			}
			details := response.Details()
			if len(details) != 1 {
				t.Fatalf("SearchCode details = %v, want one ErrorInfo", details)
			}
			info, ok := details[0].(*errdetails.ErrorInfo)
			if !ok || info.GetReason() != test.reason || info.GetDomain() != "goodkind.io/lm-semantic-search" || len(info.GetMetadata()) != 0 {
				t.Fatalf("SearchCode ErrorInfo = %v, want reason %s and daemon domain without metadata", details[0], test.reason)
			}
		})
	}
}
