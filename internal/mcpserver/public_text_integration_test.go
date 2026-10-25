package mcpserver_test

import (
	"context"
	"strings"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/response"
)

const (
	traceMarker        = "🔎"
	traceIDField       = "trace_id="
	missingJobID       = "job_missing"
	noTrackedJobsText  = "No tracked jobs."
	droppedSectionText = "Dropped codebases"
	publicToolLimit    = 2 * statusResponseLimit
)

type publicToolCall struct {
	name      string
	arguments map[string]any
	wantError bool
	wantText  string
	wantOnce  string
}

func TestPublicToolTextOmitsTraceAnnotations(t *testing.T) {
	socketPath, codebasePath := startStatusTestDaemon(t, model.CodebaseStatusDiscovered)
	client := newMCPProcessClient(t, socketPath, response.ModeHuman)
	calls := []publicToolCall{
		{name: "get_indexing_status", arguments: map[string]any{"absolutePath": codebasePath}, wantText: codebasePath, wantOnce: "codebase_id=" + statusTestCodebaseID},
		{name: "list_indexing_statuses", arguments: map[string]any{}, wantText: codebasePath},
		{name: "list_indexing_jobs", arguments: map[string]any{}, wantText: noTrackedJobsText},
		{name: "doctor_indexing", arguments: map[string]any{}, wantText: droppedSectionText},
		{name: "get_indexing_job", arguments: map[string]any{"job_id": missingJobID}, wantError: true, wantText: missingJobID},
	}
	for _, call := range calls {
		ctx, cancel := context.WithTimeout(context.Background(), publicToolLimit)
		request := statusToolRequest(call.arguments)
		request.Params.Name = call.name
		result, err := client.CallTool(ctx, request)
		cancel()
		if err != nil {
			t.Fatalf("call %s: %v", call.name, err)
		}
		if result.IsError != call.wantError || len(result.Content) != 1 {
			t.Fatalf("%s result = %#v, want one content item with error=%t", call.name, result, call.wantError)
		}
		text, ok := result.Content[0].(mcp.TextContent)
		if !ok {
			t.Fatalf("%s content type = %T, want text", call.name, result.Content[0])
		}
		t.Logf("%s text:\n%s", call.name, text.Text)
		if strings.Contains(text.Text, traceMarker) || strings.Contains(text.Text, traceIDField) {
			t.Fatalf("%s text contains a trace annotation:\n%s", call.name, text.Text)
		}
		if !strings.Contains(text.Text, call.wantText) {
			t.Fatalf("%s text = %q, want a result that contains %q", call.name, text.Text, call.wantText)
		}
		if call.wantOnce != "" && strings.Count(text.Text, call.wantOnce) != 1 {
			t.Fatalf("%s text contains %q %d times, want once:\n%s", call.name, call.wantOnce, strings.Count(text.Text, call.wantOnce), text.Text)
		}
	}
}
