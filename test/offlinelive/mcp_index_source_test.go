//go:build offlinelive

package offlinelive

import (
	"context"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/v2/client"
	"github.com/mark3labs/mcp-go/v2/mcp"
)

const (
	mcpCommandPackage = "goodkind.io/lm-semantic-search/cmd/lm-semantic-search-mcp"
	mcpIndexToolName  = "index_codebase"
	mcpCallTimeout    = 3 * time.Minute
)

func buildMCPCommand(t *testing.T) string {
	t.Helper()

	binaryPath := filepath.Join(t.TempDir(), "lm-semantic-search-mcp")
	build := exec.CommandContext(context.Background(), "go", "build", "-o", binaryPath, mcpCommandPackage)
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("The lm-semantic-search-mcp command build failed with error %v and build output %s.", err, output)
	}
	return binaryPath
}

func gitHeadCommit(t *testing.T, repository string) string {
	t.Helper()

	command := exec.CommandContext(context.Background(), "git", "-C", repository, "rev-parse", "HEAD")
	output, err := command.Output()
	if err != nil {
		t.Fatalf("The git rev-parse HEAD command in repository %s failed with error %v.", repository, err)
	}
	return strings.TrimSpace(string(output))
}

func toolResultText(result *mcp.CallToolResult) string {
	texts := make([]string, 0, len(result.Content))
	for _, content := range result.Content {
		texts = append(texts, mcp.GetTextFromContent(content))
	}
	return strings.Join(texts, "\n")
}

func TestMCPIndexCodebaseRecordsSourcePathAndStartingCommit(t *testing.T) {
	binaryPath := buildMCPCommand(t)
	harness := newHarness(t)
	repository := newSmallCommittedRepository(t)
	worktree := addNestedWorktree(t, repository)
	startingCommit := gitHeadCommit(t, repository)

	mcpClient, err := client.NewStdioMCPClient(binaryPath, nil)
	if err != nil {
		t.Fatalf("The MCP command %s failed to start over stdio with error %v.", binaryPath, err)
	}
	t.Cleanup(func() { _ = mcpClient.Close() })
	callContext, cancel := context.WithTimeout(context.Background(), mcpCallTimeout)
	defer cancel()

	initializeRequest := mcp.InitializeRequest{}
	initializeRequest.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initializeRequest.Params.ClientInfo = mcp.Implementation{Name: "offline-live-harness", Version: "test"}
	if _, err := mcpClient.Initialize(callContext, initializeRequest); err != nil {
		t.Fatalf("The MCP initialize request failed with error %v.", err)
	}

	indexRequest := mcp.CallToolRequest{}
	indexRequest.Params.Name = mcpIndexToolName
	indexRequest.Params.Arguments = map[string]any{
		"absolutePath":   worktree,
		"splitter":       "ast",
		"sourcePath":     repository,
		"startingCommit": startingCommit,
		"wait":           true,
	}
	result, err := mcpClient.CallTool(callContext, indexRequest)
	if err != nil {
		t.Fatalf("The MCP request for tool %s returned protocol error %v.", mcpIndexToolName, err)
	}
	if result.IsError {
		t.Fatalf("The MCP tool %s returned an error result with text %s.", mcpIndexToolName, toolResultText(result))
	}

	codebase := harness.indexStatusAt(worktree).GetCodebase()
	if codebase.GetCanonicalPath() != worktree {
		t.Fatalf("GetIndex reports canonical path %q instead of the worktree %q.", codebase.GetCanonicalPath(), worktree)
	}
	if codebase.GetWorktreeSourcePath() != repository {
		t.Fatalf("The codebase records source path %q instead of the sourcePath input %q.", codebase.GetWorktreeSourcePath(), repository)
	}
	if codebase.GetWorktreeStartingCommit() != startingCommit {
		t.Fatalf("The codebase records starting commit %q instead of the startingCommit input %q.", codebase.GetWorktreeStartingCommit(), startingCommit)
	}
}
