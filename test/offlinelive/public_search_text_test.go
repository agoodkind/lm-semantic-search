//go:build offlinelive

package offlinelive

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/mark3labs/mcp-go/v2/client"
	"github.com/mark3labs/mcp-go/v2/mcp"
)

const (
	cliCommandPackage   = "goodkind.io/lm-semantic-search/cmd/lm-semantic-search"
	mcpSearchToolName   = "search_code"
	publicSearchLimit   = 5
	searchHeaderFormat  = "Found %d results for query:"
	searchLocationLabel = "Location: "
)

// publicTextDecorations lists the strings that normal public search text must not contain.
var publicTextDecorations = []string{"\U0001F50D", "\U0001F50E", "trace_id="}

func buildCLICommand(t *testing.T) string {
	t.Helper()

	binaryPath := filepath.Join(t.TempDir(), "lm-semantic-search")
	build := exec.CommandContext(context.Background(), "go", "build", "-o", binaryPath, cliCommandPackage)
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("The lm-semantic-search command build failed with error %v and build output %s.", err, output)
	}
	return binaryPath
}

func requirePublicSearchText(t *testing.T, surface string, text string) {
	t.Helper()

	t.Logf("%s text:\n%s", surface, text)
	for _, decoration := range publicTextDecorations {
		if strings.Contains(text, decoration) {
			t.Fatalf("The %s search text contains %q:\n%s", surface, decoration, text)
		}
	}
	headerPrefix := fmt.Sprintf(searchHeaderFormat, publicSearchLimit)
	if !strings.HasPrefix(text, headerPrefix) {
		t.Fatalf("The %s search text does not start with %q:\n%s", surface, headerPrefix, text)
	}
	if !strings.Contains(text, searchLocationLabel) {
		t.Fatalf("The %s search text does not contain a result %q line:\n%s", surface, searchLocationLabel, text)
	}
}

// TestPublicSearchTextOmitsDecorations checks MCP and CLI search text for results without magnifier or trace annotations.
func TestPublicSearchTextOmitsDecorations(t *testing.T) {
	mcpBinary := buildMCPCommand(t)
	cliBinary := buildCLICommand(t)
	harness := newHarness(t)
	requireCompleted(t, harness.indexFixture())

	mcpClient, err := client.NewStdioMCPClient(mcpBinary, nil)
	if err != nil {
		t.Fatalf("The MCP command %s failed to start over stdio with error %v.", mcpBinary, err)
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
	searchRequest := mcp.CallToolRequest{}
	searchRequest.Params.Name = mcpSearchToolName
	searchRequest.Params.Arguments = map[string]any{
		"absolutePath": harness.fixturePath,
		"query":        fixtureQuery,
		"limit":        publicSearchLimit,
	}
	result, err := mcpClient.CallTool(callContext, searchRequest)
	if err != nil {
		t.Fatalf("The MCP request for tool %s returned protocol error %v.", mcpSearchToolName, err)
	}
	if result.IsError {
		t.Fatalf("The MCP tool %s returned an error result with text %s.", mcpSearchToolName, toolResultText(result))
	}
	requirePublicSearchText(t, "MCP "+mcpSearchToolName, toolResultText(result))

	command := exec.CommandContext(callContext, cliBinary, "--socket", harness.socketPath, "codebase", "search",
		harness.fixturePath, fixtureQuery, "--limit", strconv.Itoa(publicSearchLimit))
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	if err := command.Run(); err != nil {
		t.Fatalf("The CLI codebase search command failed with error %v and stderr %s.", err, stderr.String())
	}
	requirePublicSearchText(t, "CLI codebase search", stdout.String()+stderr.String())
}
