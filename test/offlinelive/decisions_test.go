//go:build offlinelive

package offlinelive

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

const (
	retainedDecisionsContent   = "## PURPOSE\nSynthetic retained decision record\n"
	retainedDecisionsDirectory = "retained-decisions"
	decisionToolName           = "manage_adr"
	decisionStatusUpdated      = "updated"
	decisionChildDirectory     = "child"
	decisionParentSourceName   = "root.go"
	decisionParentSource       = "package parent\n\nfunc ParentRoot() string {\n\treturn \"parent\"\n}\n"
	decisionFilePermission     = 0o644
)

// decisionToolEnvelope stores the manage_adr fields that the tests read.
type decisionToolEnvelope struct {
	Content []struct {
		Text string `json:"text"`
	} `json:"content"`
	IsError bool `json:"isError"`
}

// decisionToolResult stores the manage_adr result fields that the tests read.
type decisionToolResult struct {
	Status  string `json:"status"`
	Content string `json:"content"`
}

// TestClearIndexKeepsDecisionsForNextIndex calls ClearIndex on an indexed
// codebase, indexes the same path under a new codebase ID, and reads the
// decision record with GraphTool.
func TestClearIndexKeepsDecisionsForNextIndex(t *testing.T) {
	harness := newHarness(t)
	requireCompleted(t, harness.indexFixture())
	first := harness.indexStatus().GetCodebase()
	harness.updateDecisions()

	harness.clearIndex()
	databasePath := filepath.Join(harness.config.GraphDir, first.GetId()+".db")
	for _, path := range []string{databasePath, databasePath + ".adr"} {
		if _, statErr := os.Stat(path); !errors.Is(statErr, os.ErrNotExist) {
			t.Fatalf("Stat(%q) returned %v instead of os.ErrNotExist", path, statErr)
		}
	}

	requireCompleted(t, harness.indexFixture())
	second := harness.indexStatus().GetCodebase()
	if second.GetId() == first.GetId() {
		t.Fatalf("The daemon reused codebase ID %q instead of creating a new ID", second.GetId())
	}
	if got := harness.readDecisions(); got != retainedDecisionsContent {
		t.Fatalf("The decision record after ClearIndex and a new index is %q instead of %q", got, retainedDecisionsContent)
	}
	retainedFiles, err := os.ReadDir(filepath.Join(harness.config.GraphDir, retainedDecisionsDirectory))
	if err != nil {
		t.Fatalf("ReadDir returned an error: %v", err)
	}
	if len(retainedFiles) != 0 {
		t.Fatalf("The retained decisions directory contains %d files after restore instead of 0", len(retainedFiles))
	}
}

// TestAbsorbedChildKeepsDecisions indexes a child path, then its parent path.
// The parent index removes the child codebase. The test clears the parent,
// indexes the child path again, and reads the decision record with GraphTool.
func TestAbsorbedChildKeepsDecisions(t *testing.T) {
	harness := newHarness(t)
	parentPath := filepath.Join(t.TempDir(), "parent")
	childPath := filepath.Join(parentPath, decisionChildDirectory)
	if err := os.CopyFS(childPath, os.DirFS(harness.fixturePath)); err != nil {
		t.Fatalf("CopyFS returned an error: %v", err)
	}
	if err := os.WriteFile(filepath.Join(parentPath, decisionParentSourceName), []byte(decisionParentSource), decisionFilePermission); err != nil {
		t.Fatalf("WriteFile returned an error: %v", err)
	}

	harness.fixturePath = childPath
	requireCompleted(t, harness.indexFixture())
	child := harness.indexStatus().GetCodebase()
	harness.updateDecisions()

	harness.fixturePath = parentPath
	requireCompleted(t, harness.indexFixture())
	childDatabasePath := filepath.Join(harness.config.GraphDir, child.GetId()+".db")
	if _, statErr := os.Stat(childDatabasePath); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("Stat returned %v for the removed child graph instead of os.ErrNotExist", statErr)
	}

	harness.clearIndex()
	harness.fixturePath = childPath
	requireCompleted(t, harness.indexFixture())
	if got := harness.readDecisions(); got != retainedDecisionsContent {
		t.Fatalf("The decision record after child codebase removal and a new index is %q instead of %q", got, retainedDecisionsContent)
	}
}

// TestClearIndexKeepsGraphWhenDecisionsCannotBeRetained blocks the retained
// decisions directory with a file. ClearIndex must fail and keep the codebase,
// its graph, and its decision record.
func TestClearIndexKeepsGraphWhenDecisionsCannotBeRetained(t *testing.T) {
	harness := newHarness(t)
	requireCompleted(t, harness.indexFixture())
	harness.updateDecisions()
	blockingPath := filepath.Join(harness.config.GraphDir, retainedDecisionsDirectory)
	if err := os.WriteFile(blockingPath, []byte("blocks directory"), decisionFilePermission); err != nil {
		t.Fatalf("WriteFile returned an error: %v", err)
	}

	_, err := harness.client.ClearIndex(correlatedContext(), &pb.ClearIndexRequest{
		Path:   harness.fixturePath,
		Client: harnessClientInfo(),
	})
	if err == nil {
		t.Fatal("ClearIndex returned nil instead of an error for the failed decision copy")
	}

	harness.indexStatus()
	if got := harness.readDecisions(); got != retainedDecisionsContent {
		t.Fatalf("The decision record after the failed ClearIndex is %q instead of %q", got, retainedDecisionsContent)
	}
}

func (harness *harness) clearIndex() {
	harness.t.Helper()
	_, err := harness.client.ClearIndex(correlatedContext(), &pb.ClearIndexRequest{
		Path:   harness.fixturePath,
		Client: harnessClientInfo(),
	})
	if err != nil {
		harness.t.Fatalf("ClearIndex returned an error: %v", err)
	}
}

func (harness *harness) updateDecisions() {
	harness.t.Helper()
	arguments, err := json.Marshal(map[string]string{"mode": "update", "content": retainedDecisionsContent})
	if err != nil {
		harness.t.Fatalf("Marshal returned an error: %v", err)
	}
	result := harness.callDecisionTool(string(arguments))
	if result.Status != decisionStatusUpdated {
		harness.t.Fatalf("manage_adr returned status %q instead of %q", result.Status, decisionStatusUpdated)
	}
}

func (harness *harness) readDecisions() string {
	harness.t.Helper()
	return harness.callDecisionTool(`{"mode":"get"}`).Content
}

// callDecisionTool uses GraphTool to call manage_adr and returns its result.
func (harness *harness) callDecisionTool(arguments string) decisionToolResult {
	harness.t.Helper()
	response, err := harness.client.GraphTool(correlatedContext(), &pb.GraphToolRequest{
		Path:     harness.fixturePath,
		Client:   harnessClientInfo(),
		ToolName: decisionToolName,
		ArgsJson: arguments,
	})
	if err != nil {
		harness.t.Fatalf("GraphTool returned an error: %v", err)
	}
	rawResponse := response.GetResultJson()
	var envelope decisionToolEnvelope
	if err = json.Unmarshal([]byte(rawResponse), &envelope); err != nil {
		harness.t.Fatalf("Unmarshal returned an error: %v: %s", err, rawResponse)
	}
	if envelope.IsError || len(envelope.Content) != 1 {
		harness.t.Fatalf("manage_adr returned an error or %d content items: %s", len(envelope.Content), rawResponse)
	}
	var result decisionToolResult
	if err = json.Unmarshal([]byte(envelope.Content[0].Text), &result); err != nil {
		harness.t.Fatalf("Unmarshal returned an error: %v: %s", err, rawResponse)
	}
	return result
}
