//go:build offlinelive

package offlinelive

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

const (
	retainedDecisionsContent   = "## PURPOSE\nSynthetic retained decision record\n"
	olderDecisionsContent      = "## PURPOSE\nSynthetic older decision record\n"
	retainedDecisionsDirectory = "retained-decisions"
	decisionToolName           = "manage_adr"
	decisionStatusUpdated      = "updated"
	decisionChildDirectory     = "child"
	decisionParentSourceName   = "root.go"
	decisionParentSource       = "package parent\n\nfunc ParentRoot() string {\n\treturn \"parent\"\n}\n"
	decisionFilePermission     = 0o644
	decisionDamageScript       = "../../internal/cbm/testdata/decisions_damage_row.sql"
	decisionHeaderBytes        = 100
	decisionDamageByte         = 0xA5
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

// TestClearIndexRestoresNewestDecisions seeds an older retained record for an
// indexed path, calls ClearIndex, and indexes the path again. It checks that
// the index restores the newer record and that one remaining retained file
// contains the older record.
func TestClearIndexRestoresNewestDecisions(t *testing.T) {
	harness := newHarness(t)
	requireCompleted(t, harness.indexFixture())
	harness.updateDecisions()
	canonicalPath := harness.indexStatus().GetCodebase().GetCanonicalPath()
	digest := sha256.Sum256([]byte(canonicalPath))
	retainedDirectory := filepath.Join(harness.config.GraphDir, retainedDecisionsDirectory)
	if err := os.MkdirAll(retainedDirectory, 0o755); err != nil {
		t.Fatalf("MkdirAll returned an error: %v", err)
	}
	seededPath := filepath.Join(retainedDirectory, hex.EncodeToString(digest[:])+".adr")
	if err := os.WriteFile(seededPath, []byte(olderDecisionsContent), decisionFilePermission); err != nil {
		t.Fatalf("WriteFile returned an error: %v", err)
	}

	harness.clearIndex()
	requireCompleted(t, harness.indexFixture())

	if got := harness.readDecisions(); got != retainedDecisionsContent {
		t.Fatalf("The decision record after ClearIndex and a new index is %q instead of %q", got, retainedDecisionsContent)
	}
	if _, statErr := os.Stat(seededPath); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("Stat(%q) returned %v instead of os.ErrNotExist", seededPath, statErr)
	}
	retainedFiles, err := os.ReadDir(retainedDirectory)
	if err != nil {
		t.Fatalf("ReadDir returned an error: %v", err)
	}
	if len(retainedFiles) != 1 {
		t.Fatalf("The retained decisions directory contains %d files instead of 1", len(retainedFiles))
	}
	olderRecord, err := os.ReadFile(filepath.Join(retainedDirectory, retainedFiles[0].Name()))
	if err != nil {
		t.Fatalf("ReadFile returned an error: %v", err)
	}
	if string(olderRecord) != olderDecisionsContent {
		t.Fatalf("The second retained file contains %q instead of %q", olderRecord, olderDecisionsContent)
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

// TestClearIndexKeepsUnreadableDatabase checks that ClearIndex backs up a damaged
// database when neither the database nor the decision file provides a readable
// record. It checks that the backup or its WAL file contains the record and that
// an earlier backup remains unchanged. When the decision file or cached engine
// provides the record, ClearIndex retains it without creating a backup.
func TestClearIndexKeepsUnreadableDatabase(t *testing.T) {
	testCases := []struct {
		name             string
		damage           func(t *testing.T, databasePath string)
		keepDecisionFile bool
		keepCachedEngine bool
		earlierBackup    bool
		wantBackupSuffix string
	}{
		{name: "A damaged decision table", damage: damageDecisionTable, wantBackupSuffix: ".corrupt"},
		{name: "A damaged file header", damage: damageDatabaseHeader, wantBackupSuffix: ".corrupt"},
		{name: "An earlier backup", damage: damageDatabaseHeader, earlierBackup: true, wantBackupSuffix: ".corrupt.1"},
		{name: "A decision file", damage: damageDatabaseHeader, keepDecisionFile: true},
		{name: "A cached engine", damage: damageDecisionTable, keepCachedEngine: true},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			harness := newHarness(t)
			requireCompleted(t, harness.indexFixture())
			harness.updateDecisions()
			if got := harness.readDecisions(); got != retainedDecisionsContent {
				t.Fatalf("The stored decision record is %q instead of %q", got, retainedDecisionsContent)
			}
			codebase := harness.indexStatus().GetCodebase()
			databasePath := filepath.Join(harness.config.GraphDir, codebase.GetId()+".db")
			if !testCase.keepDecisionFile {
				if err := os.Remove(databasePath + ".adr"); err != nil {
					t.Fatalf("Remove returned an error: %v", err)
				}
			}
			if testCase.earlierBackup {
				if err := os.WriteFile(databasePath+".corrupt", []byte(olderDecisionsContent), decisionFilePermission); err != nil {
					t.Fatalf("WriteFile returned an error: %v", err)
				}
			}
			if !testCase.keepCachedEngine {
				harness.restart(harnessOptions{})
			}
			testCase.damage(t, databasePath)

			harness.clearIndex()

			if _, statErr := os.Stat(databasePath); !errors.Is(statErr, os.ErrNotExist) {
				t.Fatalf("Stat(%q) returned %v instead of os.ErrNotExist", databasePath, statErr)
			}
			digest := sha256.Sum256([]byte(codebase.GetCanonicalPath()))
			retainedPath := filepath.Join(harness.config.GraphDir, retainedDecisionsDirectory, hex.EncodeToString(digest[:])+".adr")
			retained, retainedErr := os.ReadFile(retainedPath)
			if testCase.keepDecisionFile || testCase.keepCachedEngine {
				if retainedErr != nil || string(retained) != retainedDecisionsContent {
					t.Fatalf("ReadFile returned record %q and error %v. The expected record is %q", retained, retainedErr, retainedDecisionsContent)
				}
				if _, statErr := os.Stat(databasePath + ".corrupt"); !errors.Is(statErr, os.ErrNotExist) {
					t.Fatalf("Stat of the backup returned %v instead of os.ErrNotExist", statErr)
				}
				return
			}
			if !errors.Is(retainedErr, os.ErrNotExist) {
				t.Fatalf("ReadFile of the retained record returned %v instead of os.ErrNotExist", retainedErr)
			}
			backup, err := os.ReadFile(databasePath + testCase.wantBackupSuffix)
			if err != nil {
				t.Fatalf("ReadFile of the backup returned an error: %v", err)
			}
			backupWAL, walErr := os.ReadFile(databasePath + testCase.wantBackupSuffix + "-wal")
			if walErr != nil && !errors.Is(walErr, os.ErrNotExist) {
				t.Fatalf("ReadFile of the backup WAL file returned an error: %v", walErr)
			}
			record := []byte(retainedDecisionsContent)
			if !bytes.Contains(backup, record) && !bytes.Contains(backupWAL, record) {
				t.Fatal("The backup and its WAL file do not contain the decision record")
			}
			if testCase.earlierBackup {
				earlier, readErr := os.ReadFile(databasePath + ".corrupt")
				if readErr != nil || string(earlier) != olderDecisionsContent {
					t.Fatalf("ReadFile returned earlier backup %q and error %v. The expected content is %q", earlier, readErr, olderDecisionsContent)
				}
			}
		})
	}
}

func damageDatabaseHeader(t *testing.T, databasePath string) {
	t.Helper()
	data, err := os.ReadFile(databasePath)
	if err != nil {
		t.Fatalf("ReadFile returned an error: %v", err)
	}
	for i := range decisionHeaderBytes {
		data[i] = decisionDamageByte
	}
	if err = os.WriteFile(databasePath, data, decisionFilePermission); err != nil {
		t.Fatalf("WriteFile returned an error: %v", err)
	}
}

func damageDecisionTable(t *testing.T, databasePath string) {
	t.Helper()
	script, err := os.ReadFile(decisionDamageScript)
	if err != nil {
		t.Fatalf("ReadFile returned an error: %v", err)
	}
	command := exec.CommandContext(context.Background(), "sqlite3", databasePath)
	command.Stdin = bytes.NewReader(script)
	if output, runErr := command.CombinedOutput(); runErr != nil {
		t.Fatalf("sqlite3 returned an error: %v: %s", runErr, output)
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
