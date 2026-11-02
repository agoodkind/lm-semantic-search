package cbm_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/cbm"
)

const (
	decisionsProject         = "decisions"
	decisionsContent         = "## PURPOSE\nSynthetic \"decision\" record\n"
	decisionsReplacement     = "## PURPOSE\nSynthetic replacement record\n"
	decisionsInvalidDatabase = "invalid database"
	decisionsBackupSuffix    = ".corrupt"
	decisionsRemoveRowSQL    = "testdata/decisions_remove_row.sql"
	decisionsBreakLoadSQL    = "testdata/decisions_break_load.sql"
	decisionsFunctionsQuery  = "testdata/decisions_functions.cypher"
	decisionsStatusUpdated   = "updated"
	decisionsStatusRejected  = "write_error"
	decisionsAlphaSource     = "package repo\n\nfunc AlphaOne() string {\n\treturn \"alpha\"\n}\n"
	decisionsBetaSource      = "package repo\n\nfunc BetaOne() string {\n\treturn \"beta\"\n}\n"
	decisionsChangedSource   = "package repo\n\nfunc AlphaChanged() string {\n\treturn \"changed\"\n}\n"
	decisionsFilePermission  = 0o644
)

// decisionUpdateEnvelope stores the manage_adr fields that the tests read.
type decisionUpdateEnvelope struct {
	Content []struct {
		Text string `json:"text"`
	} `json:"content"`
	IsError bool `json:"isError"`
}

// decisionFixture stores the paths of one indexed synthetic repository.
type decisionFixture struct {
	repositoryPath string
	cacheDir       string
	databasePath   string
	decisionsPath  string
}

// TestEngineKeepsDecisionsAcrossIndexFailures stores a decision record through
// manage_adr, applies one disruption, and runs Index with a new engine handle.
// It opens another engine handle and reads the record through manage_adr.
func TestEngineKeepsDecisionsAcrossIndexFailures(t *testing.T) {
	testCases := []struct {
		name    string
		disrupt func(t *testing.T, fixture decisionFixture)
	}{
		{
			name: "A source file changes",
			disrupt: func(t *testing.T, fixture decisionFixture) {
				t.Helper()
				changeAlphaSource(t, fixture)
			},
		},
		{
			name: "The test deletes a source file",
			disrupt: func(t *testing.T, fixture decisionFixture) {
				t.Helper()
				removeFile(t, filepath.Join(fixture.repositoryPath, "beta.go"))
			},
		},
		{
			name: "The decision file does not exist for a stored record",
			disrupt: func(t *testing.T, fixture decisionFixture) {
				t.Helper()
				removeFile(t, fixture.decisionsPath)
				changeAlphaSource(t, fixture)
			},
		},
		{
			name: "The test removes the database to simulate a stopped process",
			disrupt: func(t *testing.T, fixture decisionFixture) {
				t.Helper()
				removeFile(t, fixture.databasePath)
			},
		},
		{
			name: "The database contains invalid data",
			disrupt: func(t *testing.T, fixture decisionFixture) {
				t.Helper()
				writeFile(t, fixture.databasePath, decisionsInvalidDatabase)
			},
		},
		{
			name: "The incremental load fails",
			disrupt: func(t *testing.T, fixture decisionFixture) {
				t.Helper()
				runSQLiteFile(t, fixture.databasePath, decisionsBreakLoadSQL)
				changeAlphaSource(t, fixture)
			},
		},
		{
			name: "The decision row does not exist before an incremental index",
			disrupt: func(t *testing.T, fixture decisionFixture) {
				t.Helper()
				runSQLiteFile(t, fixture.databasePath, decisionsRemoveRowSQL)
				changeAlphaSource(t, fixture)
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			fixture := newDecisionFixture(t)
			testCase.disrupt(t, fixture)

			indexDecisionFixture(t, fixture)

			if got := readDecisionsWithNewEngine(t, fixture); got != decisionsContent {
				t.Fatalf("The decision record after Index is %q instead of %q", got, decisionsContent)
			}
		})
	}
}

// TestEngineKeepsInvalidDatabaseBeforeRebuild replaces the database with invalid
// data and runs Index. It verifies that the renamed database contains the same
// bytes.
func TestEngineKeepsInvalidDatabaseBeforeRebuild(t *testing.T) {
	fixture := newDecisionFixture(t)
	writeFile(t, fixture.databasePath, decisionsInvalidDatabase)

	indexDecisionFixture(t, fixture)

	backup, err := os.ReadFile(fixture.databasePath + decisionsBackupSuffix)
	if err != nil {
		t.Fatalf("ReadFile returned an error: %v", err)
	}
	if string(backup) != decisionsInvalidDatabase {
		t.Fatalf("The database backup contains %q instead of %q", backup, decisionsInvalidDatabase)
	}
}

// TestEngineReadsDecisionFileWhenRowIsMissing removes the decision row without
// running Index and checks that manage_adr restores the row from the decision
// file.
func TestEngineReadsDecisionFileWhenRowIsMissing(t *testing.T) {
	fixture := newDecisionFixture(t)
	runSQLiteFile(t, fixture.databasePath, decisionsRemoveRowSQL)

	if got := readDecisionsWithNewEngine(t, fixture); got != decisionsContent {
		t.Fatalf("The first read returned %q instead of %q", got, decisionsContent)
	}
	removeFile(t, fixture.decisionsPath)
	if got := readDecisionsWithNewEngine(t, fixture); got != decisionsContent {
		t.Fatalf("The read after removal of the decision file returned %q instead of %q", got, decisionsContent)
	}
}

// TestEngineKeepsGraphWhenDecisionFileCannotBeWritten replaces the decision file
// with a directory. Index must report an error and keep the old graph and record.
func TestEngineKeepsGraphWhenDecisionFileCannotBeWritten(t *testing.T) {
	fixture := newDecisionFixture(t)
	blockDecisionFile(t, fixture)
	changeAlphaSource(t, fixture)

	engine := openDecisionEngine(t, fixture)
	defer engine.Close()
	if err := engine.Index(context.Background(), fixture.repositoryPath, "fast"); err == nil {
		t.Fatal("Index returned nil instead of an error for the failed decision file write")
	}

	if got := readDecisionsWithNewEngine(t, fixture); got != decisionsContent {
		t.Fatalf("The decision record after the failed Index is %q instead of %q", got, decisionsContent)
	}
	functionNames := queryColumn(t, engine, decisionsProject, decisionsFunctionsQuery)
	if slices.Contains(functionNames, "AlphaChanged") {
		t.Fatalf("The graph contains AlphaChanged after the failed Index: %q", functionNames)
	}
	if !slices.Contains(functionNames, "AlphaOne") {
		t.Fatalf("The graph does not contain AlphaOne after the failed Index: %q", functionNames)
	}
}

// TestEngineRejectsUpdateWhenDecisionFileCannotBeWritten checks that manage_adr
// reports write_error and keeps the old record when the file write fails.
func TestEngineRejectsUpdateWhenDecisionFileCannotBeWritten(t *testing.T) {
	fixture := newDecisionFixture(t)
	blockDecisionFile(t, fixture)

	engine := openDecisionEngine(t, fixture)
	defer engine.Close()
	if status := updateDecisions(t, engine, decisionsReplacement); status != decisionsStatusRejected {
		t.Fatalf("manage_adr returned status %q instead of %q", status, decisionsStatusRejected)
	}

	if got := readDecisionsWithNewEngine(t, fixture); got != decisionsContent {
		t.Fatalf("The decision record after the rejected update is %q instead of %q", got, decisionsContent)
	}
}

// newDecisionFixture indexes a synthetic repository and stores one decision
// record.
func newDecisionFixture(t *testing.T) decisionFixture {
	t.Helper()
	repositoryPath := t.TempDir()
	writeFile(t, filepath.Join(repositoryPath, "alpha.go"), decisionsAlphaSource)
	writeFile(t, filepath.Join(repositoryPath, "beta.go"), decisionsBetaSource)

	cacheDir := t.TempDir()
	databasePath := filepath.Join(cacheDir, decisionsProject+".db")
	fixture := decisionFixture{
		repositoryPath: repositoryPath,
		cacheDir:       cacheDir,
		databasePath:   databasePath,
		decisionsPath:  databasePath + ".adr",
	}
	indexDecisionFixture(t, fixture)

	engine := openDecisionEngine(t, fixture)
	defer engine.Close()
	if status := updateDecisions(t, engine, decisionsContent); status != decisionsStatusUpdated {
		t.Fatalf("manage_adr returned status %q instead of %q", status, decisionsStatusUpdated)
	}
	return fixture
}

func openDecisionEngine(t *testing.T, fixture decisionFixture) *cbm.Engine {
	t.Helper()
	engine, err := cbm.Open(decisionsProject, fixture.cacheDir)
	if err != nil {
		t.Fatalf("Open returned an error: %v", err)
	}
	return engine
}

func indexDecisionFixture(t *testing.T, fixture decisionFixture) {
	t.Helper()
	engine := openDecisionEngine(t, fixture)
	defer engine.Close()
	if err := engine.Index(context.Background(), fixture.repositoryPath, "fast"); err != nil {
		t.Fatalf("Index returned an error: %v", err)
	}
}

func readDecisionsWithNewEngine(t *testing.T, fixture decisionFixture) string {
	t.Helper()
	engine := openDecisionEngine(t, fixture)
	defer engine.Close()
	content, err := engine.ReadDecisions()
	if err != nil {
		t.Fatalf("ReadDecisions returned an error: %v", err)
	}
	return content
}

// updateDecisions stores content through manage_adr and returns the reported
// status.
func updateDecisions(t *testing.T, engine *cbm.Engine, content string) string {
	t.Helper()
	arguments, err := json.Marshal(map[string]string{
		"project": decisionsProject,
		"mode":    "update",
		"content": content,
	})
	if err != nil {
		t.Fatalf("updateDecisions could not encode manage_adr arguments as JSON: %v", err)
	}
	rawResponse, err := engine.Tool("manage_adr", string(arguments))
	if err != nil {
		t.Fatalf("manage_adr returned an error: %v", err)
	}
	var envelope decisionUpdateEnvelope
	if err = json.Unmarshal([]byte(rawResponse), &envelope); err != nil {
		t.Fatalf("updateDecisions could not decode the manage_adr response: %v: %s", err, rawResponse)
	}
	if len(envelope.Content) != 1 {
		t.Fatalf("manage_adr returned %d content items instead of 1: %s", len(envelope.Content), rawResponse)
	}
	var result struct {
		Status string `json:"status"`
	}
	if err = json.Unmarshal([]byte(envelope.Content[0].Text), &result); err != nil {
		t.Fatalf("updateDecisions could not decode the manage_adr result: %v: %s", err, rawResponse)
	}
	return result.Status
}

func changeAlphaSource(t *testing.T, fixture decisionFixture) {
	t.Helper()
	writeFile(t, filepath.Join(fixture.repositoryPath, "alpha.go"), decisionsChangedSource)
}

// blockDecisionFile replaces the decision file with a directory that rename
// cannot replace.
func blockDecisionFile(t *testing.T, fixture decisionFixture) {
	t.Helper()
	removeFile(t, fixture.decisionsPath)
	if err := os.Mkdir(fixture.decisionsPath, 0o755); err != nil {
		t.Fatalf("blockDecisionFile could not create a directory at %s: %v", fixture.decisionsPath, err)
	}
	writeFile(t, filepath.Join(fixture.decisionsPath, "entry"), "blocks rename")
}

func runSQLiteFile(t *testing.T, databasePath string, scriptPath string) {
	t.Helper()
	script, err := os.ReadFile(scriptPath)
	if err != nil {
		t.Fatalf("runSQLiteFile could not read %s: %v", scriptPath, err)
	}
	if output := strings.TrimSpace(string(runSQLite(t, databasePath, script))); output != "" {
		t.Fatalf("The SQL script %s printed %q instead of empty output", scriptPath, output)
	}
}

func writeFile(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), decisionsFilePermission); err != nil {
		t.Fatalf("writeFile could not write %s: %v", path, err)
	}
}

func removeFile(t *testing.T, path string) {
	t.Helper()
	if err := os.Remove(path); err != nil {
		t.Fatalf("removeFile could not remove %s: %v", path, err)
	}
}
