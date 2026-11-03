package cbm_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"goodkind.io/lm-semantic-search/internal/cbm"
)

const (
	decisionsProject          = "decisions"
	decisionsContent          = "## PURPOSE\nSynthetic \"decision\" record\n"
	decisionsReplacement      = "## PURPOSE\nSynthetic replacement record\n"
	decisionsInvalidDatabase  = "invalid database"
	decisionsSecondInvalid    = "second invalid database"
	decisionsBackupSuffix     = ".corrupt"
	decisionsOtherProject     = "decisions-other"
	decisionsHoldWALSQL       = "testdata/decisions_hold_wal.sql"
	decisionsHolderReady      = "holder-ready"
	decisionsHolderWait       = 10 * time.Second
	decisionsHolderPoll       = 10 * time.Millisecond
	decisionsWALContent       = "Synthetic WAL decision record"
	decisionsDamageRowSQL     = "testdata/decisions_damage_row.sql"
	decisionsFailIntegritySQL = "testdata/decisions_fail_integrity.sql"
	decisionsSelectRowSQL     = "testdata/decisions_select_row.sql"
	decisionsDropSummarySQL   = "testdata/decisions_drop_summary.sql"
	decisionsBlockRestoreSQL  = "testdata/decisions_block_restore.sql"
	decisionsRemoveRowSQL     = "testdata/decisions_remove_row.sql"
	decisionsBreakLoadSQL     = "testdata/decisions_break_load.sql"
	decisionsFunctionsQuery   = "testdata/decisions_functions.cypher"
	decisionsStatusUpdated    = "updated"
	decisionsStatusRejected   = "write_error"
	decisionsAlphaSource      = "package repo\n\nfunc AlphaOne() string {\n\treturn \"alpha\"\n}\n"
	decisionsBetaSource       = "package repo\n\nfunc BetaOne() string {\n\treturn \"beta\"\n}\n"
	decisionsChangedSource    = "package repo\n\nfunc AlphaChanged() string {\n\treturn \"changed\"\n}\n"
	decisionsGammaSource      = "package repo\n\nfunc GammaOne() string {\n\treturn \"gamma\"\n}\n"
	decisionsDeltaSource      = "package repo\n\nfunc DeltaOne() string {\n\treturn \"delta\"\n}\n"
	decisionsFilePermission   = 0o644
)

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

// TestEngineKeepsEachInvalidDatabase checks that successive Index calls save
// two invalid databases in separate backups without replacing the first backup.
func TestEngineKeepsEachInvalidDatabase(t *testing.T) {
	fixture := newDecisionFixture(t)
	writeFile(t, fixture.databasePath, decisionsInvalidDatabase)
	indexDecisionFixture(t, fixture)
	writeFile(t, fixture.databasePath, decisionsSecondInvalid)
	indexDecisionFixture(t, fixture)

	wantBackups := map[string]string{
		fixture.databasePath + decisionsBackupSuffix:        decisionsInvalidDatabase,
		fixture.databasePath + decisionsBackupSuffix + ".1": decisionsSecondInvalid,
	}
	for path, want := range wantBackups {
		backup, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile returned an error: %v", err)
		}
		if string(backup) != want {
			t.Fatalf("The backup %s contains %q instead of %q", path, backup, want)
		}
	}
}

// TestEngineBackupKeepsWALAfterIntegrityFailure checks that manage_adr backs up
// the database and its WAL file after an integrity check fails. It uses sqlite3
// to read the updated decision record from the new backup and checks that the
// earlier backup remains unchanged.
func TestEngineBackupKeepsWALAfterIntegrityFailure(t *testing.T) {
	fixture := newDecisionFixture(t)
	earlierBackupPath := fixture.databasePath + decisionsBackupSuffix
	writeFile(t, earlierBackupPath, decisionsInvalidDatabase)
	removeFile(t, fixture.decisionsPath)
	runSQLiteFile(t, fixture.databasePath, decisionsFailIntegritySQL)
	walInfo, err := os.Stat(fixture.databasePath + "-wal")
	if err != nil {
		t.Fatalf("Stat of the WAL file returned an error: %v", err)
	}
	if walInfo.Size() == 0 {
		t.Fatal("The WAL file is empty before the read")
	}
	stopHolder := holdDatabaseConnection(t, fixture.databasePath)
	defer stopHolder()

	// manage_adr opens the requested project's database and checks its integrity when
	// the request specifies a different project from the engine's current project.
	engine, err := cbm.Open(decisionsOtherProject, fixture.cacheDir)
	if err != nil {
		t.Fatalf("Open returned an error: %v", err)
	}
	defer engine.Close()
	arguments, err := json.Marshal(map[string]string{"project": decisionsProject, "mode": "get"})
	if err != nil {
		t.Fatalf("Marshal returned an error: %v", err)
	}
	rawResponse, err := engine.Tool("manage_adr", string(arguments))
	if err != nil {
		t.Fatalf("manage_adr returned an error: %v", err)
	}
	var envelope struct {
		IsError bool `json:"isError"`
	}
	if err = json.Unmarshal([]byte(rawResponse), &envelope); err != nil {
		t.Fatalf("The test could not decode the manage_adr response: %v: %s", err, rawResponse)
	}
	if !envelope.IsError {
		t.Fatalf("manage_adr did not report an error for the failed integrity check: %s", rawResponse)
	}

	if _, statErr := os.Stat(fixture.databasePath); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("Stat of the database returned %v instead of os.ErrNotExist", statErr)
	}
	backupWAL, err := os.Stat(earlierBackupPath + ".1-wal")
	if err != nil {
		t.Fatalf("Stat of the backup WAL file returned an error: %v", err)
	}
	if backupWAL.Size() == 0 {
		t.Fatal("The backup WAL file is empty")
	}
	if _, statErr := os.Stat(fixture.databasePath + "-wal"); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("Stat of the original WAL file returned %v instead of os.ErrNotExist", statErr)
	}
	stopHolder()
	earlier, err := os.ReadFile(earlierBackupPath)
	if err != nil {
		t.Fatalf("ReadFile of the earlier backup returned an error: %v", err)
	}
	if string(earlier) != decisionsInvalidDatabase {
		t.Fatalf("The earlier backup contains %q instead of %q", earlier, decisionsInvalidDatabase)
	}
	script, err := os.ReadFile(decisionsSelectRowSQL)
	if err != nil {
		t.Fatalf("ReadFile of the select script returned an error: %v", err)
	}
	stored := strings.TrimSpace(string(runSQLite(t, earlierBackupPath+".1", script)))
	if stored != decisionsWALContent {
		t.Fatalf("The new backup stores %q instead of %q", stored, decisionsWALContent)
	}
}

// TestEngineReportsUnreadableDecisionRow damages the decision table and checks
// that ReadDecisions returns an error wrapping ErrDecisionsUnreadable.
func TestEngineReportsUnreadableDecisionRow(t *testing.T) {
	fixture := newDecisionFixture(t)
	removeFile(t, fixture.decisionsPath)
	runSQLiteFile(t, fixture.databasePath, decisionsDamageRowSQL)

	engine := openDecisionEngine(t, fixture)
	defer engine.Close()
	if _, err := engine.ReadDecisions(); !errors.Is(err, cbm.ErrDecisionsUnreadable) {
		t.Fatalf("ReadDecisions returned %v instead of %v", err, cbm.ErrDecisionsUnreadable)
	}
}

// TestEngineReportsUnreadableDatabase replaces the database with invalid data
// and checks that ReadDecisions returns an error wrapping ErrDecisionsUnreadable.
func TestEngineReportsUnreadableDatabase(t *testing.T) {
	fixture := newDecisionFixture(t)
	writeFile(t, fixture.databasePath, decisionsInvalidDatabase)

	engine := openDecisionEngine(t, fixture)
	defer engine.Close()
	if _, err := engine.ReadDecisions(); !errors.Is(err, cbm.ErrDecisionsUnreadable) {
		t.Fatalf("ReadDecisions returned %v instead of %v", err, cbm.ErrDecisionsUnreadable)
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

// TestEngineKeepsGraphWhenDecisionReadFails removes the decision column from a database
// that passes its integrity check. It verifies that Index reports an error and
// keeps the existing graph.
func TestEngineKeepsGraphWhenDecisionReadFails(t *testing.T) {
	testCases := []struct {
		name    string
		disrupt func(t *testing.T, fixture decisionFixture)
	}{
		{name: "A full index", disrupt: addSourcesForFullIndex},
		{name: "An incremental index", disrupt: changeAlphaSource},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			fixture := newDecisionFixture(t)
			runSQLiteFile(t, fixture.databasePath, decisionsDropSummarySQL)
			testCase.disrupt(t, fixture)

			engine := openDecisionEngine(t, fixture)
			defer engine.Close()
			if err := engine.Index(context.Background(), fixture.repositoryPath, "fast"); err == nil {
				t.Fatal("Index returned nil instead of an error for the failed decision read")
			}
			functionNames := queryColumn(t, engine, decisionsProject, decisionsFunctionsQuery)
			if !slices.Contains(functionNames, "AlphaOne") {
				t.Fatalf("The graph does not contain AlphaOne after the failed Index: %q", functionNames)
			}
		})
	}
}

// TestEngineIndexReportsFailedDecisionRestore changes source files and runs a full
// index and an incremental index. Index does not dump the database without a source
// change. The test places a directory at the database's shared memory path. The dump
// can write the database, but SQLite cannot store the decision row. The test verifies
// that Index reports an error and the decision file keeps the record.
func TestEngineIndexReportsFailedDecisionRestore(t *testing.T) {
	testCases := []struct {
		name    string
		disrupt func(t *testing.T, fixture decisionFixture)
	}{
		{name: "A full index", disrupt: addSourcesForFullIndex},
		{name: "An incremental index", disrupt: changeAlphaSource},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			fixture := newDecisionFixture(t)
			sharedMemoryPath := fixture.databasePath + "-shm"
			if err := os.RemoveAll(sharedMemoryPath); err != nil {
				t.Fatalf("RemoveAll returned an error: %v", err)
			}
			if err := os.Mkdir(sharedMemoryPath, 0o755); err != nil {
				t.Fatalf("Mkdir returned an error: %v", err)
			}
			writeFile(t, filepath.Join(sharedMemoryPath, "entry"), "blocks unlink")
			testCase.disrupt(t, fixture)

			engine := openDecisionEngine(t, fixture)
			defer engine.Close()
			if err := engine.Index(context.Background(), fixture.repositoryPath, "fast"); err == nil {
				t.Fatal("Index returned nil instead of an error for the failed decision restore")
			}
			saved, readErr := os.ReadFile(fixture.decisionsPath)
			if readErr != nil {
				t.Fatalf("ReadFile returned an error: %v", readErr)
			}
			if string(saved) != decisionsContent {
				t.Fatalf("The decision file contains %q instead of %q", saved, decisionsContent)
			}
		})
	}
}

// TestEngineReportsFailedDecisionRestore blocks the row that manage_adr restores
// from the decision file. It verifies that ReadDecisions reports an error.
func TestEngineReportsFailedDecisionRestore(t *testing.T) {
	fixture := newDecisionFixture(t)
	runSQLiteFile(t, fixture.databasePath, decisionsBlockRestoreSQL)

	engine := openDecisionEngine(t, fixture)
	defer engine.Close()
	if _, err := engine.ReadDecisions(); !errors.Is(err, cbm.ErrDecisionsUnreadable) {
		t.Fatalf("ReadDecisions returned %v instead of %v", err, cbm.ErrDecisionsUnreadable)
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

// addSourcesForFullIndex adds enough source files for Index to select a full index.
func addSourcesForFullIndex(t *testing.T, fixture decisionFixture) {
	t.Helper()
	writeFile(t, filepath.Join(fixture.repositoryPath, "gamma.go"), decisionsGammaSource)
	writeFile(t, filepath.Join(fixture.repositoryPath, "delta.go"), decisionsDeltaSource)
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

// holdDatabaseConnection opens a second SQLite connection in a sqlite3 process.
// This connection prevents SQLite from deleting the WAL file when the engine
// closes its connection. The returned function kills sqlite3 without copying
// pending WAL changes into the database.
func holdDatabaseConnection(t *testing.T, databasePath string) func() {
	t.Helper()
	script, err := os.ReadFile(decisionsHoldWALSQL)
	if err != nil {
		t.Fatalf("holdDatabaseConnection could not read the script: %v", err)
	}
	directory := t.TempDir()
	holder := exec.CommandContext(context.Background(), "sqlite3", databasePath)
	holder.Dir = directory
	stdin, err := holder.StdinPipe()
	if err != nil {
		t.Fatalf("StdinPipe returned an error: %v", err)
	}
	if err = holder.Start(); err != nil {
		t.Fatalf("Start returned an error: %v", err)
	}
	stopped := false
	stop := func() {
		if stopped {
			return
		}
		stopped = true
		_ = holder.Process.Kill()
		_ = holder.Wait()
		_ = stdin.Close()
	}
	if _, err = stdin.Write(script); err != nil {
		stop()
		t.Fatalf("Write returned an error: %v", err)
	}
	readyPath := filepath.Join(directory, decisionsHolderReady)
	deadline := time.Now().Add(decisionsHolderWait)
	for time.Now().Before(deadline) {
		if info, statErr := os.Stat(readyPath); statErr == nil && info.Size() > 0 {
			return stop
		}
		time.Sleep(decisionsHolderPoll)
	}
	stop()
	t.Fatal("The second SQLite connection did not open before the limit")
	return stop
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
