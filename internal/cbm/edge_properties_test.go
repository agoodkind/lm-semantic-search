package cbm_test

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/cbm"
)

const (
	edgePropertiesProject    = "edgeprops"
	edgePropertiesQuery      = "testdata/edge_properties.sql"
	edgePropertiesRepository = "testdata/edge_properties_repo"
	edgePropertiesUpdate     = "testdata/edge_properties_update"
	asyncCallsRecordType     = "ASYNC_CALLS"
	importsRecordType        = "IMPORTS"
	routeRecordType          = "Route"
	wantBroker               = "kafka"
	wantMethod               = "GET"
	wantAsyncTopic           = "error"
	wantQuotedLocalName      = `inc")`
	wantMultilineLocalName   = "text(\"first\nsecond\")"
	longLocalNameBytes       = 300
	edgePropertiesLongCall   = "testdata/edge_properties_repo/kafka_long.go"
	longCalleeMinimumBytes   = 200
	longTopicMinimumBytes    = 256
	importTargetsQuery       = "testdata/import_targets.cypher"
	escapedImportProject     = "escapedimport"
	escapedImportRepository  = "testdata/escaped_import_repo"
	escapedImportUpdate      = "testdata/escaped_import_update/main.js"
	escapedImportEdgeSQL     = "testdata/escaped_import_edge.sql"
	escapedImportCallsQuery  = "testdata/escaped_import_calls.cypher"
	escapedImportWantTarget  = "escapedimport.settings.helper"
	escapedImportOtherTarget = "escapedimport.unrelated.helper"
)

var (
	longCallPattern   = regexp.MustCompile(`(?m)^\t(\w+)\("([^"]+)"\)$`)
	wantImportTargets = []string{"other", "helper", "longname"}
)

// queryGraphResponse stores the query_graph fields that the test reads.
type queryGraphResponse struct {
	StructuredContent struct {
		Rows []json.RawMessage `json:"rows"`
	} `json:"structuredContent"`
	IsError bool `json:"isError"`
}

// graphPropertyRow is one row from edge_properties.sql.
type graphPropertyRow struct {
	RecordType    string `json:"record_type"`
	PropertyValue string `json:"property_value"`
}

// graphProperties stores the property fields that the test checks.
type graphProperties struct {
	Callee    string `json:"callee"`
	Broker    string `json:"broker"`
	Method    string `json:"method"`
	URLPath   string `json:"url_path"`
	LocalName string `json:"local_name"`
}

// TestEngineWritesValidGraphProperties indexes synthetic Go, Python, and PHP
// files through the engine. It changes a Python file and a Go file, then runs an
// incremental index. It reads properties with sqlite3 and checks valid JSON,
// complete long callee and url_path values, and IMPORTS local_name values.
// It checks Route broker and method values separately and IMPORTS targets with
// query_graph.
func TestEngineWritesValidGraphProperties(t *testing.T) {
	repositoryPath := filepath.Join(t.TempDir(), "repository")
	if err := os.CopyFS(repositoryPath, os.DirFS(edgePropertiesRepository)); err != nil {
		t.Fatalf("copy %s: %v", edgePropertiesRepository, err)
	}

	cacheDir := t.TempDir()
	engine, err := cbm.Open(edgePropertiesProject, cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	if err = engine.Index(context.Background(), repositoryPath, "fast"); err != nil {
		t.Fatalf("Index returned error: %v", err)
	}

	for _, name := range []string{"http_caller.py", "telemetry.go"} {
		updatedSource, readErr := os.ReadFile(filepath.Join(edgePropertiesUpdate, name))
		if readErr != nil {
			t.Fatalf("read updated source %s: %v", name, readErr)
		}
		if writeErr := os.WriteFile(filepath.Join(repositoryPath, name), updatedSource, 0o644); writeErr != nil {
			t.Fatalf("write updated source %s: %v", name, writeErr)
		}
	}
	if err = engine.Index(context.Background(), repositoryPath, "fast"); err != nil {
		t.Fatalf("incremental Index returned error: %v", err)
	}

	wantLongCallee, wantLongTopic := readLongCall(t)
	var asyncCallees, asyncTopics, edgeBrokers, routeBrokers, localNames, methods []string
	for _, row := range readGraphProperties(t, cacheDir) {
		var properties graphProperties
		if decodeErr := json.Unmarshal([]byte(row.PropertyValue), &properties); decodeErr != nil {
			t.Errorf("%s row has invalid properties JSON %q: %v", row.RecordType, row.PropertyValue, decodeErr)
			continue
		}
		switch row.RecordType {
		case asyncCallsRecordType:
			asyncCallees = append(asyncCallees, properties.Callee)
			asyncTopics = append(asyncTopics, properties.URLPath)
			edgeBrokers = append(edgeBrokers, properties.Broker)
		case importsRecordType:
			localNames = append(localNames, properties.LocalName)
		case routeRecordType:
			routeBrokers = append(routeBrokers, properties.Broker)
			methods = append(methods, properties.Method)
		}
	}
	for _, wantTopic := range []string{wantAsyncTopic, wantLongTopic} {
		if !slices.Contains(asyncTopics, wantTopic) {
			t.Errorf("ASYNC_CALLS url_path values are %q; want %q", asyncTopics, wantTopic)
		}
	}
	if !slices.Contains(asyncCallees, wantLongCallee) {
		t.Errorf("ASYNC_CALLS callee values are %q; want %q", asyncCallees, wantLongCallee)
	}
	if !slices.Contains(edgeBrokers, wantBroker) {
		t.Errorf("ASYNC_CALLS broker values are %q; want %q", edgeBrokers, wantBroker)
	}
	if !slices.Contains(routeBrokers, wantBroker) {
		t.Errorf("Route broker values are %q; want %q", routeBrokers, wantBroker)
	}
	if !slices.Contains(methods, wantMethod) {
		t.Errorf("Route method values are %q; want %q", methods, wantMethod)
	}
	wantLongLocalName := `value("` + strings.Repeat("a", longLocalNameBytes) + `")`
	for _, wantLocalName := range []string{wantQuotedLocalName, wantMultilineLocalName, wantLongLocalName} {
		if !slices.Contains(localNames, wantLocalName) {
			t.Errorf("IMPORTS local_name values are %q; want %q", localNames, wantLocalName)
		}
	}
	importTargets := queryColumn(t, engine, edgePropertiesProject, importTargetsQuery)
	for _, wantTarget := range wantImportTargets {
		if !slices.Contains(importTargets, wantTarget) {
			t.Errorf("query_graph IMPORTS targets are %q; want %q", importTargets, wantTarget)
		}
	}
}

// readGraphProperties reads graph properties from the database in cacheDir with
// the sqlite3 command.
func readGraphProperties(t *testing.T, cacheDir string) []graphPropertyRow {
	t.Helper()
	script, err := os.ReadFile(edgePropertiesQuery)
	if err != nil {
		t.Fatalf("read %s: %v", edgePropertiesQuery, err)
	}
	output := runSQLite(t, graphDatabasePath(t, cacheDir, edgePropertiesProject), script)
	var rows []graphPropertyRow
	if err = json.Unmarshal(output, &rows); err != nil {
		t.Fatalf("decode sqlite3 output: %v: %s", err, output)
	}
	return rows
}

// graphDatabasePath returns the path to the graph database for project in
// cacheDir.
func graphDatabasePath(t *testing.T, cacheDir string, project string) string {
	t.Helper()
	matches, err := filepath.Glob(filepath.Join(cacheDir, project+"*.db"))
	if err != nil {
		t.Fatalf("find graph database: %v", err)
	}
	if len(matches) != 1 {
		t.Fatalf("found %d graph databases in %s, want 1", len(matches), cacheDir)
	}
	return matches[0]
}

// runSQLite runs script on databasePath and returns its standard output.
func runSQLite(t *testing.T, databasePath string, script []byte) []byte {
	t.Helper()
	command := exec.Command("sqlite3", "-batch", "-init", os.DevNull, databasePath)
	command.Stdin = bytes.NewReader(script)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	output, err := command.Output()
	if err != nil {
		t.Fatalf("run sqlite3 on %s: %v: %s", databasePath, err, stderr.String())
	}
	return output
}

// readLongCall returns the callee and topic from the long call in the fixture.
func readLongCall(t *testing.T) (string, string) {
	t.Helper()
	source, err := os.ReadFile(edgePropertiesLongCall)
	if err != nil {
		t.Fatalf("read %s: %v", edgePropertiesLongCall, err)
	}
	match := longCallPattern.FindSubmatch(source)
	if match == nil {
		t.Fatalf("%s does not contain the long call", edgePropertiesLongCall)
	}
	callee, topic := string(match[1]), string(match[2])
	if len(callee) < longCalleeMinimumBytes || len(topic) < longTopicMinimumBytes {
		t.Fatalf("the long call has a %d-byte callee and a %d-byte topic; want at least %d and %d bytes", len(callee), len(topic), longCalleeMinimumBytes, longTopicMinimumBytes)
	}
	return callee, topic
}

// TestEngineResolvesCallThroughEscapedImportKey adds two IMPORTS edges to a
// synthetic graph database. One edge uses a Unicode-escaped local_name for Cfg
// and targets settings.js. The other uses Other and targets unrelated.js.
// The test changes main.js, runs an incremental index, and checks that
// query_graph reports that run calls settings.helper instead of unrelated.helper.
func TestEngineResolvesCallThroughEscapedImportKey(t *testing.T) {
	repositoryPath := filepath.Join(t.TempDir(), "repository")
	if err := os.CopyFS(repositoryPath, os.DirFS(escapedImportRepository)); err != nil {
		t.Fatalf("copy %s: %v", escapedImportRepository, err)
	}
	cacheDir := t.TempDir()
	engine, err := cbm.Open(escapedImportProject, cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	if err = engine.Index(context.Background(), repositoryPath, "fast"); err != nil {
		t.Fatalf("Index returned error: %v", err)
	}
	edgeScript, err := os.ReadFile(escapedImportEdgeSQL)
	if err != nil {
		t.Fatalf("read %s: %v", escapedImportEdgeSQL, err)
	}
	inserted := strings.TrimSpace(string(runSQLite(t, graphDatabasePath(t, cacheDir, escapedImportProject), edgeScript)))
	if inserted != "2" {
		t.Fatalf("%s created %q IMPORTS edges; want 2", escapedImportEdgeSQL, inserted)
	}
	updatedSource, err := os.ReadFile(escapedImportUpdate)
	if err != nil {
		t.Fatalf("read %s: %v", escapedImportUpdate, err)
	}
	if err = os.WriteFile(filepath.Join(repositoryPath, "main.js"), updatedSource, 0o644); err != nil {
		t.Fatalf("write updated main.js: %v", err)
	}
	if err = engine.Index(context.Background(), repositoryPath, "fast"); err != nil {
		t.Fatalf("incremental Index returned error: %v", err)
	}
	targets := queryColumn(t, engine, escapedImportProject, escapedImportCallsQuery)
	if !slices.Contains(targets, escapedImportWantTarget) {
		t.Errorf("run calls %q; want %q", targets, escapedImportWantTarget)
	}
	if slices.Contains(targets, escapedImportOtherTarget) {
		t.Errorf("run calls %q; it must not call %q", targets, escapedImportOtherTarget)
	}
}

// queryColumn uses query_graph to run the Cypher file at queryPath for project
// and returns the single column of each row.
func queryColumn(t *testing.T, engine *cbm.Engine, project string, queryPath string) []string {
	t.Helper()
	query, err := os.ReadFile(queryPath)
	if err != nil {
		t.Fatalf("read %s: %v", queryPath, err)
	}
	arguments, err := json.Marshal(map[string]string{"query": strings.TrimSpace(string(query)), "project": project})
	if err != nil {
		t.Fatalf("marshal query_graph arguments: %v", err)
	}
	rawResponse, err := engine.Tool("query_graph", string(arguments))
	if err != nil {
		t.Fatalf("query_graph returned error: %v", err)
	}
	var response queryGraphResponse
	if err = json.Unmarshal([]byte(rawResponse), &response); err != nil {
		t.Fatalf("decode query_graph response: %v: %s", err, rawResponse)
	}
	if response.IsError {
		t.Fatalf("query_graph reported an error: %s", rawResponse)
	}
	targets := make([]string, 0, len(response.StructuredContent.Rows))
	for _, row := range response.StructuredContent.Rows {
		var columns []string
		if err = json.Unmarshal(row, &columns); err != nil {
			t.Fatalf("decode query_graph row %s: %v", row, err)
		}
		if len(columns) != 1 {
			t.Fatalf("query_graph row %s has %d columns; want 1", row, len(columns))
		}
		targets = append(targets, columns[0])
	}
	return targets
}
