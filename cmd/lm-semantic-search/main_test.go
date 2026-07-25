package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"goodkind.io/gklog/correlation"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

func TestRootNoArgsShowsHelp(t *testing.T) {
	root, stdout, _ := testRoot()
	root.SetArgs([]string{})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute root help: %v", err)
	}
	output := stdout.String()
	if !strings.Contains(output, "Usage:") {
		t.Fatalf("help output missing Usage: %q", output)
	}
	if !strings.Contains(output, "codebase") {
		t.Fatalf("help output missing codebase group: %q", output)
	}
}

func TestRootUnknownCommandErrors(t *testing.T) {
	root, _, _ := testRoot()
	root.SetArgs([]string{"definitely-not-a-lm-semantic-command"})

	err := root.Execute()
	if err == nil {
		t.Fatalf("expected unknown command error")
	}
	if !strings.Contains(err.Error(), "unknown command") {
		t.Fatalf("error = %q, want unknown command", err.Error())
	}
}

func TestRootRegistersGroupedCommands(t *testing.T) {
	root, _, _ := testRoot()
	expected := [][]string{
		{"codebase", "list"},
		{"codebase", "status"},
		{"codebase", "index"},
		{"codebase", "sync"},
		{"codebase", "search"},
		{"codebase", "clear"},
		{"job", "list"},
		{"job", "get"},
		{"job", "cancel"},
		{"daemon", "status"},
		{"daemon", "stop"},
		{"daemon", "doctor"},
		{"update", "check"},
		{"update", "apply"},
		{"update", "status"},
		{"profile"},
	}
	for _, path := range expected {
		if _, _, err := root.Find(path); err != nil {
			t.Fatalf("command %v not registered: %v", path, err)
		}
	}
}

func TestProfileCommandWritesDaemonConfig(t *testing.T) {
	configRoot := t.TempDir()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("CLAUDE_CONTEXTD_CONFIG_ROOT", configRoot)
	t.Setenv("CLAUDE_CONTEXT_PROFILE", "")

	configPath := filepath.Join(configRoot, "config.json")
	initialConfig := []byte(`{"embeddingModel":"existing-model","futureField":{"enabled":true}}`)
	if err := os.WriteFile(configPath, initialConfig, 0o600); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	root, stdout, _ := testRoot()
	root.SetArgs(
		[]string{
			"profile",
			config.ProfileOffline,
			"--model",
			offlinemodel.BGESmall,
		},
	)
	if err := root.Execute(); err != nil {
		t.Fatalf("profile command returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	var persisted map[string]json.RawMessage
	if err := json.Unmarshal(data, &persisted); err != nil {
		t.Fatalf("Unmarshal returned error: %v", err)
	}
	var profile string
	if err := json.Unmarshal(persisted["profile"], &profile); err != nil {
		t.Fatalf("Unmarshal profile returned error: %v", err)
	}
	if profile != config.ProfileOffline {
		t.Fatalf("profile = %q want %q", profile, config.ProfileOffline)
	}
	var offlineModel string
	if err := json.Unmarshal(persisted["offlineEmbeddingModel"], &offlineModel); err != nil {
		t.Fatalf("Unmarshal offlineEmbeddingModel returned error: %v", err)
	}
	if offlineModel != offlinemodel.BGESmall {
		t.Fatalf(
			"offlineEmbeddingModel = %q want %q",
			offlineModel,
			offlinemodel.BGESmall,
		)
	}
	if string(persisted["embeddingModel"]) != `"existing-model"` {
		t.Fatalf("embeddingModel was not preserved: %s", persisted["embeddingModel"])
	}
	if _, found := persisted["futureField"]; !found {
		t.Fatal("futureField was not preserved")
	}
	if !strings.Contains(stdout.String(), configPath) {
		t.Fatalf("profile output does not name config path: %q", stdout.String())
	}

	resolved, err := config.Default()
	if err != nil {
		t.Fatalf("Default returned error: %v", err)
	}
	if resolved.IndexBackend != config.IndexBackendLocal {
		t.Fatalf("IndexBackend = %q want %q", resolved.IndexBackend, config.IndexBackendLocal)
	}
	if resolved.EmbeddingProvider != config.EmbeddingProviderONNX {
		t.Fatalf(
			"EmbeddingProvider = %q want %q",
			resolved.EmbeddingProvider,
			config.EmbeddingProviderONNX,
		)
	}
}

func TestProfileCommandRejectsUnknownModelWithoutWriting(t *testing.T) {
	configRoot := t.TempDir()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("CLAUDE_CONTEXTD_CONFIG_ROOT", configRoot)
	t.Setenv("CLAUDE_CONTEXT_PROFILE", "")
	configPath := filepath.Join(configRoot, "config.json")
	initialData := []byte("{\"profile\":\"standard\"}\n")
	if err := os.WriteFile(configPath, initialData, 0o600); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	root, _, _ := testRoot()
	root.SetArgs([]string{"profile", config.ProfileOffline, "--model", "unknown"})
	err := root.Execute()
	if err == nil {
		t.Fatal("profile command returned no error")
	}
	wantError := `validate offline model: offline embedding model "unknown" ` +
		`is not supported; use "embeddinggemma" or "bge-small"`
	if err.Error() != wantError {
		t.Fatalf("error = %q want %q", err.Error(), wantError)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	if string(data) != string(initialData) {
		t.Fatalf("config changed after invalid model: %q", data)
	}
}

func TestProfileCommandRejectsModelForStandard(t *testing.T) {
	configRoot := t.TempDir()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("CLAUDE_CONTEXTD_CONFIG_ROOT", configRoot)
	t.Setenv("CLAUDE_CONTEXT_PROFILE", "")
	configPath := filepath.Join(configRoot, "config.json")

	root, _, _ := testRoot()
	root.SetArgs(
		[]string{
			"profile",
			config.ProfileStandard,
			"--model",
			offlinemodel.BGESmall,
		},
	)
	err := root.Execute()
	if err == nil {
		t.Fatal("profile command returned no error")
	}
	if err.Error() != "--model only applies to the offline profile" {
		t.Fatalf("error = %q", err.Error())
	}
	if _, err := os.Stat(configPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("config exists after rejected model or stat failed: %v", err)
	}
}

func TestProfileCommandWritesModelOverNullConfig(t *testing.T) {
	configRoot := t.TempDir()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("CLAUDE_CONTEXTD_CONFIG_ROOT", configRoot)
	t.Setenv("CLAUDE_CONTEXT_PROFILE", "")
	configPath := filepath.Join(configRoot, "config.json")
	if err := os.WriteFile(configPath, []byte("null"), 0o600); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	root, _, _ := testRoot()
	root.SetArgs(
		[]string{
			"profile",
			config.ProfileOffline,
			"--model",
			offlinemodel.BGESmall,
		},
	)
	if err := root.Execute(); err != nil {
		t.Fatalf("profile command returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	var persisted map[string]string
	if err := json.Unmarshal(data, &persisted); err != nil {
		t.Fatalf("Unmarshal returned error: %v", err)
	}
	if persisted["profile"] != config.ProfileOffline {
		t.Fatalf("profile = %q want %q", persisted["profile"], config.ProfileOffline)
	}
	if persisted["offlineEmbeddingModel"] != offlinemodel.BGESmall {
		t.Fatalf(
			"offlineEmbeddingModel = %q want %q",
			persisted["offlineEmbeddingModel"],
			offlinemodel.BGESmall,
		)
	}
}

func TestProfileCommandValidatesArgumentsBeforeWritingConfig(t *testing.T) {
	testCases := []struct {
		name           string
		arguments      []string
		wantError      string
		wantConfigFile bool
	}{
		{
			name:      "zero arguments",
			arguments: []string{"profile"},
			wantError: "profile requires a PROFILE argument (standard|offline)",
		},
		{
			name:           "one valid argument",
			arguments:      []string{"profile", config.ProfileOffline},
			wantConfigFile: true,
		},
		{
			name:      "one invalid argument",
			arguments: []string{"profile", "invalid"},
			wantError: `set daemon profile: profile must be "standard" or "offline"`,
		},
		{
			name:      "extra arguments",
			arguments: []string{"profile", config.ProfileOffline, "extra", "trailing"},
			wantError: "profile received too many arguments: extra trailing",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			configRoot := t.TempDir()
			t.Setenv("HOME", t.TempDir())
			t.Setenv("CLAUDE_CONTEXTD_CONFIG_ROOT", configRoot)
			t.Setenv("CLAUDE_CONTEXT_PROFILE", "")
			configPath := filepath.Join(configRoot, "config.json")

			root, _, _ := testRoot()
			root.SetArgs(testCase.arguments)
			err := root.Execute()
			if testCase.wantError == "" {
				if err != nil {
					t.Fatalf("profile command returned error: %v", err)
				}
			} else {
				if err == nil {
					t.Fatalf("profile command returned no error, want %q", testCase.wantError)
				}
				if err.Error() != testCase.wantError {
					t.Fatalf("error = %q want %q", err.Error(), testCase.wantError)
				}
			}

			_, statErr := os.Stat(configPath)
			if testCase.wantConfigFile {
				if statErr != nil {
					t.Fatalf("config file was not written: %v", statErr)
				}
			} else if !errors.Is(statErr, os.ErrNotExist) {
				t.Fatalf("config file exists or stat failed: %v", statErr)
			}
		})
	}
}

func TestCodebaseStatusRequiresPath(t *testing.T) {
	root, _, _ := testRoot()
	root.SetArgs([]string{"codebase", "status"})

	err := root.Execute()
	if err == nil {
		t.Fatalf("expected missing path error")
	}
	if err.Error() != "codebase status requires PATH" {
		t.Fatalf("error = %q", err.Error())
	}
}

// TestIndexRejectsWaitOutsideHumanMode proves --wait cannot interleave
// progress rendering with machine output.
func TestIndexRejectsWaitOutsideHumanMode(t *testing.T) {
	root, _, _ := testRoot()
	root.SetArgs([]string{"--json", "codebase", "index", "/tmp/x", "--wait"})

	err := root.Execute()
	if err == nil {
		t.Fatal("expected --wait with --json to error")
	}
	if !strings.Contains(err.Error(), "--wait requires human output") {
		t.Fatalf("error = %q, want --wait requires human output", err.Error())
	}
}

// TestRuntimeErrorsDoNotPrintUsage proves a post-parse failure prints no
// usage block (the Ctrl-C dump from the incident).
func TestRuntimeErrorsDoNotPrintUsage(t *testing.T) {
	root, stdout, stderr := testRoot()
	root.SetArgs([]string{"daemon", "status", "--socket", "/nonexistent/socket.sock"})

	_ = root.Execute()
	combined := stdout.String() + stderr.String()
	if strings.Contains(combined, "Usage:") {
		t.Fatalf("runtime error printed usage:\n%s", combined)
	}
}

func TestJobGetRequiresID(t *testing.T) {
	root, _, _ := testRoot()
	root.SetArgs([]string{"job", "get"})

	err := root.Execute()
	if err == nil {
		t.Fatalf("expected missing job id error")
	}
	if err.Error() != "job get requires JOB_ID" {
		t.Fatalf("error = %q", err.Error())
	}
}

func TestVersionCommandPrintsValidatorToken(t *testing.T) {
	root, stdout, _ := testRoot()
	root.SetArgs([]string{"version"})

	if err := root.Execute(); err != nil {
		t.Fatalf("version command returned error: %v", err)
	}
	output := stdout.String()
	if !strings.Contains(output, "version:") {
		t.Fatalf("version output = %q, want version: token", output)
	}
	if !strings.Contains(output, "commit=") {
		t.Fatalf("version output = %q, want commit field", output)
	}
	if !strings.Contains(output, "build_time=") {
		t.Fatalf("version output = %q, want build_time field", output)
	}
}

func TestDaemonRequiresSubcommand(t *testing.T) {
	root, _, _ := testRoot()
	root.SetArgs([]string{"daemon"})

	err := root.Execute()
	if err == nil {
		t.Fatalf("expected daemon subcommand error")
	}
	if err.Error() != "daemon requires a subcommand" {
		t.Fatalf("error = %q", err.Error())
	}
}

func TestFormatCLIErrorKeepsCorrelationHeaderFirst(t *testing.T) {
	t.Parallel()
	message := correlation.HeaderMarker + "trace_id=abc span_id=def\ninternal error; see daemon logs"
	got := formatCLIError(assertError(message))
	if got != message+"\n" {
		t.Fatalf("formatCLIError returned %q", got)
	}
}

func TestFormatCLIErrorPrefixesOrdinaryErrors(t *testing.T) {
	t.Parallel()
	got := formatCLIError(assertError("plain failure"))
	if got != "Error: plain failure\n" {
		t.Fatalf("formatCLIError returned %q", got)
	}
}

func testRoot() (*cobra.Command, *bytes.Buffer, *bytes.Buffer) {
	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}
	root := newRoot("/tmp/lm-semantic-search.sock")
	root.SetOut(stdout)
	root.SetErr(stderr)
	return root, stdout, stderr
}

func assertError(message string) error {
	return &staticError{message: message}
}

type staticError struct {
	message string
}

func (err *staticError) Error() string {
	return err.message
}
