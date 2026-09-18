//go:build installlive

// Package installlive runs the built lm-semantic-search CLI's install command
// against the real GitHub release and the real pinned ONNX Runtime archive,
// writing only into temporary directories.
package installlive

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
)

const (
	cliBinary    = "lm-semantic-search"
	daemonBinary = "lm-semantic-search-daemon"
	mcpBinary    = "lm-semantic-search-mcp"
	cliAlias     = "lms"
	cliPackage   = "goodkind.io/lm-semantic-search/cmd/lm-semantic-search"

	unrelatedProgram = "#!/bin/sh\necho unrelated lms\n"
)

var builtCLIPath string

func TestMain(m *testing.M) {
	buildDirectory, err := os.MkdirTemp("", "lms-install-live-")
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "create build directory: %v\n", err)
		os.Exit(1)
	}
	builtCLIPath = filepath.Join(buildDirectory, cliBinary)
	build := exec.Command("go", "build", "-o", builtCLIPath, cliPackage)
	if output, buildErr := build.CombinedOutput(); buildErr != nil {
		_, _ = fmt.Fprintf(os.Stderr, "build CLI: %v\n%s", buildErr, output)
		_ = os.RemoveAll(buildDirectory)
		os.Exit(1)
	}
	exitCode := m.Run()
	_ = os.RemoveAll(buildDirectory)
	os.Exit(exitCode)
}

type onnxLibraryNames struct {
	versioned   string
	soname      string
	unversioned string
}

func expectedONNXLibraryNames(t *testing.T) onnxLibraryNames {
	t.Helper()
	switch runtime.GOOS {
	case "darwin":
		return onnxLibraryNames{
			versioned:   "libonnxruntime.1.27.0.dylib",
			soname:      "libonnxruntime.1.dylib",
			unversioned: "libonnxruntime.dylib",
		}
	case "linux":
		return onnxLibraryNames{
			versioned:   "libonnxruntime.so.1.27.0",
			soname:      "libonnxruntime.so.1",
			unversioned: "libonnxruntime.so",
		}
	default:
		t.Fatalf("no ONNX Runtime release for GOOS %s", runtime.GOOS)
		return onnxLibraryNames{}
	}
}

type commandResult struct {
	exitCode int
	stdout   string
	stderr   string
}

func runCommand(t *testing.T, path string, args ...string) commandResult {
	t.Helper()
	command := exec.Command(path, args...)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	err := command.Run()
	exitCode := 0
	if err != nil {
		var exitError *exec.ExitError
		if !errors.As(err, &exitError) {
			t.Fatalf("run %s %v: %v", path, args, err)
		}
		exitCode = exitError.ExitCode()
	}
	return commandResult{exitCode: exitCode, stdout: stdout.String(), stderr: stderr.String()}
}

func runInstall(t *testing.T, binDir string) commandResult {
	t.Helper()
	return runCommand(t, builtCLIPath, "install", "--no-service", "--bin-dir", binDir)
}

func directoryEntryNames(t *testing.T, directory string) []string {
	t.Helper()
	entries, err := os.ReadDir(directory)
	if err != nil {
		t.Fatalf("read %s: %v", directory, err)
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	sort.Strings(names)
	return names
}

func TestInstallPlacesReleaseBinariesONNXRuntimeAndAliasIdempotently(t *testing.T) {
	binDir := t.TempDir()
	libraryNames := expectedONNXLibraryNames(t)
	expectedEntries := []string{
		cliBinary,
		daemonBinary,
		mcpBinary,
		libraryNames.versioned,
		libraryNames.soname,
		libraryNames.unversioned,
		cliAlias,
	}
	sort.Strings(expectedEntries)

	// The second run proves a reinstall over a complete install converges on
	// the same directory instead of failing on the files the first run left.
	for attempt := 1; attempt <= 2; attempt++ {
		result := runInstall(t, binDir)
		if result.exitCode != 0 {
			t.Fatalf("install attempt %d exit = %d\nstdout:\n%s\nstderr:\n%s",
				attempt, result.exitCode, result.stdout, result.stderr)
		}
		if strings.Contains(result.stderr, "ERROR") {
			t.Fatalf("install attempt %d logged an error:\n%s", attempt, result.stderr)
		}
		if !strings.Contains(result.stdout, "service setup skipped") {
			t.Fatalf("install attempt %d did not skip the service:\n%s", attempt, result.stdout)
		}
		entries := directoryEntryNames(t, binDir)
		if strings.Join(entries, ",") != strings.Join(expectedEntries, ",") {
			t.Fatalf("install attempt %d bin dir = %v, want %v", attempt, entries, expectedEntries)
		}
	}

	for _, binary := range []string{cliBinary, daemonBinary, mcpBinary, libraryNames.versioned} {
		info, err := os.Lstat(filepath.Join(binDir, binary))
		if err != nil {
			t.Fatalf("stat %s: %v", binary, err)
		}
		if !info.Mode().IsRegular() || info.Mode().Perm()&0o111 == 0 {
			t.Fatalf("%s mode = %v, want an executable regular file", binary, info.Mode())
		}
	}
	symlinkTargets := map[string]string{
		libraryNames.soname:      libraryNames.versioned,
		libraryNames.unversioned: libraryNames.versioned,
		cliAlias:                 cliBinary,
	}
	for linkName, wantTarget := range symlinkTargets {
		target, err := os.Readlink(filepath.Join(binDir, linkName))
		if err != nil {
			t.Fatalf("read %s symlink: %v", linkName, err)
		}
		if target != wantTarget {
			t.Fatalf("%s -> %q, want relative %q", linkName, target, wantTarget)
		}
	}

	aliasHelp := runCommand(t, filepath.Join(binDir, cliAlias), "--help")
	if aliasHelp.exitCode != 0 || !strings.Contains(aliasHelp.stdout, "Usage:") {
		t.Fatalf("lms --help exit = %d\nstdout:\n%s\nstderr:\n%s",
			aliasHelp.exitCode, aliasHelp.stdout, aliasHelp.stderr)
	}
	// The daemon links libonnxruntime through a loader-relative runpath, so the
	// dynamic loader refuses to start it unless the library sits beside it.
	daemonVersion := runCommand(t, filepath.Join(binDir, daemonBinary), "version")
	if daemonVersion.exitCode != 0 || !strings.Contains(daemonVersion.stdout, "version:") {
		t.Fatalf("daemon version exit = %d\nstdout:\n%s\nstderr:\n%s",
			daemonVersion.exitCode, daemonVersion.stdout, daemonVersion.stderr)
	}
}

func TestInstallRefusesRegularLMSFileAndLeavesItUnchanged(t *testing.T) {
	binDir := t.TempDir()
	aliasPath := filepath.Join(binDir, cliAlias)
	if err := os.WriteFile(aliasPath, []byte(unrelatedProgram), 0o755); err != nil {
		t.Fatalf("write unrelated lms: %v", err)
	}

	result := runInstall(t, binDir)
	if result.exitCode == 0 {
		t.Fatalf("install over a regular lms file succeeded\nstdout:\n%s", result.stdout)
	}
	if !strings.Contains(result.stderr, "exists and is not a symlink") {
		t.Fatalf("install stderr does not explain the lms conflict:\n%s", result.stderr)
	}

	info, err := os.Lstat(aliasPath)
	if err != nil {
		t.Fatalf("stat lms: %v", err)
	}
	if !info.Mode().IsRegular() {
		t.Fatalf("lms mode = %v, want the original regular file", info.Mode())
	}
	contents, err := os.ReadFile(aliasPath)
	if err != nil {
		t.Fatalf("read lms: %v", err)
	}
	if string(contents) != unrelatedProgram {
		t.Fatalf("lms contents changed to %q", contents)
	}
	entries := directoryEntryNames(t, binDir)
	if strings.Join(entries, ",") != cliAlias {
		t.Fatalf("bin dir after refusal = %v, want only the original lms", entries)
	}
}
