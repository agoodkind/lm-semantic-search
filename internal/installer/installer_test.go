package installer

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

const unrelatedProgram = "#!/bin/sh\necho unrelated lms\n"

func TestLinkCLIAliasCreatesRelativeSymlinkIdempotently(t *testing.T) {
	binDir := t.TempDir()
	cliPath := filepath.Join(binDir, cliBinary)
	if err := os.WriteFile(cliPath, []byte("cli"), 0o755); err != nil {
		t.Fatalf("write CLI binary: %v", err)
	}

	for attempt := 1; attempt <= 2; attempt++ {
		if err := LinkCLIAlias(binDir); err != nil {
			t.Fatalf("LinkCLIAlias() attempt %d error = %v", attempt, err)
		}
	}

	aliasPath := filepath.Join(binDir, CLIAlias)
	target, err := os.Readlink(aliasPath)
	if err != nil {
		t.Fatalf("read alias symlink: %v", err)
	}
	if target != cliBinary {
		t.Fatalf("alias target = %q, want relative %q", target, cliBinary)
	}
	contents, err := os.ReadFile(aliasPath)
	if err != nil {
		t.Fatalf("read through alias: %v", err)
	}
	if string(contents) != "cli" {
		t.Fatalf("alias resolves to contents %q, want the CLI binary", contents)
	}
	entries, err := os.ReadDir(binDir)
	if err != nil {
		t.Fatalf("read bin dir: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("bin dir has %d entries, want the CLI binary and the alias", len(entries))
	}
}

func TestLinkCLIAliasRefusesRegularFile(t *testing.T) {
	binDir := t.TempDir()
	aliasPath := writeUnrelatedAlias(t, binDir)

	if err := LinkCLIAlias(binDir); err == nil {
		t.Fatal("LinkCLIAlias() replaced a regular file, want an error")
	}
	assertUnrelatedAliasUnchanged(t, aliasPath)
}

func TestRunRefusesRegularAliasBeforeDownloading(t *testing.T) {
	var requestCount atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(
		responseWriter http.ResponseWriter,
		_ *http.Request,
	) {
		requestCount.Add(1)
		responseWriter.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()
	t.Setenv(updateAPIBaseURLEnv, server.URL)

	binDir := t.TempDir()
	aliasPath := writeUnrelatedAlias(t, binDir)
	var stdout bytes.Buffer
	err := Run(context.Background(), Options{
		BinDir:         binDir,
		InstallService: false,
		Stdout:         &stdout,
	})
	if err == nil {
		t.Fatal("Run() installed over a regular lms file, want an error")
	}
	if requestCount.Load() != 0 {
		t.Fatalf("Run() made %d release requests before refusing", requestCount.Load())
	}
	assertUnrelatedAliasUnchanged(t, aliasPath)
	entries, readErr := os.ReadDir(binDir)
	if readErr != nil {
		t.Fatalf("read bin dir: %v", readErr)
	}
	if len(entries) != 1 {
		t.Fatalf("bin dir has %d entries after refusal, want only the original lms", len(entries))
	}
}

func writeUnrelatedAlias(t *testing.T, binDir string) string {
	t.Helper()
	aliasPath := filepath.Join(binDir, CLIAlias)
	if err := os.WriteFile(aliasPath, []byte(unrelatedProgram), 0o755); err != nil {
		t.Fatalf("write unrelated lms: %v", err)
	}
	return aliasPath
}

func assertUnrelatedAliasUnchanged(t *testing.T, aliasPath string) {
	t.Helper()
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
}
