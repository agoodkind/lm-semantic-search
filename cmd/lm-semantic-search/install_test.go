package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

func TestInstallCommandRefusesRegularLMSFileWithoutDownloading(t *testing.T) {
	var requestCount atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(
		responseWriter http.ResponseWriter,
		_ *http.Request,
	) {
		requestCount.Add(1)
		responseWriter.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()
	t.Setenv("LM_SEMANTIC_SEARCH_UPDATE_API_BASE_URL", server.URL)

	binDir := t.TempDir()
	aliasPath := filepath.Join(binDir, "lms")
	const unrelatedProgram = "#!/bin/sh\necho unrelated lms\n"
	if err := os.WriteFile(aliasPath, []byte(unrelatedProgram), 0o755); err != nil {
		t.Fatalf("write unrelated lms: %v", err)
	}

	root, _, _ := testRoot()
	root.SetArgs([]string{"install", "--no-service", "--bin-dir", binDir})
	if err := root.Execute(); err == nil {
		t.Fatal("install replaced a regular lms file, want an error")
	}

	if requestCount.Load() != 0 {
		t.Fatalf("install made %d release requests before refusing", requestCount.Load())
	}
	contents, err := os.ReadFile(aliasPath)
	if err != nil {
		t.Fatalf("read lms: %v", err)
	}
	if string(contents) != unrelatedProgram {
		t.Fatalf("lms contents changed to %q", contents)
	}
}
