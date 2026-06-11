package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCurrentClientInfoCarriesCallerCwd(t *testing.T) {
	info, err := currentClientInfo()
	if err != nil {
		t.Fatalf("currentClientInfo returned error: %v", err)
	}
	if info.GetCallerCwd() == "" {
		t.Fatal("currentClientInfo did not set caller_cwd")
	}
	if !filepath.IsAbs(info.GetCallerCwd()) {
		t.Fatalf("caller_cwd %q is not absolute", info.GetCallerCwd())
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if info.GetCallerCwd() != wd {
		t.Fatalf("caller_cwd = %q, want %q", info.GetCallerCwd(), wd)
	}
}
