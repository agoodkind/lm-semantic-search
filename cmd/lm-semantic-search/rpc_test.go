package main

import (
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/response"
)

func TestCurrentClientInfoCarriesCallerCwd(t *testing.T) {
	info, err := response.CurrentClientInfo()
	if err != nil {
		t.Fatalf("response.CurrentClientInfo returned error: %v", err)
	}
	if info.GetCallerCwd() == "" {
		t.Fatal("response.CurrentClientInfo did not set caller_cwd")
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
