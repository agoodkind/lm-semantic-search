package daemon

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/discovery"
	"goodkind.io/lm-semantic-search/internal/model"
)

// TestCodeItemSourceCaptureReportsRules proves one capture both snapshots the
// tree and hands the resolved ignore rules to the registered callback, so the
// daemon can persist them without a second walk.
func TestCodeItemSourceCaptureReportsRules(t *testing.T) {
	t.Parallel()
	tempDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(tempDir, ".gitignore"), []byte("skipped/\n"), 0o644); err != nil {
		t.Fatalf("write .gitignore: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tempDir, "kept.go"), []byte("package x\n"), 0o644); err != nil {
		t.Fatalf("write kept.go: %v", err)
	}

	var reported discovery.IgnoreRules
	source := newCodeItemSource(nil, tempDir, model.IndexConfig{}, func(rules discovery.IgnoreRules) {
		reported = rules
	})

	if _, err := source.capture(context.Background()); err != nil {
		t.Fatalf("capture returned error: %v", err)
	}
	if reported.IsEmpty() {
		t.Fatal("capture did not report the resolved ignore rules")
	}
	if excluded, _, _ := discovery.PathIgnored("skipped/file.go", reported); !excluded {
		t.Fatal("reported rules do not contain the root .gitignore pattern")
	}
}
