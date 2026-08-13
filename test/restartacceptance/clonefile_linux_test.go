//go:build restartacceptance && linux

package restartacceptance

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestCloneFileRemovesDestinationAfterCloneFailure(t *testing.T) {
	source := t.TempDir()
	destination := filepath.Join(t.TempDir(), "clone")

	if err := cloneFile(source, destination); err == nil {
		t.Fatal("cloning a directory succeeded")
	}
	if _, err := os.Lstat(destination); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("failed clone destination remains: %v", err)
	}
}
