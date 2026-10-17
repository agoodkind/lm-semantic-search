package merkle_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/indexability"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
)

// TestCaptureRecordsInodeForRegularFileAndNotForSymlink proves Capture records
// the (device, inode) entry that StatInode reads for a regular file. Capture
// must not record an inode entry for a symbolic link to that file.
func TestCaptureRecordsInodeForRegularFileAndNotForSymlink(t *testing.T) {
	root := t.TempDir()
	regularPath := filepath.Join(root, "regular.go")
	if err := os.WriteFile(regularPath, []byte("package p\nfunc Regular() {}\n"), 0o644); err != nil {
		t.Fatalf("write %s: %v", regularPath, err)
	}
	linkPath := filepath.Join(root, "link.go")
	if err := os.Symlink(regularPath, linkPath); err != nil {
		t.Fatalf("symlink %s to %s: %v", linkPath, regularPath, err)
	}

	snapshot, err := merkle.Capture(context.Background(), indexability.NewResolver(nil, nil), "cb", root, model.IndexConfig{})
	if err != nil {
		t.Fatalf("Capture returned error: %v", err)
	}

	wantRegular, err := merkle.StatInode(regularPath)
	if err != nil {
		t.Fatalf("StatInode(%s) returned error: %v", regularPath, err)
	}
	if got := snapshot.Inodes["regular.go"]; got != wantRegular {
		t.Fatalf("Capture recorded inode %+v for regular.go. StatInode reads %+v.", got, wantRegular)
	}
	if got, found := snapshot.Inodes["link.go"]; found {
		t.Fatalf("Capture recorded inode %+v for the symbolic link link.go. The test expected the snapshot to omit an inode entry for the link.", got)
	}
}
