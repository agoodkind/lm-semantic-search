//go:build restartacceptance

package sandboxharness

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestRestoreArchiveRejectsEscapesAndMakesSourceReadOnly(t *testing.T) {
	var archive bytes.Buffer
	writer := tar.NewWriter(&archive)
	if err := writer.WriteHeader(&tar.Header{Name: "data", Typeflag: tar.TypeDir, Mode: 0o700}); err != nil {
		t.Fatal(err)
	}
	body := []byte("restored")
	if err := writer.WriteHeader(&tar.Header{Name: "data/value", Typeflag: tar.TypeReg, Mode: 0o600, Size: int64(len(body))}); err != nil {
		t.Fatal(err)
	}
	if _, err := writer.Write(body); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	restoreParent := t.TempDir()
	destination := filepath.Join(restoreParent, "source")
	t.Cleanup(func() { _ = RemoveTree(context.Background(), restoreParent, destination, "source") })
	if err := RestoreArchiveReader(context.Background(), bytes.NewReader(archive.Bytes()), destination); err != nil {
		t.Fatalf("restore archive: %v", err)
	}
	info, err := os.Stat(filepath.Join(destination, "data", "value"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o400 {
		t.Fatalf("restored mode = %o, want 400", info.Mode().Perm())
	}

	archive.Reset()
	writer = tar.NewWriter(&archive)
	if err := writer.WriteHeader(&tar.Header{Name: "../outside", Typeflag: tar.TypeReg, Size: 0}); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	if err := RestoreArchiveReader(context.Background(), bytes.NewReader(archive.Bytes()), filepath.Join(t.TempDir(), "escape")); err == nil {
		t.Fatal("escaping archive was accepted")
	}
}

func TestVerifyChecksumsAndSpaceGuard(t *testing.T) {
	root := t.TempDir()
	body := []byte("verified")
	digest := sha256.Sum256(body)
	hash := hex.EncodeToString(digest[:])
	if err := os.WriteFile(filepath.Join(root, "data.tar"), body, 0o600); err != nil {
		t.Fatal(err)
	}
	manifest := []byte(hash + "  data.tar\n")
	if err := os.WriteFile(filepath.Join(root, "SHA256SUMS"), manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := VerifyChecksums(root, "SHA256SUMS", map[string]string{"data.tar": hash}); err != nil {
		t.Fatalf("verify checksums: %v", err)
	}
	if required := RequiredBytes([]int64{100}); required != 125 {
		t.Fatalf("required bytes = %d, want 125", required)
	}
	if err := RequireFreeSpace(124, []int64{100}); err == nil {
		t.Fatal("insufficient space was accepted")
	}
}

func TestCleanupRemovesOnlyExactGuardedRoot(t *testing.T) {
	parent := t.TempDir()
	root := filepath.Join(parent, "run-accepted")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(parent, "outside")
	if err := os.Mkdir(outside, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := RemoveTree(context.Background(), parent, root, "run-accepted"); err != nil {
		t.Fatalf("remove guarded root: %v", err)
	}
	if _, err := os.Stat(outside); err != nil {
		t.Fatalf("outside path changed: %v", err)
	}
	if err := RemoveTree(context.Background(), parent, outside+"-escape", "run-accepted"); err == nil {
		t.Fatal("unapproved root was accepted")
	}
}

func TestCloneTreeCreatesWritableCopyWithoutChangingSource(t *testing.T) {
	source := filepath.Join(t.TempDir(), "source")
	if err := os.Mkdir(source, 0o700); err != nil {
		t.Fatal(err)
	}
	sourceFile := filepath.Join(source, "data")
	if err := os.WriteFile(sourceFile, []byte("source"), 0o400); err != nil {
		t.Fatal(err)
	}
	destination := filepath.Join(t.TempDir(), "case")
	if err := CloneTree(source, destination); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(destination, "data"), []byte("case"), 0o600); err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(sourceFile)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != "source" {
		t.Fatalf("source changed to %q", body)
	}
}
