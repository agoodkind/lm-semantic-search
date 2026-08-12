//go:build restartacceptance

package restartacceptance

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestPrepareAcceptanceRunRejectsMissingOptInBeforeDockerProbe(t *testing.T) {
	t.Setenv(restartAcceptanceOptIn, "")
	t.Setenv("PATH", t.TempDir())

	_, err := prepareAcceptanceRun(context.Background(), time.Now(), bytes.NewReader(make([]byte, 4)))
	if err == nil {
		t.Fatal("missing opt-in was accepted")
	}
	if !strings.Contains(err.Error(), restartAcceptanceOptIn) {
		t.Fatalf("prepare error = %q, want opt-in error", err)
	}
}

func TestPrepareRunValidatesEveryGuardBeforeCreatingRunRoot(t *testing.T) {
	root := t.TempDir()
	backup := filepath.Join(root, "backup")
	parent := filepath.Join(root, "lms-restart-acceptance")
	home := filepath.Join(root, "home")
	if err := os.MkdirAll(backup, 0o755); err != nil {
		t.Fatalf("create backup: %v", err)
	}
	createInstalledBinaryFixtures(t, home)
	wantChecksums := writeBackupFixtures(t, backup)
	if err := os.WriteFile(filepath.Join(backup, "milvus.tar"), []byte("corrupt"), 0o600); err != nil {
		t.Fatalf("corrupt backup: %v", err)
	}
	now := time.Date(2026, 8, 12, 1, 2, 3, 0, time.UTC)
	input := preparationInput{
		Confirmation:      restartAcceptanceConfirmation,
		BackupRoot:        backup,
		RunParent:         parent,
		Home:              home,
		ExpectedChecksums: wantChecksums,
		AvailableBytes:    1 << 30,
		Now:               now,
		Entropy:           bytes.NewReader([]byte{0xab, 0xcd, 0xef, 0x01}),
	}
	if _, err := prepareRun(input); err == nil {
		t.Fatal("corrupt backup was accepted")
	}
	runRoot := filepath.Join(parent, "20260812T010203Z-abcdef01")
	if _, err := os.Lstat(runRoot); !os.IsNotExist(err) {
		t.Fatalf("run root was created before all guards passed: %v", err)
	}
}

func TestPrepareRunRestoresImmutableSourcesAfterAllGuardsPass(t *testing.T) {
	root := t.TempDir()
	backup := filepath.Join(root, "backup")
	parent := filepath.Join(root, "lms-restart-acceptance")
	home := filepath.Join(root, "home")
	if err := os.MkdirAll(backup, 0o755); err != nil {
		t.Fatalf("create backup: %v", err)
	}
	createInstalledBinaryFixtures(t, home)
	wantChecksums := writeBackupFixtures(t, backup)
	now := time.Date(2026, 8, 12, 1, 2, 3, 0, time.UTC)
	run, err := prepareRun(preparationInput{
		Confirmation:      restartAcceptanceConfirmation,
		BackupRoot:        backup,
		RunParent:         parent,
		Home:              home,
		ExpectedChecksums: wantChecksums,
		AvailableBytes:    1 << 30,
		Now:               now,
		Entropy:           bytes.NewReader([]byte{0xab, 0xcd, 0xef, 0x01}),
	})
	if err != nil {
		t.Fatalf("prepare run: %v", err)
	}
	t.Cleanup(func() { makeTreeWritable(run.Paths.Source) })
	if run.ID != "20260812T010203Z-abcdef01" || run.ComposeProject != "lms-restart-abcdef01" {
		t.Fatalf("run = %+v", run)
	}
	for _, path := range []string{
		filepath.Join(run.Paths.SourceEtcd, "data"),
		filepath.Join(run.Paths.SourceMilvus, "data"),
		filepath.Join(run.Paths.SourceMinIO, "data"),
		filepath.Join(run.Paths.SourceMinIODefault, "data"),
	} {
		info, statErr := os.Stat(path)
		if statErr != nil {
			t.Fatalf("stat restored source %s: %v", path, statErr)
		}
		if info.Mode().Perm()&0o222 != 0 {
			t.Fatalf("restored source %s mode = %o", path, info.Mode().Perm())
		}
	}
}

func TestVerifiedArchiveExtractionUsesTheHashedDescriptor(t *testing.T) {
	backup := filepath.Join(t.TempDir(), "backup")
	if err := os.MkdirAll(backup, 0o755); err != nil {
		t.Fatalf("create backup: %v", err)
	}
	wantChecksums := writeBackupFixtures(t, backup)
	archivePath := filepath.Join(backup, "milvus.tar")
	archive, err := openNoFollowRegular(archivePath)
	if err != nil {
		t.Fatalf("open archive: %v", err)
	}
	defer func() { _ = archive.Close() }()
	if err := os.Rename(archivePath, archivePath+".opened"); err != nil {
		t.Fatalf("rename opened archive: %v", err)
	}
	if err := os.WriteFile(archivePath, []byte("replacement"), 0o600); err != nil {
		t.Fatalf("replace archive path: %v", err)
	}
	destination := filepath.Join(t.TempDir(), "restored")
	t.Cleanup(func() { makeTreeWritable(destination) })
	if err := restoreVerifiedArchiveFile(
		context.Background(),
		archive,
		wantChecksums["milvus.tar"],
		destination,
	); err != nil {
		t.Fatalf("restore verified descriptor: %v", err)
	}
	body, err := os.ReadFile(filepath.Join(destination, "data"))
	if err != nil {
		t.Fatalf("read restored data: %v", err)
	}
	if string(body) != "milvus.tar" {
		t.Fatalf("restored body = %q", body)
	}
}

func TestPrepareRunRemovesGuardedRootAfterExtractionFailure(t *testing.T) {
	root := t.TempDir()
	backup := filepath.Join(root, "backup")
	parent := filepath.Join(root, "lms-restart-acceptance")
	home := filepath.Join(root, "home")
	if err := os.MkdirAll(backup, 0o755); err != nil {
		t.Fatalf("create backup: %v", err)
	}
	createInstalledBinaryFixtures(t, home)
	wantChecksums := writeBackupFixtures(t, backup)
	corrupt := []byte("not a tar archive")
	if err := os.WriteFile(filepath.Join(backup, "milvus.tar"), corrupt, 0o600); err != nil {
		t.Fatalf("corrupt archive: %v", err)
	}
	wantChecksums["milvus.tar"] = fmt.Sprintf("%x", sha256.Sum256(corrupt))
	writeChecksumManifest(t, backup, wantChecksums)
	input := validPreparationInput(backup, parent, home, wantChecksums)
	if _, err := prepareRun(input); err == nil {
		t.Fatal("invalid tar was accepted")
	}
	if _, err := os.Lstat(filepath.Join(parent, "20260812T010203Z-abcdef01")); !os.IsNotExist(err) {
		t.Fatalf("failed run root remains: %v", err)
	}
}

func TestPrepareRunRemovesGuardedRootAfterCancellation(t *testing.T) {
	root := t.TempDir()
	backup := filepath.Join(root, "backup")
	parent := filepath.Join(root, "lms-restart-acceptance")
	home := filepath.Join(root, "home")
	if err := os.MkdirAll(backup, 0o755); err != nil {
		t.Fatalf("create backup: %v", err)
	}
	createInstalledBinaryFixtures(t, home)
	wantChecksums := writeBackupFixtures(t, backup)
	input := validPreparationInput(backup, parent, home, wantChecksums)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	input.Context = ctx
	if _, err := prepareRun(input); err == nil {
		t.Fatal("cancelled preparation succeeded")
	}
	if _, err := os.Lstat(filepath.Join(parent, "20260812T010203Z-abcdef01")); !os.IsNotExist(err) {
		t.Fatalf("cancelled run root remains: %v", err)
	}
}

func TestPrepareRunCancelsDuringExtractionAndRemovesGuardedRoot(t *testing.T) {
	root := t.TempDir()
	backup := filepath.Join(root, "backup")
	parent := filepath.Join(root, "lms-restart-acceptance")
	home := filepath.Join(root, "home")
	if err := os.MkdirAll(backup, 0o755); err != nil {
		t.Fatalf("create backup: %v", err)
	}
	createInstalledBinaryFixtures(t, home)
	wantChecksums := writeBackupFixtures(t, backup)
	runRoot := filepath.Join(parent, "20260812T010203Z-abcdef01")
	partialFile := filepath.Join(runRoot, "restore", "source", "etcd", "data")
	extractionContext := &cancelAfterFileWriteContext{
		Context: context.Background(),
		path:    partialFile,
	}
	input := validPreparationInput(backup, parent, home, wantChecksums)
	input.Context = extractionContext
	if _, err := prepareRun(input); !errors.Is(err, context.Canceled) {
		t.Fatalf("prepare cancellation error = %v, want context canceled", err)
	}
	if !extractionContext.cancelled.Load() {
		t.Fatal("preparation did not reach archive file extraction")
	}
	if _, err := os.Lstat(runRoot); !os.IsNotExist(err) {
		t.Fatalf("cancelled run root remains: %v", err)
	}
}

type cancelAfterFileWriteContext struct {
	context.Context
	path      string
	cancelled atomic.Bool
}

func (ctx *cancelAfterFileWriteContext) Err() error {
	if ctx.cancelled.Load() {
		return context.Canceled
	}
	info, err := os.Stat(ctx.path)
	if err == nil && info.Size() > 0 {
		ctx.cancelled.Store(true)
		return context.Canceled
	}
	return nil
}

func validPreparationInput(backup string, parent string, home string, checksums map[string]string) preparationInput {
	return preparationInput{
		Confirmation:      restartAcceptanceConfirmation,
		BackupRoot:        backup,
		RunParent:         parent,
		Home:              home,
		ExpectedChecksums: checksums,
		AvailableBytes:    1 << 30,
		Now:               time.Date(2026, 8, 12, 1, 2, 3, 0, time.UTC),
		Entropy:           bytes.NewReader([]byte{0xab, 0xcd, 0xef, 0x01}),
	}
}

func createInstalledBinaryFixtures(t *testing.T, home string) {
	t.Helper()
	binDirectory := filepath.Join(home, ".local", "bin")
	if err := os.MkdirAll(binDirectory, 0o755); err != nil {
		t.Fatalf("create binary directory: %v", err)
	}
	for _, name := range []string{"lm-semantic-search-daemon", "lm-semantic-search", "clyde"} {
		if err := os.WriteFile(filepath.Join(binDirectory, name), []byte("binary"), 0o755); err != nil {
			t.Fatalf("write binary fixture: %v", err)
		}
	}
}

func writeBackupFixtures(t *testing.T, root string) map[string]string {
	t.Helper()
	fixtures := map[string][]byte{
		"backup.yaml":        []byte("backup"),
		"docker-compose.yml": []byte("compose"),
	}
	for _, name := range []string{"etcd.tar", "milvus.tar", "minio.tar", "minio-default.tar"} {
		var archive bytes.Buffer
		writer := tar.NewWriter(&archive)
		body := []byte(name)
		if err := writer.WriteHeader(&tar.Header{Name: "data", Size: int64(len(body)), Mode: 0o600}); err != nil {
			t.Fatalf("write archive header: %v", err)
		}
		if _, err := writer.Write(body); err != nil {
			t.Fatalf("write archive body: %v", err)
		}
		if err := writer.Close(); err != nil {
			t.Fatalf("close archive: %v", err)
		}
		fixtures[name] = archive.Bytes()
	}
	want := make(map[string]string, len(fixtures))
	var manifest bytes.Buffer
	for _, name := range []string{"backup.yaml", "docker-compose.yml", "etcd.tar", "milvus.tar", "minio-default.tar", "minio.tar"} {
		body := fixtures[name]
		if err := os.WriteFile(filepath.Join(root, name), body, 0o600); err != nil {
			t.Fatalf("write backup fixture %s: %v", name, err)
		}
		hash := fmt.Sprintf("%x", sha256.Sum256(body))
		want[name] = hash
		fmt.Fprintf(&manifest, "%s  %s\n", hash, name)
	}
	if err := os.WriteFile(filepath.Join(root, checksumManifestName), manifest.Bytes(), 0o600); err != nil {
		t.Fatalf("write checksum manifest: %v", err)
	}
	return want
}

func writeChecksumManifest(t *testing.T, root string, checksums map[string]string) {
	t.Helper()
	var manifest bytes.Buffer
	for _, name := range []string{"backup.yaml", "docker-compose.yml", "etcd.tar", "milvus.tar", "minio-default.tar", "minio.tar"} {
		fmt.Fprintf(&manifest, "%s  %s\n", checksums[name], name)
	}
	if err := os.WriteFile(filepath.Join(root, checksumManifestName), manifest.Bytes(), 0o600); err != nil {
		t.Fatalf("rewrite checksum manifest: %v", err)
	}
}

func makeTreeWritable(root string) {
	_ = filepath.WalkDir(root, func(path string, _ os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		return os.Chmod(path, 0o700)
	})
}
