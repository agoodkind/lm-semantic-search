//go:build restartacceptance

package restartacceptance

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"golang.org/x/sys/unix"
)

type preparationInput struct {
	Context           context.Context
	Confirmation      string
	BackupRoot        string
	RunParent         string
	Home              string
	ExpectedChecksums map[string]string
	AvailableBytes    int64
	Ports             []int
	Now               time.Time
	Entropy           io.Reader
}

type acceptanceRun struct {
	ID             string
	Paths          runPaths
	Binaries       installedBinaries
	ComposeProject string
	BackupRoot     string
	ArchiveSizes   []int64
}

func prepareAcceptanceRun(ctx context.Context, now time.Time, entropy io.Reader) (acceptanceRun, error) {
	confirmation := os.Getenv(restartAcceptanceOptIn)
	if err := validateOptIn(confirmation); err != nil {
		return acceptanceRun{}, err
	}
	backupRoot, err := requiredDirectoryFromEnvironment(backupRootEnvironment)
	if err != nil {
		return acceptanceRun{}, err
	}
	configuredRunParent, err := requiredDirectoryFromEnvironment(runParentEnvironment)
	if err != nil {
		return acceptanceRun{}, err
	}
	if err := validateRecordedImages(ctx, execCommandRunner{}); err != nil {
		return acceptanceRun{}, err
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return acceptanceRun{}, fmt.Errorf("resolve home directory: %w", err)
	}
	availableBytes, err := availableDiskBytes(configuredRunParent)
	if err != nil {
		return acceptanceRun{}, err
	}
	return prepareRun(preparationInput{
		Context:           ctx,
		Confirmation:      confirmation,
		BackupRoot:        backupRoot,
		RunParent:         configuredRunParent,
		Home:              home,
		ExpectedChecksums: expectedBackupChecksums,
		AvailableBytes:    availableBytes,
		Ports: []int{
			etcdClientPort,
			minioAPIPort,
			minioConsolePort,
			milvusGRPCPort,
			milvusHealthPort,
			milvusProxyPort,
			embeddingProxyPort,
		},
		Now:     now,
		Entropy: entropy,
	})
}

func prepareRun(input preparationInput) (_ acceptanceRun, runErr error) {
	if err := validateOptIn(input.Confirmation); err != nil {
		return acceptanceRun{}, err
	}
	runID, err := newRunID(input.Now, input.Entropy)
	if err != nil {
		return acceptanceRun{}, err
	}
	runRoot := filepath.Join(input.RunParent, runID)
	if err := validateRunRoot(input.RunParent, runRoot); err != nil {
		return acceptanceRun{}, err
	}
	if err := verifyChecksums(input.BackupRoot, input.ExpectedChecksums); err != nil {
		return acceptanceRun{}, err
	}
	archiveSizes, err := restoreArchiveSizes(input.BackupRoot)
	if err != nil {
		return acceptanceRun{}, err
	}
	if err := validateFreeSpace(input.AvailableBytes, archiveSizes); err != nil {
		return acceptanceRun{}, err
	}
	if err := validatePorts(input.Ports); err != nil {
		return acceptanceRun{}, err
	}
	binaries, err := validateInstalledBinaries(input.Home)
	if err != nil {
		return acceptanceRun{}, err
	}
	if err := os.MkdirAll(input.RunParent, 0o700); err != nil {
		return acceptanceRun{}, fmt.Errorf("create run parent: %w", err)
	}
	parentInfo, err := os.Lstat(input.RunParent)
	if err != nil {
		return acceptanceRun{}, fmt.Errorf("inspect run parent: %w", err)
	}
	if !parentInfo.IsDir() || parentInfo.Mode()&os.ModeSymlink != 0 {
		return acceptanceRun{}, fmt.Errorf("run parent %q is not a real directory", input.RunParent)
	}
	if err := os.Mkdir(runRoot, 0o700); err != nil {
		return acceptanceRun{}, fmt.Errorf("create guarded run root: %w", err)
	}
	keepRunRoot := false
	defer func() {
		if keepRunRoot {
			return
		}
		cleanupContext, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		runErr = errors.Join(runErr, removeTree(cleanupContext, runRoot))
	}()
	preparationContext := input.Context
	if preparationContext == nil {
		preparationContext = context.Background()
	}
	if err := context.Cause(preparationContext); err != nil {
		return acceptanceRun{}, fmt.Errorf("prepare guarded run: %w", err)
	}
	paths := pathsForRun(runRoot)
	if err := createIsolationLayout(paths); err != nil {
		return acceptanceRun{}, err
	}
	if err := restoreSourceData(preparationContext, input.BackupRoot, input.ExpectedChecksums, paths); err != nil {
		return acceptanceRun{}, err
	}
	recorder := newEvidenceRecorder(paths, time.Now)
	if err := recorder.Record("restore", "passed", map[string]string{
		"backup_root": input.BackupRoot,
		"run_root":    runRoot,
	}); err != nil {
		return acceptanceRun{}, err
	}
	result := acceptanceRun{
		ID:             runID,
		Paths:          paths,
		Binaries:       binaries,
		ComposeProject: "lms-restart-" + runID[len(runID)-8:],
		BackupRoot:     input.BackupRoot,
		ArchiveSizes:   append([]int64(nil), archiveSizes...),
	}
	keepRunRoot = true
	return result, nil
}

func restoreArchiveSizes(backupRoot string) ([]int64, error) {
	sizes := make([]int64, 0, 4)
	for _, name := range []string{"etcd.tar", "milvus.tar", "minio.tar", "minio-default.tar"} {
		info, err := os.Stat(filepath.Join(backupRoot, name))
		if err != nil {
			return nil, fmt.Errorf("inspect restore archive %q: %w", name, err)
		}
		if !info.Mode().IsRegular() {
			return nil, fmt.Errorf("restore archive %q is not a regular file", name)
		}
		sizes = append(sizes, info.Size())
	}
	return sizes, nil
}

func restoreSourceData(
	ctx context.Context,
	backupRoot string,
	expectedChecksums map[string]string,
	paths runPaths,
) error {
	restores := []struct {
		archive     string
		destination string
	}{
		{archive: "etcd.tar", destination: paths.SourceEtcd},
		{archive: "milvus.tar", destination: paths.SourceMilvus},
		{archive: "minio.tar", destination: paths.SourceMinIO},
		{archive: "minio-default.tar", destination: paths.SourceMinIODefault},
	}
	for _, restore := range restores {
		if err := context.Cause(ctx); err != nil {
			return fmt.Errorf("restore source data: %w", err)
		}
		if err := restoreVerifiedArchive(
			ctx,
			filepath.Join(backupRoot, restore.archive),
			expectedChecksums[restore.archive],
			restore.destination,
		); err != nil {
			return fmt.Errorf("restore %s: %w", restore.archive, err)
		}
	}
	return nil
}

func restoreVerifiedArchive(ctx context.Context, path string, expectedChecksum string, destination string) error {
	archive, err := openNoFollowRegular(path)
	if err != nil {
		return err
	}
	defer func() { _ = archive.Close() }()
	return restoreVerifiedArchiveFile(ctx, archive, expectedChecksum, destination)
}

func restoreVerifiedArchiveFile(
	ctx context.Context,
	archive *os.File,
	expectedChecksum string,
	destination string,
) error {
	if expectedChecksum == "" {
		return fmt.Errorf("restore archive checksum is empty")
	}
	digest := sha256.New()
	if _, err := io.Copy(digest, &contextReader{ctx: ctx, reader: archive}); err != nil {
		return fmt.Errorf("hash restore archive: %w", err)
	}
	actualChecksum := hex.EncodeToString(digest.Sum(nil))
	if actualChecksum != expectedChecksum {
		return fmt.Errorf("restore archive checksum is %s, want %s", actualChecksum, expectedChecksum)
	}
	if _, err := archive.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("rewind verified restore archive: %w", err)
	}
	return restoreImmutableArchiveReader(ctx, archive, destination)
}

func openNoFollowRegular(path string) (*os.File, error) {
	descriptor, err := unix.Open(path, unix.O_RDONLY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("open no-follow file %q: %w", path, err)
	}
	file := os.NewFile(uintptr(descriptor), path)
	info, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("inspect opened file %q: %w", path, err)
	}
	if !info.Mode().IsRegular() {
		_ = file.Close()
		return nil, fmt.Errorf("opened file %q is not regular", path)
	}
	return file, nil
}

func availableDiskBytes(path string) (int64, error) {
	var statistics unix.Statfs_t
	if err := unix.Statfs(path, &statistics); err != nil {
		return 0, fmt.Errorf("read free space for %q: %w", path, err)
	}
	available := uint64(statistics.Bavail) * uint64(statistics.Bsize)
	maximumInt64 := uint64(^uint64(0) >> 1)
	if available > maximumInt64 {
		return 0, fmt.Errorf("free space for %q exceeds supported range", path)
	}
	return int64(available), nil
}
