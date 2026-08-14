//go:build restartacceptance

package restartacceptance

import (
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"time"

	"goodkind.io/lm-semantic-search/test/sandboxharness"
)

const (
	restartAcceptanceOptIn        = "LMS_RESTART_ACCEPTANCE_CONFIRM"
	restartAcceptanceConfirmation = "isolated-clone"
	backupRootEnvironment         = "LMS_RESTART_ACCEPTANCE_BACKUP"
	runParentEnvironment          = "LMS_RESTART_ACCEPTANCE_RUN_PARENT"
	checksumManifestName          = "SHA256SUMS"

	etcdClientPort      = 22379
	minioAPIPort        = 29000
	minioConsolePort    = 29001
	milvusGRPCPort      = 29530
	milvusHealthPort    = 29091
	milvusProxyPort     = 39530
	embeddingProxyPort  = 35400
	cloneMilvusDatabase = "default"
)

var (
	runIDPattern            = regexp.MustCompile(`^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$`)
	expectedBackupChecksums = map[string]string{
		"backup.yaml":        "febf8809d5cab4d2856ec6978027ff19e5682523c583ea041a9d3c18bce0c4d6",
		"docker-compose.yml": "72b5af28080fb22997758a336050238bc77340e065716e5b02c31c449d0815b7",
		"etcd.tar":           "9b8a3342f5397ee8a58dbc0eb30b3be78624d3680d660b1496997486cd06e78c",
		"milvus.tar":         "11f32b75998e0330a26c3d78095f71058c7cd3f87be90ea9f88bee0c6dacedec",
		"minio-default.tar":  "2698c0a0deba71fe5480b49d6f789d669bc8e3c161c3f2cf561d438d1cfe2269",
		"minio.tar":          "6e781c34f7850869f4fa520aa6940a84ac0a3f5069534e8b99b64f6bf591356f",
	}
)

type installedBinaries struct {
	Daemon string
	CLI    string
	Clyde  string
}

func validateOptIn(value string) error {
	if value != restartAcceptanceConfirmation {
		return fmt.Errorf("%s must equal %q", restartAcceptanceOptIn, restartAcceptanceConfirmation)
	}
	return nil
}

func newRunID(now time.Time, entropy io.Reader) (string, error) {
	var suffix [4]byte
	if _, err := io.ReadFull(entropy, suffix[:]); err != nil {
		return "", fmt.Errorf("read run entropy: %w", err)
	}
	return now.UTC().Format("20060102T150405Z-") + hex.EncodeToString(suffix[:]), nil
}

func validateRunRoot(parent string, root string) error {
	cleanParent := filepath.Clean(parent)
	cleanRoot := filepath.Clean(root)
	if filepath.Dir(cleanRoot) != cleanParent {
		return fmt.Errorf("run root %q is outside exact parent %q", root, parent)
	}
	if !runIDPattern.MatchString(filepath.Base(cleanRoot)) {
		return fmt.Errorf("run root %q has invalid run id", root)
	}
	_, err := os.Lstat(cleanRoot)
	if err == nil {
		return fmt.Errorf("run root %q already exists", root)
	}
	if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("inspect run root %q: %w", root, err)
	}
	return nil
}

func requiredRestoreBytes(sizes []int64) int64 {
	return sandboxharness.RequiredBytes(sizes)
}

func validateFreeSpace(available int64, sizes []int64) error {
	return sandboxharness.RequireFreeSpace(available, sizes)
}

func validateCaseFreeSpace(available int64, sizes []int64) error {
	// Container startup can break every clonefile extent in the writable case.
	required := requiredRestoreBytes(sizes)
	if available < required {
		return fmt.Errorf("free space %d bytes is less than case reserve %d bytes", available, required)
	}
	return nil
}

func validatePorts(ports []int) error {
	seen := make(map[int]struct{}, len(ports))
	for _, port := range ports {
		if port < 1 || port > 65535 {
			return fmt.Errorf("invalid port %d", port)
		}
		if _, exists := seen[port]; exists {
			return fmt.Errorf("duplicate port %d", port)
		}
		seen[port] = struct{}{}
		listener, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
		if err != nil {
			return fmt.Errorf("port %d is bound: %w", port, err)
		}
		if err := listener.Close(); err != nil {
			return fmt.Errorf("close port %d probe: %w", port, err)
		}
	}
	return nil
}

func validateInstalledBinaries(home string) (installedBinaries, error) {
	binDirectory := filepath.Join(home, ".local", "bin")
	binaries := installedBinaries{
		Daemon: filepath.Join(binDirectory, "lm-semantic-search-daemon"),
		CLI:    filepath.Join(binDirectory, "lm-semantic-search"),
		Clyde:  filepath.Join(binDirectory, "clyde"),
	}
	for _, path := range []string{binaries.Daemon, binaries.CLI, binaries.Clyde} {
		info, err := os.Stat(path)
		if err != nil {
			return installedBinaries{}, fmt.Errorf("inspect installed binary %q: %w", path, err)
		}
		if !info.Mode().IsRegular() || info.Mode().Perm()&0o111 == 0 {
			return installedBinaries{}, fmt.Errorf("installed binary %q is not a regular executable file", path)
		}
	}
	return binaries, nil
}

func verifyChecksums(root string, expected map[string]string) error {
	return sandboxharness.VerifyChecksums(root, checksumManifestName, expected)
}

func requiredDirectoryFromEnvironment(name string) (string, error) {
	value := os.Getenv(name)
	if value == "" {
		return "", fmt.Errorf("%s must name an existing absolute directory", name)
	}
	cleanPath := filepath.Clean(value)
	if !filepath.IsAbs(cleanPath) {
		return "", fmt.Errorf("%s must name an absolute directory", name)
	}
	info, err := os.Lstat(cleanPath)
	if err != nil {
		return "", fmt.Errorf("inspect %s directory %q: %w", name, cleanPath, err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", fmt.Errorf("%s must name a real directory", name)
	}
	resolvedPath, err := filepath.EvalSymlinks(cleanPath)
	if err != nil {
		return "", fmt.Errorf("resolve %s directory %q: %w", name, cleanPath, err)
	}
	if resolvedPath != cleanPath {
		return "", fmt.Errorf("%s directory must not traverse a symbolic link", name)
	}
	return cleanPath, nil
}
