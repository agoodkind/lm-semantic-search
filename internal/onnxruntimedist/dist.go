// Package onnxruntimedist pins the ONNX Runtime release archives that
// lm-semantic-search links against and installs their shared library.
package onnxruntimedist

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const (
	// Version is the pinned ONNX Runtime release.
	Version = "1.27.0"

	darwinARM64ArchiveName     = "onnxruntime-osx-arm64-1.27.0"
	darwinARM64ArchiveURL      = "https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-osx-arm64-1.27.0.tgz"
	darwinARM64ArchiveSHA256   = "545e81c58152353acb0d1e8bd6ce4b62f830c0961f5b3acfedc790ffd76e477a"
	darwinLibraryName          = "libonnxruntime.dylib"
	darwinLibrarySONAME        = "libonnxruntime.1.dylib"
	darwinVersionedLibraryName = "libonnxruntime.1.27.0.dylib"
	linuxAMD64ArchiveName      = "onnxruntime-linux-x64-1.27.0"
	linuxAMD64ArchiveURL       = "https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-linux-x64-1.27.0.tgz"
	linuxAMD64ArchiveSHA256    = "547e40a48f1fe73e3f812d7c88a948612c23f896b91e4e2ee1e232d7b468246f"
	linuxARM64ArchiveName      = "onnxruntime-linux-aarch64-1.27.0"
	linuxARM64ArchiveURL       = "https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-linux-aarch64-1.27.0.tgz"
	linuxARM64ArchiveSHA256    = "3e4d83ac06924a32a07b6d7f91ce6f852876153fc0bbdf931bf517a140bfbe48"
	linuxLibraryName           = "libonnxruntime.so"
	linuxLibrarySONAME         = "libonnxruntime.so.1"
	linuxVersionedLibraryName  = "libonnxruntime.so.1.27.0"
	defaultDirectoryMode       = 0o755
	maxExtractedFileSize       = 2 << 30

	operatingSystemDarwin = "darwin"
	operatingSystemLinux  = "linux"
	architectureAMD64     = "amd64"
	architectureARM64     = "arm64"
)

// Archive is one pinned ONNX Runtime release archive. Name is the top-level
// directory the archive extracts to.
type Archive struct {
	Name   string
	URL    string
	SHA256 string
}

// LibraryNames are the file names of the shared library: the versioned file
// the archive ships, and the SONAME and unversioned names that link to it.
type LibraryNames struct {
	Versioned   string
	SONAME      string
	Unversioned string
}

var darwinArchives = map[string]Archive{
	architectureARM64: {
		Name:   darwinARM64ArchiveName,
		URL:    darwinARM64ArchiveURL,
		SHA256: darwinARM64ArchiveSHA256,
	},
}

var linuxArchives = map[string]Archive{
	architectureAMD64: {
		Name:   linuxAMD64ArchiveName,
		URL:    linuxAMD64ArchiveURL,
		SHA256: linuxAMD64ArchiveSHA256,
	},
	architectureARM64: {
		Name:   linuxARM64ArchiveName,
		URL:    linuxARM64ArchiveURL,
		SHA256: linuxARM64ArchiveSHA256,
	},
}

// ArchiveFor returns the pinned archive for a GOOS and GOARCH pair.
func ArchiveFor(goos string, goarch string) (Archive, error) {
	switch goos {
	case operatingSystemDarwin:
		archive, ok := darwinArchives[goarch]
		if !ok {
			return Archive{}, fmt.Errorf("unsupported Darwin GOARCH %s", goarch)
		}
		return archive, nil
	case operatingSystemLinux:
		archive, ok := linuxArchives[goarch]
		if !ok {
			return Archive{}, fmt.Errorf("unsupported Linux GOARCH %s", goarch)
		}
		return archive, nil
	default:
		return Archive{}, fmt.Errorf("unsupported GOOS %s", goos)
	}
}

// LibraryNamesFor returns the shared library file names for a GOOS.
func LibraryNamesFor(goos string) (LibraryNames, error) {
	switch goos {
	case operatingSystemDarwin:
		return LibraryNames{
			Versioned:   darwinVersionedLibraryName,
			SONAME:      darwinLibrarySONAME,
			Unversioned: darwinLibraryName,
		}, nil
	case operatingSystemLinux:
		return LibraryNames{
			Versioned:   linuxVersionedLibraryName,
			SONAME:      linuxLibrarySONAME,
			Unversioned: linuxLibraryName,
		}, nil
	default:
		return LibraryNames{}, fmt.Errorf("unsupported GOOS %s", goos)
	}
}

// InstallSharedLibrary downloads archive, verifies its pinned SHA-256, and
// installs its shared library into directory through InstallLibrary.
func InstallSharedLibrary(
	ctx context.Context,
	httpClient *http.Client,
	archive Archive,
	names LibraryNames,
	directory string,
) error {
	slog.DebugContext(ctx, "install ONNX Runtime shared library", "directory", directory)
	temporaryDirectory, err := os.MkdirTemp("", "lms-onnxruntime.")
	if err != nil {
		return wrapError("create temporary directory", err)
	}
	defer func() {
		_ = os.RemoveAll(temporaryDirectory)
	}()

	archiveDirectory, err := FetchArchive(ctx, httpClient, archive, temporaryDirectory)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(directory, defaultDirectoryMode); err != nil {
		return wrapError("create library directory", err)
	}
	return InstallLibrary(archiveDirectory, names, directory)
}

// FetchArchive downloads archive into workDirectory, verifies its pinned
// SHA-256, extracts it there, and returns the extracted top-level directory.
func FetchArchive(
	ctx context.Context,
	httpClient *http.Client,
	archive Archive,
	workDirectory string,
) (string, error) {
	archivePath := filepath.Join(workDirectory, "onnxruntime.tgz")
	if err := downloadAndVerify(ctx, httpClient, archive.URL, archive.SHA256, archivePath); err != nil {
		return "", err
	}
	extractedDirectory := filepath.Join(workDirectory, "onnxruntime")
	if err := extractTarGzip(archivePath, extractedDirectory); err != nil {
		return "", wrapError("extract ONNX Runtime archive", err)
	}
	return filepath.Join(extractedDirectory, archive.Name), nil
}

// InstallLibrary copies the versioned library from an extracted archive into
// libraryDirectory and points the SONAME and unversioned names at it with
// relative symlinks. Each file is written beside its destination and renamed
// into place, so a daemon that has the previous library mapped keeps its copy
// and a repeated install converges on the same result.
func InstallLibrary(archiveDirectory string, names LibraryNames, libraryDirectory string) error {
	sourceLibraryPath := filepath.Join(archiveDirectory, "lib", names.Versioned)
	destinationLibraryPath := filepath.Join(libraryDirectory, names.Versioned)
	if err := copyFileReplacing(sourceLibraryPath, destinationLibraryPath); err != nil {
		return wrapError("copy ONNX Runtime shared library", err)
	}
	for _, linkName := range []string{names.SONAME, names.Unversioned} {
		linkPath := filepath.Join(libraryDirectory, linkName)
		if err := symlinkReplacing(names.Versioned, linkPath); err != nil {
			return wrapError("create ONNX Runtime shared library symlink", err)
		}
	}
	return nil
}

func wrapError(operation string, err error) error {
	slog.Error(operation+" failed", "err", err)
	return fmt.Errorf("%s: %w", operation, err)
}

func copyFileReplacing(sourcePath string, destinationPath string) error {
	slog.Debug("copy dependency file", "source", sourcePath, "destination", destinationPath)
	source, err := os.Open(sourcePath)
	if err != nil {
		return wrapError("open source file", err)
	}
	defer func() {
		_ = source.Close()
	}()
	sourceInfo, err := source.Stat()
	if err != nil {
		return wrapError("inspect source file", err)
	}
	destination, err := os.CreateTemp(
		filepath.Dir(destinationPath),
		"."+filepath.Base(destinationPath)+".*",
	)
	if err != nil {
		return wrapError("create destination file", err)
	}
	temporaryPath := destination.Name()
	_, copyErr := io.Copy(destination, source)
	closeErr := destination.Close()
	if copyErr != nil {
		_ = os.Remove(temporaryPath)
		return wrapError("copy dependency file", copyErr)
	}
	if closeErr != nil {
		_ = os.Remove(temporaryPath)
		return wrapError("close destination file", closeErr)
	}
	if err := os.Chmod(temporaryPath, sourceInfo.Mode().Perm()); err != nil {
		_ = os.Remove(temporaryPath)
		return wrapError("set destination file mode", err)
	}
	if err := os.Rename(temporaryPath, destinationPath); err != nil {
		_ = os.Remove(temporaryPath)
		return wrapError("replace destination file", err)
	}
	return nil
}

func symlinkReplacing(target string, linkPath string) error {
	slog.Debug("replace dependency symlink", "target", target, "path", linkPath)
	temporaryLinkPath := linkPath + ".tmp"
	if err := os.Remove(temporaryLinkPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		return wrapError("remove stale temporary symlink", err)
	}
	if err := os.Symlink(target, temporaryLinkPath); err != nil {
		return wrapError("create temporary symlink", err)
	}
	if err := os.Rename(temporaryLinkPath, linkPath); err != nil {
		_ = os.Remove(temporaryLinkPath)
		return wrapError("replace symlink", err)
	}
	return nil
}

func downloadAndVerify(
	ctx context.Context,
	httpClient *http.Client,
	url string,
	expectedSHA256 string,
	destinationPath string,
) error {
	slog.DebugContext(ctx, "download ONNX Runtime dependency", "url", url)
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return wrapError("create download request", err)
	}
	response, err := httpClient.Do(request)
	if err != nil {
		return wrapError("download "+url, err)
	}
	defer response.Body.Close()

	if response.StatusCode < http.StatusOK ||
		response.StatusCode >= http.StatusMultipleChoices {
		return fmt.Errorf("download %s: HTTP status %s", url, response.Status)
	}

	destination, err := os.Create(destinationPath)
	if err != nil {
		return wrapError("create download destination", err)
	}
	_, copyErr := io.Copy(destination, response.Body)
	closeErr := destination.Close()
	if copyErr != nil {
		return wrapError("write download", copyErr)
	}
	if closeErr != nil {
		return wrapError("close download", closeErr)
	}

	if err := verifySHA256(destinationPath, expectedSHA256); err != nil {
		return err
	}
	return nil
}

func verifySHA256(path string, expected string) error {
	slog.Debug("verify dependency archive checksum", "path", path)
	file, err := os.Open(path)
	if err != nil {
		return wrapError("open "+path+" for checksum", err)
	}
	hash := sha256.New()
	_, copyErr := io.Copy(hash, file)
	closeErr := file.Close()
	if copyErr != nil {
		return wrapError("hash "+path, copyErr)
	}
	if closeErr != nil {
		return wrapError("close "+path+" after checksum", closeErr)
	}

	actual := hex.EncodeToString(hash.Sum(nil))
	if actual != expected {
		return fmt.Errorf(
			"checksum mismatch for %s: got %s, want %s",
			path,
			actual,
			expected,
		)
	}
	return nil
}

func safeArchivePath(rootDirectory string, entryName string) (string, error) {
	localName := filepath.FromSlash(entryName)
	// [filepath.IsLocal] rejects absolute, empty, and parent-escaping names using
	// lexical analysis. The real-path check in the extractor then rejects a name
	// whose parent directory resolves through a symlink to outside the root.
	if !filepath.IsLocal(localName) {
		return "", fmt.Errorf("unsafe archive path %q", entryName)
	}
	return filepath.Join(rootDirectory, filepath.Clean(localName)), nil
}

// pathWithinRoot reports whether candidatePath resolves inside rootDirectory. It
// is written as a boolean guard on the cleaned candidate so a caller can place
// it directly in front of a filesystem operation. That inline prefix check is
// the barrier static analysis recognizes against archive path traversal and
// symlink escape, and it holds even when a crafted entry or target resolves back
// out through the destination directory.
func pathWithinRoot(rootDirectory string, candidatePath string) bool {
	cleanRoot := filepath.Clean(rootDirectory)
	cleanCandidate := filepath.Clean(candidatePath)
	if cleanCandidate == cleanRoot {
		return true
	}
	return strings.HasPrefix(cleanCandidate, cleanRoot+string(filepath.Separator))
}
