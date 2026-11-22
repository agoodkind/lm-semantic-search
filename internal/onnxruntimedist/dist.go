// Package onnxruntimedist installs the pinned ONNX Runtime shared library
// and headers from Microsoft's NuGet package.
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
)

const (
	// Version is the pinned ONNX Runtime release.
	Version = "1.32.0"

	packageURL = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/" + Version +
		"/microsoft.ml.onnxruntime." + Version + ".nupkg"
	packageSHA256 = "a9150bc2992cbbb6722070da9910e1ffa989a8bfc67bdd5bb8e745ca755641db"

	darwinLibraryName          = "libonnxruntime.dylib"
	darwinLibrarySONAME        = "libonnxruntime.1.dylib"
	darwinVersionedLibraryName = "libonnxruntime.1.32.0.dylib"
	linuxLibraryName           = "libonnxruntime.so"
	linuxLibrarySONAME         = "libonnxruntime.so.1"
	linuxVersionedLibraryName  = "libonnxruntime.so.1.32.0"
	defaultDirectoryMode       = 0o755
	maxExtractedFileSize       = 2 << 30

	operatingSystemDarwin = "darwin"
	operatingSystemLinux  = "linux"
	architectureAMD64     = "amd64"
	architectureARM64     = "arm64"
)

// Archive identifies the pinned package and library for one platform.
type Archive struct {
	URL    string
	SHA256 string

	libraryPath string
	libraryName string
}

// LibraryNames contains the versioned library file name and both symlink names.
type LibraryNames struct {
	Versioned   string
	SONAME      string
	Unversioned string
}

var (
	errUnsupportedGOOS   = errors.New("unsupported GOOS")
	errUnsupportedGOARCH = errors.New("unsupported GOARCH")
	errDownloadStatus    = errors.New("unexpected HTTP status")
	errChecksumMismatch  = errors.New("checksum mismatch")
)

// ArchiveFor returns the pinned package and the library entry for a platform.
//
// Microsoft uses its NuGet runtime identifiers for library directories.
//
// Example:
//
//	ArchiveFor("linux", "amd64")
func ArchiveFor(goos string, goarch string) (Archive, error) {
	var runtimeID string

	switch goos {
	case operatingSystemDarwin:
		if goarch != architectureARM64 {
			return Archive{}, wrapError("select library for darwin/"+goarch, errUnsupportedGOARCH)
		}

		runtimeID = "osx-arm64"
	case operatingSystemLinux:
		switch goarch {
		case architectureAMD64:
			runtimeID = "linux-x64"
		case architectureARM64:
			runtimeID = "linux-arm64"
		default:
			return Archive{}, wrapError("select library for linux/"+goarch, errUnsupportedGOARCH)
		}
	default:
		return Archive{}, wrapError("select library for "+goos, errUnsupportedGOOS)
	}

	names, err := LibraryNamesFor(goos)
	if err != nil {
		return Archive{}, err
	}

	return Archive{
		URL:         packageURL,
		SHA256:      packageSHA256,
		libraryPath: "runtimes/" + runtimeID + "/native/" + names.Unversioned,
		libraryName: names.Versioned,
	}, nil
}

// LibraryNamesFor returns the shared library file names for a GOOS.
//
// Installers use the versioned name for the file and the other names for
// symlinks to that file.
//
// Linux layout:
//   - libonnxruntime.so.1.32.0
//   - libonnxruntime.so.1
//   - libonnxruntime.so
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
		return LibraryNames{}, wrapError("select library names for "+goos, errUnsupportedGOOS)
	}
}

// StageSharedLibrary downloads, verifies, and writes the versioned library.
//
// The installer can update the SONAME and unversioned symlinks separately.
//
// Example output on Linux:
//   - libonnxruntime.so.1.32.0
func StageSharedLibrary(
	ctx context.Context,
	httpClient *http.Client,
	archive Archive,
	names LibraryNames,
	directory string,
) error {
	slog.DebugContext(ctx, "stage ONNX Runtime shared library", "directory", directory)
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

	sourceLibraryPath := filepath.Join(archiveDirectory, "lib", names.Versioned)
	if err := copyFileReplacing(
		sourceLibraryPath,
		filepath.Join(directory, names.Versioned),
	); err != nil {
		return wrapError("copy ONNX Runtime shared library", err)
	}

	return nil
}

// FetchArchive downloads and verifies the pinned ONNX Runtime package.
//
// It extracts the selected library and the public headers into a new
// subdirectory of workDirectory. Pass that directory to [InstallLibrary].
//
// Extracted layout:
//   - lib/ contains the selected shared library.
//   - include/ contains the public headers.
func FetchArchive(
	ctx context.Context,
	httpClient *http.Client,
	archive Archive,
	workDirectory string,
) (string, error) {
	archivePath := filepath.Join(workDirectory, "onnxruntime.nupkg")
	if err := downloadAndVerify(
		ctx,
		httpClient,
		archive.URL,
		archive.SHA256,
		archivePath,
	); err != nil {
		return "", err
	}

	extractedDirectory, err := os.MkdirTemp(workDirectory, "onnxruntime.")
	if err != nil {
		return "", wrapError("create ONNX Runtime extraction directory", err)
	}

	if err := extractPackage(archivePath, extractedDirectory, archive); err != nil {
		return "", wrapError("extract ONNX Runtime package", err)
	}

	return extractedDirectory, nil
}

// InstallLibrary copies the versioned library and creates relative symlinks.
//
// It renames each file into place. A running daemon keeps the library that
// it already mapped.
//
// Installed layout:
//   - names.Versioned contains the library.
//   - names.SONAME and names.Unversioned link to names.Versioned.
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
		return wrapError("download "+url+" ("+response.Status+")", errDownloadStatus)
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
		operation := fmt.Sprintf("verify %s (got %s, want %s)", path, actual, expected)

		return wrapError(operation, errChecksumMismatch)
	}

	return nil
}
