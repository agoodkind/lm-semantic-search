// Command tokenizers-dep stages the static Hugging Face tokenizer library.
package main

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
)

const (
	tokenizersVersion       = "1.27.0"
	tokenizersArchivePrefix = "https://github.com/daulet/tokenizers/releases/download/v" +
		tokenizersVersion + "/libtokenizers."
	toolLogPrefix        = "setup-cgo-tokenizers"
	defaultFileMode      = 0o644
	defaultDirectoryMode = 0o755
	maximumLibraryBytes  = 512 << 20
)

func nativeLibraryArchiveEntry() string {
	return "libtokenizers.a"
}

type targetArchive struct {
	filename string
	sha256   string
}

var targetArchives = map[string]targetArchive{
	"darwin/amd64": {
		filename: "darwin-x86_64.tar.gz",
		sha256:   "6239efe5a81fde8089ef2df8ae710366542b4e5deab6d8ecb74d7d1862db2ddb",
	},
	"darwin/arm64": {
		filename: "darwin-arm64.tar.gz",
		sha256:   "fb84b8b2e349a5952767ffe80ccd862fc44084de47f3b0cc3f0b7c9d4e649cf7",
	},
	"linux/amd64": {
		filename: "linux-amd64.tar.gz",
		sha256:   "72556cdca798dd4ea7cdaba308e5f0d68a8cb93b67c96edf485b7a0edd7b07f4",
	},
	"linux/arm64": {
		filename: "linux-arm64.tar.gz",
		sha256:   "e96545ad05930c26f51f63d932ee6d3bbd32bbed149e102c5290d587a2293067",
	},
}

type installer struct {
	prefix     string
	targetName string
	archive    targetArchive
	client     *http.Client
}

func main() {
	slog.Debug("tokenizers dependency process entry")
	os.Exit(runMain())
}

func runMain() int {
	slog.Debug("tokenizers dependency command entry")
	ctx, stop := signal.NotifyContext(
		context.Background(),
		os.Interrupt,
		syscall.SIGTERM,
	)
	defer stop()

	if err := run(ctx); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "%s: %v\n", toolLogPrefix, err)
		return 1
	}
	return 0
}

func run(ctx context.Context) error {
	rootDirectory, err := findModuleRoot()
	if err != nil {
		return err
	}
	targetName, err := resolveTargetName(ctx)
	if err != nil {
		return err
	}
	archive, found := targetArchives[targetName]
	if !found {
		return fmt.Errorf("tokenizers does not support build target %s", targetName)
	}
	prefix := os.Getenv("GO_MK_CGO_PREFIX")
	if prefix == "" {
		prefix = filepath.Join(
			rootDirectory,
			".make",
			"cgo",
			strings.ReplaceAll(targetName, "/", "-"),
		)
	} else if !filepath.IsAbs(prefix) {
		prefix = filepath.Join(rootDirectory, prefix)
	}

	dependencyInstaller := installer{
		prefix:     prefix,
		targetName: targetName,
		archive:    archive,
		client:     http.DefaultClient,
	}
	if dependencyInstaller.cached() {
		fmt.Printf(
			"%s: using cached tokenizers %s for %s\n",
			toolLogPrefix,
			tokenizersVersion,
			targetName,
		)
		return nil
	}
	if err := dependencyInstaller.install(ctx); err != nil {
		return err
	}
	fmt.Printf(
		"%s: installed static tokenizers %s for %s\n",
		toolLogPrefix,
		tokenizersVersion,
		targetName,
	)
	return nil
}

func (dependencyInstaller installer) cached() bool {
	libraryInfo, libraryErr := os.Stat(dependencyInstaller.libraryPath())
	if libraryErr != nil || !libraryInfo.Mode().IsRegular() {
		return false
	}
	version, versionErr := os.ReadFile(dependencyInstaller.versionPath())
	if versionErr != nil {
		return false
	}
	return strings.TrimSpace(string(version)) == tokenizersVersion
}

func (dependencyInstaller installer) install(ctx context.Context) error {
	slog.InfoContext(
		ctx,
		"install tokenizers dependency",
		"target",
		dependencyInstaller.targetName,
		"prefix",
		dependencyInstaller.prefix,
	)
	if err := os.MkdirAll(dependencyInstaller.prefix, defaultDirectoryMode); err != nil {
		return wrapError("create tokenizers prefix", err)
	}
	temporaryDirectory, err := os.MkdirTemp(
		dependencyInstaller.prefix,
		".tokenizers.",
	)
	if err != nil {
		return wrapError("create tokenizers temporary directory", err)
	}
	defer func() {
		_ = os.RemoveAll(temporaryDirectory)
	}()

	archivePath := filepath.Join(temporaryDirectory, "libtokenizers.tar.gz")
	archiveURL := tokenizersArchivePrefix + dependencyInstaller.archive.filename
	if err := dependencyInstaller.download(
		ctx,
		archiveURL,
		archivePath,
	); err != nil {
		return err
	}
	temporaryLibraryPath := filepath.Join(
		temporaryDirectory,
		nativeLibraryArchiveEntry(),
	)
	if err := extractLibrary(archivePath, temporaryLibraryPath); err != nil {
		return err
	}
	if err := os.MkdirAll(
		filepath.Dir(dependencyInstaller.libraryPath()),
		defaultDirectoryMode,
	); err != nil {
		return wrapError("create tokenizers library directory", err)
	}
	if err := os.Rename(
		temporaryLibraryPath,
		dependencyInstaller.libraryPath(),
	); err != nil {
		return wrapError("install tokenizers library", err)
	}
	if err := os.MkdirAll(
		filepath.Dir(dependencyInstaller.versionPath()),
		defaultDirectoryMode,
	); err != nil {
		return wrapError("create tokenizers version directory", err)
	}
	if err := os.WriteFile(
		dependencyInstaller.versionPath(),
		[]byte(tokenizersVersion+"\n"),
		defaultFileMode,
	); err != nil {
		return wrapError("write tokenizers version", err)
	}
	return nil
}

func (dependencyInstaller installer) download(
	ctx context.Context,
	url string,
	destinationPath string,
) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return wrapError("create tokenizers download request", err)
	}
	response, err := dependencyInstaller.client.Do(request)
	if err != nil {
		return wrapError("download tokenizers archive", err)
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK ||
		response.StatusCode >= http.StatusMultipleChoices {
		return fmt.Errorf(
			"download tokenizers archive: HTTP status %s",
			response.Status,
		)
	}

	destination, err := os.OpenFile(
		destinationPath,
		os.O_CREATE|os.O_WRONLY|os.O_TRUNC,
		defaultFileMode,
	)
	if err != nil {
		return wrapError("create tokenizers archive", err)
	}
	hash := sha256.New()
	_, copyErr := io.Copy(io.MultiWriter(destination, hash), response.Body)
	closeErr := destination.Close()
	if copyErr != nil {
		return wrapError("write tokenizers archive", copyErr)
	}
	if closeErr != nil {
		return wrapError("close tokenizers archive", closeErr)
	}
	actualSHA256 := hex.EncodeToString(hash.Sum(nil))
	if actualSHA256 != dependencyInstaller.archive.sha256 {
		return fmt.Errorf(
			"tokenizers archive checksum mismatch: got %s, want %s",
			actualSHA256,
			dependencyInstaller.archive.sha256,
		)
	}
	return nil
}

func extractLibrary(archivePath string, destinationPath string) error {
	archive, err := os.Open(archivePath)
	if err != nil {
		return wrapError("open tokenizers archive", err)
	}
	gzipReader, err := gzip.NewReader(archive)
	if err != nil {
		_ = archive.Close()
		return wrapError("open tokenizers gzip stream", err)
	}
	tarReader := tar.NewReader(gzipReader)
	found := false
	for {
		header, nextErr := tarReader.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			_ = gzipReader.Close()
			_ = archive.Close()
			return wrapError("read tokenizers archive", nextErr)
		}
		if header.Name != nativeLibraryArchiveEntry() {
			continue
		}
		if header.Typeflag != tar.TypeReg && header.Typeflag != 0 {
			_ = gzipReader.Close()
			_ = archive.Close()
			return errors.New("tokenizers archive library is not a regular file")
		}
		if header.Size <= 0 || header.Size > maximumLibraryBytes {
			_ = gzipReader.Close()
			_ = archive.Close()
			return fmt.Errorf(
				"tokenizers library has unsupported size %d",
				header.Size,
			)
		}
		if err := writeLibrary(tarReader, destinationPath, header.Size); err != nil {
			_ = gzipReader.Close()
			_ = archive.Close()
			return err
		}
		found = true
		break
	}
	gzipCloseErr := gzipReader.Close()
	archiveCloseErr := archive.Close()
	if gzipCloseErr != nil {
		return wrapError("close tokenizers gzip stream", gzipCloseErr)
	}
	if archiveCloseErr != nil {
		return wrapError("close tokenizers archive", archiveCloseErr)
	}
	if !found {
		return errors.New("tokenizers archive is missing libtokenizers.a")
	}
	return nil
}

func writeLibrary(reader io.Reader, destinationPath string, size int64) error {
	destination, err := os.OpenFile(
		destinationPath,
		os.O_CREATE|os.O_WRONLY|os.O_TRUNC,
		defaultFileMode,
	)
	if err != nil {
		return wrapError("create tokenizers library", err)
	}
	written, copyErr := io.CopyN(destination, reader, size)
	closeErr := destination.Close()
	if copyErr != nil {
		return wrapError("extract tokenizers library", copyErr)
	}
	if closeErr != nil {
		return wrapError("close tokenizers library", closeErr)
	}
	if written != size {
		return fmt.Errorf(
			"tokenizers library size is %d, want %d",
			written,
			size,
		)
	}
	return nil
}

func (dependencyInstaller installer) libraryPath() string {
	return filepath.Join(
		dependencyInstaller.prefix,
		"lib",
		nativeLibraryArchiveEntry(),
	)
}

func (dependencyInstaller installer) versionPath() string {
	return filepath.Join(
		dependencyInstaller.prefix,
		"share",
		"tokenizers",
		"version",
	)
}

func resolveTargetName(ctx context.Context) (string, error) {
	targetGOOS := os.Getenv("GO_MK_TARGET_GOOS")
	if targetGOOS == "" {
		value, err := readGoEnvironment(ctx, "GOOS")
		if err != nil {
			return "", err
		}
		targetGOOS = value
	}
	targetGOARCH := os.Getenv("GO_MK_TARGET_GOARCH")
	if targetGOARCH == "" {
		value, err := readGoEnvironment(ctx, "GOARCH")
		if err != nil {
			return "", err
		}
		targetGOARCH = value
	}
	return targetGOOS + "/" + targetGOARCH, nil
}

func readGoEnvironment(ctx context.Context, name string) (string, error) {
	slog.DebugContext(ctx, "read Go environment", "name", name)
	command := exec.CommandContext(ctx, "go", "env", name)
	output, err := command.Output()
	if err != nil {
		return "", wrapError("read go env "+name, err)
	}
	value := strings.TrimSpace(string(output))
	if value == "" {
		return "", fmt.Errorf("read go env %s: empty value", name)
	}
	return value, nil
}

func findModuleRoot() (string, error) {
	slog.Debug("find Go module root")
	currentDirectory, err := os.Getwd()
	if err != nil {
		return "", wrapError("get current directory", err)
	}
	for {
		modulePath := filepath.Join(currentDirectory, "go.mod")
		fileInfo, statErr := os.Stat(modulePath)
		if statErr == nil && fileInfo.Mode().IsRegular() {
			return currentDirectory, nil
		}
		if statErr != nil && !errors.Is(statErr, os.ErrNotExist) {
			return "", wrapError("inspect "+modulePath, statErr)
		}
		parentDirectory := filepath.Dir(currentDirectory)
		if parentDirectory == currentDirectory {
			return "", errors.New("find module root: go.mod is missing")
		}
		currentDirectory = parentDirectory
	}
}

func wrapError(operation string, err error) error {
	slog.Error(operation+" failed", "err", err)
	return fmt.Errorf("%s: %w", operation, err)
}
