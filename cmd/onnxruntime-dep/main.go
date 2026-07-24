// Command onnxruntime-dep stages ONNX Runtime for cgo.
package main

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
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
)

const (
	onnxRuntimeVersion        = "1.27.0"
	appleArchiveURL           = "https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.27.0.zip"
	appleArchiveSHA256        = "8c74edd600eafc3055de9e8f7a9602afee44ed516913cb5e132bca02cc34622c"
	linuxAMD64ArchiveName     = "onnxruntime-linux-x64-1.27.0"
	linuxAMD64ArchiveURL      = "https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-linux-x64-1.27.0.tgz"
	linuxAMD64ArchiveSHA256   = "547e40a48f1fe73e3f812d7c88a948612c23f896b91e4e2ee1e232d7b468246f"
	linuxARM64ArchiveName     = "onnxruntime-linux-aarch64-1.27.0"
	linuxARM64ArchiveURL      = "https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-linux-aarch64-1.27.0.tgz"
	linuxARM64ArchiveSHA256   = "3e4d83ac06924a32a07b6d7f91ce6f852876153fc0bbdf931bf517a140bfbe48"
	linuxLibraryName          = "libonnxruntime.so"
	linuxLibrarySONAME        = "libonnxruntime.so.1"
	linuxVersionedLibraryName = "libonnxruntime.so.1.27.0"
	toolLogPrefix             = "setup-cgo-onnxruntime"
	defaultFileMode           = 0o644
	defaultDirectoryMode      = 0o755
	maxExtractedFileSize      = 2 << 30
	maxSymlinkTargetSize      = 1 << 20
)

type operatingSystem string

const (
	operatingSystemDarwin operatingSystem = "darwin"
	operatingSystemLinux  operatingSystem = "linux"
)

type architecture string

const (
	architectureAMD64 architecture = "amd64"
	architectureARM64 architecture = "arm64"
)

type buildTarget struct {
	goos   operatingSystem
	goarch architecture
}

type linuxArchive struct {
	archiveName string
	url         string
	sha256      string
}

var linuxArchives = map[architecture]linuxArchive{
	architectureAMD64: {
		archiveName: linuxAMD64ArchiveName,
		url:         linuxAMD64ArchiveURL,
		sha256:      linuxAMD64ArchiveSHA256,
	},
	architectureARM64: {
		archiveName: linuxARM64ArchiveName,
		url:         linuxARM64ArchiveURL,
		sha256:      linuxARM64ArchiveSHA256,
	},
}

type dependencyInstaller struct {
	prefix     string
	target     buildTarget
	httpClient *http.Client
}

func main() {
	slog.Debug("onnxruntime dependency command entry")
	os.Exit(runMain())
}

func runMain() int {
	slog.Debug("onnxruntime dependency setup starting")
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
	slog.DebugContext(ctx, "resolve ONNX Runtime dependency setup")
	rootDirectory, err := findModuleRoot()
	if err != nil {
		return err
	}
	target, err := resolveBuildTarget(ctx)
	if err != nil {
		return err
	}
	prefix := resolvePrefix(rootDirectory, target)

	installer := dependencyInstaller{
		prefix:     prefix,
		target:     target,
		httpClient: http.DefaultClient,
	}

	cached, err := installer.isCached()
	if err != nil {
		return err
	}
	if cached {
		fmt.Printf(
			"%s: using cached ONNX Runtime %s for %s/%s\n",
			toolLogPrefix,
			onnxRuntimeVersion,
			target.goos,
			target.goarch,
		)
		return nil
	}

	temporaryDirectory, err := os.MkdirTemp("", "lms-onnxruntime.")
	if err != nil {
		return wrapError("create temporary directory", err)
	}
	defer func() {
		_ = os.RemoveAll(temporaryDirectory)
	}()

	if err := installer.preparePrefix(); err != nil {
		return err
	}
	if err := installer.install(ctx, temporaryDirectory); err != nil {
		return err
	}
	if err := os.WriteFile(
		installer.versionFile(),
		[]byte(onnxRuntimeVersion+"\n"),
		defaultFileMode,
	); err != nil {
		return wrapError("write version file", err)
	}

	fmt.Printf(
		"%s: installed ONNX Runtime %s for %s/%s\n",
		toolLogPrefix,
		onnxRuntimeVersion,
		target.goos,
		target.goarch,
	)
	return nil
}

func wrapError(operation string, err error) error {
	slog.Error(operation+" failed", "err", err)
	return fmt.Errorf("%s: %w", operation, err)
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

func resolveBuildTarget(ctx context.Context) (buildTarget, error) {
	targetGOOS := os.Getenv("GO_MK_TARGET_GOOS")
	if targetGOOS == "" {
		value, err := readGoEnvironment(ctx, "GOOS")
		if err != nil {
			return buildTarget{}, err
		}
		targetGOOS = value
	}

	targetGOARCH := os.Getenv("GO_MK_TARGET_GOARCH")
	if targetGOARCH == "" {
		value, err := readGoEnvironment(ctx, "GOARCH")
		if err != nil {
			return buildTarget{}, err
		}
		targetGOARCH = value
	}

	return buildTarget{
		goos:   operatingSystem(targetGOOS),
		goarch: architecture(targetGOARCH),
	}, nil
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

func resolvePrefix(rootDirectory string, target buildTarget) string {
	prefix := os.Getenv("GO_MK_CGO_PREFIX")
	if prefix == "" {
		return filepath.Join(
			rootDirectory,
			".make",
			"cgo",
			string(target.goos)+"-"+string(target.goarch),
		)
	}
	if filepath.IsAbs(prefix) {
		return prefix
	}
	return rootDirectory + string(filepath.Separator) + prefix
}

func (installer dependencyInstaller) versionFile() string {
	return filepath.Join(installer.prefix, "share", "onnxruntime", "version")
}

func (installer dependencyInstaller) pkgConfigFile() string {
	return filepath.Join(installer.prefix, "lib", "pkgconfig", "onnxruntime.pc")
}

func (installer dependencyInstaller) isCached() (bool, error) {
	slog.Debug("inspect ONNX Runtime dependency cache", "prefix", installer.prefix)
	versionExists, err := regularFileExists(installer.versionFile())
	if err != nil {
		return false, err
	}
	pkgConfigExists, err := regularFileExists(installer.pkgConfigFile())
	if err != nil {
		return false, err
	}
	if !versionExists || !pkgConfigExists {
		return false, nil
	}
	libraryPaths := installer.cachedLibraryPaths()
	if len(libraryPaths) == 0 {
		return false, nil
	}
	for _, path := range libraryPaths {
		libraryExists, libraryErr := regularFileExists(path)
		if libraryErr != nil {
			return false, libraryErr
		}
		if !libraryExists {
			return false, nil
		}
	}

	versionContents, err := os.ReadFile(installer.versionFile())
	if err != nil {
		return false, wrapError("read version file", err)
	}
	version := strings.TrimRight(string(versionContents), "\n")
	return version == onnxRuntimeVersion, nil
}

func (installer dependencyInstaller) cachedLibraryPaths() []string {
	switch installer.target.goos {
	case operatingSystemDarwin:
		return []string{
			filepath.Join(installer.prefix, "lib", "libonnxruntime.a"),
		}
	case operatingSystemLinux:
		return []string{
			filepath.Join(installer.prefix, "lib", linuxVersionedLibraryName),
			filepath.Join(installer.prefix, "lib", linuxLibrarySONAME),
			filepath.Join(installer.prefix, "lib", linuxLibraryName),
		}
	default:
		return nil
	}
}

func regularFileExists(path string) (bool, error) {
	slog.Debug("inspect regular file", "path", path)
	fileInfo, err := os.Stat(path)
	if err == nil {
		return fileInfo.Mode().IsRegular(), nil
	}
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	return false, wrapError("inspect "+path, err)
}

func (installer dependencyInstaller) preparePrefix() error {
	slog.Debug("prepare ONNX Runtime dependency prefix", "prefix", installer.prefix)
	directoriesToRemove := []string{
		filepath.Join(installer.prefix, "share", "onnxruntime"),
		filepath.Join(installer.prefix, "src"),
		filepath.Join(installer.prefix, "build"),
	}
	for _, path := range directoriesToRemove {
		if err := os.RemoveAll(path); err != nil {
			return wrapError("remove "+path, err)
		}
	}

	filesToRemove := []string{
		filepath.Join(installer.prefix, "lib", "libonnxruntime.a"),
		filepath.Join(installer.prefix, "lib", linuxVersionedLibraryName),
		filepath.Join(installer.prefix, "lib", linuxLibrarySONAME),
		filepath.Join(installer.prefix, "lib", linuxLibraryName),
		installer.pkgConfigFile(),
	}
	for _, path := range filesToRemove {
		if err := removeFileIfPresent(path); err != nil {
			return err
		}
	}

	linuxArchives, err := filepath.Glob(
		filepath.Join(installer.prefix, "lib", "onnxruntime-*.a"),
	)
	if err != nil {
		return wrapError("find staged Linux archives", err)
	}
	for _, path := range linuxArchives {
		if err := removeFileIfPresent(path); err != nil {
			return err
		}
	}

	directoriesToCreate := []string{
		filepath.Join(installer.prefix, "include"),
		filepath.Join(installer.prefix, "lib", "pkgconfig"),
		filepath.Join(installer.prefix, "share", "onnxruntime"),
	}
	for _, path := range directoriesToCreate {
		if err := os.MkdirAll(path, defaultDirectoryMode); err != nil {
			return wrapError("create "+path, err)
		}
	}
	return nil
}

func removeFileIfPresent(path string) error {
	slog.Debug("remove staged dependency file", "path", path)
	err := os.Remove(path)
	if err == nil || errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return wrapError("remove "+path, err)
}

func (installer dependencyInstaller) install(
	ctx context.Context,
	temporaryDirectory string,
) error {
	switch installer.target.goos {
	case operatingSystemDarwin:
		switch installer.target.goarch {
		case architectureARM64, architectureAMD64:
			return installer.installAppleArchive(ctx, temporaryDirectory)
		default:
			return fmt.Errorf(
				"unsupported Darwin GOARCH %s",
				installer.target.goarch,
			)
		}
	case operatingSystemLinux:
		archive, ok := linuxArchives[installer.target.goarch]
		if !ok {
			return fmt.Errorf(
				"unsupported Linux GOARCH %s",
				installer.target.goarch,
			)
		}
		return installer.installLinuxSharedArchive(ctx, temporaryDirectory, archive)
	default:
		return fmt.Errorf("unsupported GOOS %s", installer.target.goos)
	}
}

func (installer dependencyInstaller) installAppleArchive(
	ctx context.Context,
	temporaryDirectory string,
) error {
	slog.DebugContext(ctx, "install Apple ONNX Runtime archive")
	archivePath := filepath.Join(temporaryDirectory, "onnxruntime-c.zip")
	if err := installer.downloadAndVerify(
		ctx,
		appleArchiveURL,
		appleArchiveSHA256,
		archivePath,
	); err != nil {
		return err
	}

	extractedDirectory := filepath.Join(temporaryDirectory, "onnxruntime-c")
	if err := extractZip(archivePath, extractedDirectory); err != nil {
		return wrapError("extract Apple archive", err)
	}

	frameworkArchive := filepath.Join(
		extractedDirectory,
		"onnxruntime.xcframework",
		"macos-arm64_x86_64",
		"onnxruntime.framework",
		"Versions",
		"A",
		"onnxruntime",
	)
	exists, err := regularFileExists(frameworkArchive)
	if err != nil {
		return err
	}
	if !exists {
		return errors.New("macOS static framework archive is missing")
	}

	libraryPath := filepath.Join(installer.prefix, "lib", "libonnxruntime.a")
	if err := copyFile(frameworkArchive, libraryPath); err != nil {
		return wrapError("copy Apple static library", err)
	}

	headersDirectory := filepath.Join(extractedDirectory, "Headers")
	if err := copyHeaderFiles(
		headersDirectory,
		filepath.Join(installer.prefix, "include"),
	); err != nil {
		return wrapError("copy Apple headers", err)
	}
	return installer.writeApplePkgConfig()
}

func (installer dependencyInstaller) installLinuxSharedArchive(
	ctx context.Context,
	temporaryDirectory string,
	archive linuxArchive,
) error {
	slog.DebugContext(ctx, "install Linux ONNX Runtime shared archive")
	archivePath := filepath.Join(temporaryDirectory, "onnxruntime.tgz")
	if err := installer.downloadAndVerify(
		ctx,
		archive.url,
		archive.sha256,
		archivePath,
	); err != nil {
		return err
	}

	extractedDirectory := filepath.Join(temporaryDirectory, "onnxruntime")
	if err := extractTarGzip(archivePath, extractedDirectory); err != nil {
		return wrapError("extract Linux archive", err)
	}

	archiveDirectory := filepath.Join(extractedDirectory, archive.archiveName)
	sourceLibraryPath := filepath.Join(
		archiveDirectory,
		"lib",
		linuxVersionedLibraryName,
	)
	destinationLibraryPath := filepath.Join(
		installer.prefix,
		"lib",
		linuxVersionedLibraryName,
	)
	if err := copyFile(sourceLibraryPath, destinationLibraryPath); err != nil {
		return wrapError("copy Linux shared library", err)
	}
	for _, linkName := range []string{linuxLibrarySONAME, linuxLibraryName} {
		linkPath := filepath.Join(installer.prefix, "lib", linkName)
		if err := os.Symlink(linuxVersionedLibraryName, linkPath); err != nil {
			return wrapError("create Linux shared library symlink", err)
		}
	}

	if err := copyHeaderFiles(
		filepath.Join(archiveDirectory, "include"),
		filepath.Join(installer.prefix, "include"),
	); err != nil {
		return wrapError("copy Linux headers", err)
	}
	return installer.writeLinuxPkgConfig()
}

func (installer dependencyInstaller) downloadAndVerify(
	ctx context.Context,
	url string,
	expectedSHA256 string,
	destinationPath string,
) error {
	slog.DebugContext(ctx, "download ONNX Runtime dependency", "url", url)
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return wrapError("create download request", err)
	}
	response, err := installer.httpClient.Do(request)
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
	cleanName := filepath.Clean(filepath.FromSlash(entryName))
	if cleanName == "." || filepath.IsAbs(cleanName) {
		return "", fmt.Errorf("unsafe archive path %q", entryName)
	}
	parentPrefix := ".." + string(filepath.Separator)
	if cleanName == ".." || strings.HasPrefix(cleanName, parentPrefix) {
		return "", fmt.Errorf("unsafe archive path %q", entryName)
	}
	return filepath.Join(rootDirectory, cleanName), nil
}

func copyHeaderFiles(sourceDirectory string, destinationDirectory string) error {
	slog.Debug("copy ONNX Runtime headers", "source", sourceDirectory)
	entries, err := os.ReadDir(sourceDirectory)
	if err != nil {
		return wrapError("read header directory", err)
	}
	headerCount := 0
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".h" {
			continue
		}
		sourcePath := filepath.Join(sourceDirectory, entry.Name())
		destinationPath := filepath.Join(destinationDirectory, entry.Name())
		if err := copyFile(sourcePath, destinationPath); err != nil {
			return err
		}
		headerCount++
	}
	if headerCount == 0 {
		return fmt.Errorf("no header files found in %s", sourceDirectory)
	}
	return nil
}

func copyFile(sourcePath string, destinationPath string) error {
	slog.Debug(
		"copy dependency file",
		"source",
		sourcePath,
		"destination",
		destinationPath,
	)
	source, err := os.Open(sourcePath)
	if err != nil {
		return wrapError("open source file", err)
	}
	sourceInfo, err := source.Stat()
	if err != nil {
		_ = source.Close()
		return wrapError("inspect source file", err)
	}
	destination, err := os.OpenFile(
		destinationPath,
		os.O_CREATE|os.O_WRONLY|os.O_TRUNC,
		sourceInfo.Mode().Perm(),
	)
	if err != nil {
		_ = source.Close()
		return wrapError("create destination file", err)
	}

	_, copyErr := io.Copy(destination, source)
	destinationCloseErr := destination.Close()
	sourceCloseErr := source.Close()
	if copyErr != nil {
		return wrapError("copy dependency file", copyErr)
	}
	if destinationCloseErr != nil {
		return wrapError("close destination file", destinationCloseErr)
	}
	if sourceCloseErr != nil {
		return wrapError("close source file", sourceCloseErr)
	}
	return nil
}

func (installer dependencyInstaller) writeApplePkgConfig() error {
	slog.Debug("write Apple ONNX Runtime pkg-config")
	contents := fmt.Sprintf(`prefix=%s
exec_prefix=${prefix}
libdir=${prefix}/lib
includedir=${prefix}/include

Name: onnxruntime
Description: statically linked ONNX Runtime
Version: %s
Cflags: -I${includedir}
Libs: -L${libdir} -lonnxruntime -ltokenizers -lc++ -framework CoreML -framework Foundation -framework Accelerate
`, installer.prefix, onnxRuntimeVersion)
	if err := os.WriteFile(
		installer.pkgConfigFile(),
		[]byte(contents),
		defaultFileMode,
	); err != nil {
		return wrapError("write Apple pkg-config file", err)
	}
	return nil
}

func (installer dependencyInstaller) writeLinuxPkgConfig() error {
	slog.Debug("write Linux ONNX Runtime pkg-config")
	// The shipped daemon finds libonnxruntime.so beside it via the
	// -Wl,-rpath,$ORIGIN cgo directive in internal/embedding/onnx.go. That
	// origin-relative path fails for `go test` and dev binaries, which run from
	// temporary build directories that do not carry the shared library. Add the
	// absolute staging lib directory as a second runpath so those binaries also
	// resolve the library. pkg-config expands ${prefix} to an absolute path, so
	// the emitted flag contains no shell metacharacter and Go's pkgconf parser
	// accepts it. The origin-relative runpath cannot go here because that parser
	// rejects the literal $ in $ORIGIN.
	contents := fmt.Sprintf(`prefix=%s
exec_prefix=${prefix}
includedir=${prefix}/include

Name: onnxruntime
Description: dynamically linked ONNX Runtime
Version: %s
Cflags: -I${includedir}
Libs: -L${prefix}/lib -Wl,-rpath,${prefix}/lib -lonnxruntime -ltokenizers -lstdc++ -ldl -lpthread -lm
`, installer.prefix, onnxRuntimeVersion)
	if err := os.WriteFile(
		installer.pkgConfigFile(),
		[]byte(contents),
		defaultFileMode,
	); err != nil {
		return wrapError("write Linux pkg-config file", err)
	}
	return nil
}
