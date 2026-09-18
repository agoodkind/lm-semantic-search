// Command onnxruntime-dep stages ONNX Runtime for cgo.
package main

import (
	"context"
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

	"goodkind.io/lm-semantic-search/internal/onnxruntimedist"
)

const (
	cacheFormatRevision             = 3
	rejectedDarwinCoreML            = "-framework CoreML"
	rejectedDarwinStaticCXX         = "-lc++"
	rejectedDarwinStaticDescription = "Description: statically linked ONNX Runtime"
	expectedDarwinRPath             = "-Wl,-rpath,${prefix}/lib"
	rejectedLinuxRPath              = ",-rpath,$ORIGIN"
	expectedLinuxRPath              = "-Wl,-rpath,${prefix}/lib"
	toolLogPrefix                   = "setup-cgo-onnxruntime"
	defaultFileMode                 = 0o644
	defaultDirectoryMode            = 0o755
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
			onnxruntimedist.Version,
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
		[]byte(dependencyCacheSentinel()+"\n"),
		defaultFileMode,
	); err != nil {
		return wrapError("write version file", err)
	}

	fmt.Printf(
		"%s: installed ONNX Runtime %s for %s/%s\n",
		toolLogPrefix,
		onnxruntimedist.Version,
		target.goos,
		target.goarch,
	)
	return nil
}

func dependencyCacheSentinel() string {
	return fmt.Sprintf("%s+%d", onnxruntimedist.Version, cacheFormatRevision)
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
	if version != dependencyCacheSentinel() {
		return false, nil
	}

	pkgConfigContents, err := os.ReadFile(installer.pkgConfigFile())
	if err != nil {
		return false, wrapError("read pkg-config file", err)
	}
	pkgConfig := string(pkgConfigContents)
	switch installer.target.goos {
	case operatingSystemDarwin:
		if strings.Contains(pkgConfig, rejectedDarwinCoreML) ||
			strings.Contains(pkgConfig, rejectedDarwinStaticCXX) ||
			strings.Contains(pkgConfig, rejectedDarwinStaticDescription) {
			return false, nil
		}
		return strings.Contains(pkgConfig, expectedDarwinRPath), nil
	case operatingSystemLinux:
		if strings.Contains(pkgConfig, rejectedLinuxRPath) {
			return false, nil
		}
		return strings.Contains(pkgConfig, expectedLinuxRPath), nil
	default:
		return false, nil
	}
}

func (installer dependencyInstaller) cachedLibraryPaths() []string {
	names, err := onnxruntimedist.LibraryNamesFor(string(installer.target.goos))
	if err != nil {
		return nil
	}
	return installer.libraryPaths(names)
}

func (installer dependencyInstaller) libraryPaths(names onnxruntimedist.LibraryNames) []string {
	return []string{
		filepath.Join(installer.prefix, "lib", names.Versioned),
		filepath.Join(installer.prefix, "lib", names.SONAME),
		filepath.Join(installer.prefix, "lib", names.Unversioned),
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
		installer.pkgConfigFile(),
	}
	for _, goos := range []operatingSystem{operatingSystemDarwin, operatingSystemLinux} {
		names, err := onnxruntimedist.LibraryNamesFor(string(goos))
		if err != nil {
			return wrapError("resolve library names", err)
		}
		filesToRemove = append(filesToRemove, installer.libraryPaths(names)...)
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
	archive, err := onnxruntimedist.ArchiveFor(
		string(installer.target.goos),
		string(installer.target.goarch),
	)
	if err != nil {
		return err
	}
	return installer.installSharedArchive(ctx, temporaryDirectory, archive)
}

func (installer dependencyInstaller) installSharedArchive(
	ctx context.Context,
	temporaryDirectory string,
	archive onnxruntimedist.Archive,
) error {
	slog.DebugContext(ctx, "install ONNX Runtime shared archive", "goos", installer.target.goos)
	names, err := onnxruntimedist.LibraryNamesFor(string(installer.target.goos))
	if err != nil {
		return err
	}
	archiveDirectory, err := onnxruntimedist.FetchArchive(
		ctx,
		installer.httpClient,
		archive,
		temporaryDirectory,
	)
	if err != nil {
		return err
	}
	if err := onnxruntimedist.InstallLibrary(
		archiveDirectory,
		names,
		filepath.Join(installer.prefix, "lib"),
	); err != nil {
		return err
	}

	if err := copyHeaderFiles(
		filepath.Join(archiveDirectory, "include"),
		filepath.Join(installer.prefix, "include"),
	); err != nil {
		return wrapError("copy headers", err)
	}
	if installer.target.goos == operatingSystemDarwin {
		return installer.writeDarwinPkgConfig()
	}
	return installer.writeLinuxPkgConfig()
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

func (installer dependencyInstaller) writeDarwinPkgConfig() error {
	slog.Debug("write Darwin ONNX Runtime pkg-config")
	contents := fmt.Sprintf(`prefix=%s
exec_prefix=${prefix}
includedir=${prefix}/include

Name: onnxruntime
Description: dynamically linked ONNX Runtime
Version: %s
Cflags: -I${includedir}
Libs: -L${prefix}/lib -Wl,-rpath,${prefix}/lib -lonnxruntime -ltokenizers
`, installer.prefix, onnxruntimedist.Version)
	if err := os.WriteFile(
		installer.pkgConfigFile(),
		[]byte(contents),
		defaultFileMode,
	); err != nil {
		return wrapError("write Darwin pkg-config file", err)
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
`, installer.prefix, onnxruntimedist.Version)
	if err := os.WriteFile(
		installer.pkgConfigFile(),
		[]byte(contents),
		defaultFileMode,
	); err != nil {
		return wrapError("write Linux pkg-config file", err)
	}
	return nil
}
