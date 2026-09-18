package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/onnxruntimedist"
)

func TestInstallDarwinSharedArchiveStagesDynamicLibraryAndHeaders(t *testing.T) {
	const archiveName = "onnxruntime-osx-arm64-1.27.0"
	archiveBytes := makeDarwinArchive(t, archiveName)
	archiveDigest := sha256.Sum256(archiveBytes)

	server := httptest.NewServer(http.HandlerFunc(func(
		responseWriter http.ResponseWriter,
		_ *http.Request,
	) {
		responseWriter.WriteHeader(http.StatusOK)
		if _, err := responseWriter.Write(archiveBytes); err != nil {
			t.Errorf("write archive response: %v", err)
		}
	}))
	defer server.Close()

	prefix := t.TempDir()
	installer := dependencyInstaller{
		prefix: prefix,
		target: buildTarget{
			goos:   operatingSystemDarwin,
			goarch: architectureARM64,
		},
		httpClient: server.Client(),
	}
	if err := installer.preparePrefix(); err != nil {
		t.Fatalf("preparePrefix() error = %v", err)
	}
	archive := onnxruntimedist.Archive{
		Name:   archiveName,
		URL:    server.URL,
		SHA256: hex.EncodeToString(archiveDigest[:]),
	}
	if err := installer.installSharedArchive(
		context.Background(),
		t.TempDir(),
		archive,
	); err != nil {
		t.Fatalf("installSharedArchive() error = %v", err)
	}

	versionedLibrary := filepath.Join(
		prefix,
		"lib",
		"libonnxruntime."+onnxruntimedist.Version+".dylib",
	)
	libraryContents, err := os.ReadFile(versionedLibrary)
	if err != nil {
		t.Fatalf("read versioned library: %v", err)
	}
	if string(libraryContents) != "shared-library" {
		t.Fatalf("versioned library contents = %q", libraryContents)
	}

	for _, linkName := range []string{
		"libonnxruntime.1.dylib",
		"libonnxruntime.dylib",
	} {
		linkTarget, linkErr := os.Readlink(filepath.Join(prefix, "lib", linkName))
		if linkErr != nil {
			t.Fatalf("read %s symlink: %v", linkName, linkErr)
		}
		if linkTarget != filepath.Base(versionedLibrary) {
			t.Fatalf(
				"%s symlink target = %q, want %q",
				linkName,
				linkTarget,
				filepath.Base(versionedLibrary),
			)
		}
	}

	headerContents, err := os.ReadFile(filepath.Join(prefix, "include", "onnxruntime_c_api.h"))
	if err != nil {
		t.Fatalf("read staged header: %v", err)
	}
	if string(headerContents) != "header" {
		t.Fatalf("header contents = %q", headerContents)
	}
	debugSymbolsPath := filepath.Join(
		prefix,
		"lib",
		"libonnxruntime."+onnxruntimedist.Version+".dylib.dSYM",
	)
	if _, err := os.Stat(debugSymbolsPath); !os.IsNotExist(err) {
		t.Fatalf("debug symbols were staged at %s", debugSymbolsPath)
	}
}

func TestInstallLinuxSharedArchiveStagesDynamicLibraryAndHeaders(t *testing.T) {
	const archiveName = "onnxruntime-linux-x64-1.27.0"
	archiveBytes := makeLinuxArchive(t, archiveName)
	archiveDigest := sha256.Sum256(archiveBytes)

	server := httptest.NewServer(http.HandlerFunc(func(
		responseWriter http.ResponseWriter,
		_ *http.Request,
	) {
		responseWriter.WriteHeader(http.StatusOK)
		if _, err := responseWriter.Write(archiveBytes); err != nil {
			t.Errorf("write archive response: %v", err)
		}
	}))
	defer server.Close()

	prefix := t.TempDir()
	installer := dependencyInstaller{
		prefix: prefix,
		target: buildTarget{
			goos:   operatingSystemLinux,
			goarch: architectureAMD64,
		},
		httpClient: server.Client(),
	}
	if err := installer.preparePrefix(); err != nil {
		t.Fatalf("preparePrefix() error = %v", err)
	}
	archive := onnxruntimedist.Archive{
		Name:   archiveName,
		URL:    server.URL,
		SHA256: hex.EncodeToString(archiveDigest[:]),
	}
	if err := installer.installSharedArchive(
		context.Background(),
		t.TempDir(),
		archive,
	); err != nil {
		t.Fatalf("installSharedArchive() error = %v", err)
	}

	versionedLibrary := filepath.Join(
		prefix,
		"lib",
		"libonnxruntime.so."+onnxruntimedist.Version,
	)
	libraryContents, err := os.ReadFile(versionedLibrary)
	if err != nil {
		t.Fatalf("read versioned library: %v", err)
	}
	if string(libraryContents) != "shared-library" {
		t.Fatalf("versioned library contents = %q", libraryContents)
	}

	libraryLink := filepath.Join(prefix, "lib", "libonnxruntime.so")
	linkTarget, err := os.Readlink(libraryLink)
	if err != nil {
		t.Fatalf("read library symlink: %v", err)
	}
	if linkTarget != filepath.Base(versionedLibrary) {
		t.Fatalf("library symlink target = %q, want %q", linkTarget, filepath.Base(versionedLibrary))
	}
	sonameLink := filepath.Join(prefix, "lib", "libonnxruntime.so.1")
	sonameTarget, err := os.Readlink(sonameLink)
	if err != nil {
		t.Fatalf("read SONAME symlink: %v", err)
	}
	if sonameTarget != filepath.Base(versionedLibrary) {
		t.Fatalf("SONAME symlink target = %q, want %q", sonameTarget, filepath.Base(versionedLibrary))
	}

	headerContents, err := os.ReadFile(filepath.Join(prefix, "include", "onnxruntime_c_api.h"))
	if err != nil {
		t.Fatalf("read staged header: %v", err)
	}
	if string(headerContents) != "header" {
		t.Fatalf("header contents = %q", headerContents)
	}
}

func TestWriteLinuxPkgConfigLinksSharedLibraries(t *testing.T) {
	prefix := t.TempDir()
	if err := os.MkdirAll(
		filepath.Join(prefix, "lib", "pkgconfig"),
		defaultDirectoryMode,
	); err != nil {
		t.Fatalf("create pkg-config directory: %v", err)
	}
	installer := dependencyInstaller{prefix: prefix}
	if err := installer.writeLinuxPkgConfig(); err != nil {
		t.Fatalf("writeLinuxPkgConfig() error = %v", err)
	}

	contents, err := os.ReadFile(installer.pkgConfigFile())
	if err != nil {
		t.Fatalf("read pkg-config file: %v", err)
	}
	const expected = "Libs: -L${prefix}/lib -Wl,-rpath,${prefix}/lib " +
		"-lonnxruntime -ltokenizers -lstdc++ -ldl -lpthread -lm"
	for _, line := range strings.Split(string(contents), "\n") {
		if strings.HasPrefix(line, "Libs:") {
			if line != expected {
				t.Fatalf("Libs line = %q, want %q", line, expected)
			}
			return
		}
	}
	t.Fatal("pkg-config file has no Libs line")
}

func TestWriteDarwinPkgConfigLinksSharedLibraries(t *testing.T) {
	prefix := t.TempDir()
	if err := os.MkdirAll(
		filepath.Join(prefix, "lib", "pkgconfig"),
		defaultDirectoryMode,
	); err != nil {
		t.Fatalf("create pkg-config directory: %v", err)
	}
	installer := dependencyInstaller{prefix: prefix}
	if err := installer.writeDarwinPkgConfig(); err != nil {
		t.Fatalf("writeDarwinPkgConfig() error = %v", err)
	}

	contents, err := os.ReadFile(installer.pkgConfigFile())
	if err != nil {
		t.Fatalf("read pkg-config file: %v", err)
	}
	const expected = "Libs: -L${prefix}/lib -Wl,-rpath,${prefix}/lib " +
		"-lonnxruntime -ltokenizers"
	for _, line := range strings.Split(string(contents), "\n") {
		if strings.HasPrefix(line, "Libs:") {
			if line != expected {
				t.Fatalf("Libs line = %q, want %q", line, expected)
			}
			return
		}
	}
	t.Fatal("pkg-config file has no Libs line")
}

func TestIsCachedRejectsLegacySentinel(t *testing.T) {
	installer := dependencyInstaller{
		prefix: t.TempDir(),
		target: buildTarget{
			goos:   operatingSystemLinux,
			goarch: architectureAMD64,
		},
	}
	stageDependencyCache(
		t,
		installer,
		onnxruntimedist.Version,
		"Libs: -L${prefix}/lib -Wl,-rpath,${prefix}/lib -lonnxruntime\n",
	)

	cached, err := installer.isCached()
	if err != nil {
		t.Fatalf("isCached() error = %v", err)
	}
	if cached {
		t.Fatal("isCached() = true for a legacy cache sentinel")
	}
}

func TestIsCachedRejectsStaleLinuxPkgConfig(t *testing.T) {
	installer := dependencyInstaller{
		prefix: t.TempDir(),
		target: buildTarget{
			goos:   operatingSystemLinux,
			goarch: architectureAMD64,
		},
	}
	stageDependencyCache(
		t,
		installer,
		dependencyCacheSentinel(),
		"Libs: -L${prefix}/lib -lonnxruntime -Wl,-rpath,$ORIGIN\n",
	)

	cached, err := installer.isCached()
	if err != nil {
		t.Fatalf("isCached() error = %v", err)
	}
	if cached {
		t.Fatal("isCached() = true for stale Linux pkg-config contents")
	}
}

func TestIsCachedRejectsStaleDarwinPkgConfig(t *testing.T) {
	testCases := []struct {
		name              string
		pkgConfigContents string
	}{
		{
			name: "CoreML framework",
			pkgConfigContents: "Description: dynamically linked ONNX Runtime\n" +
				"Libs: -L${prefix}/lib -Wl,-rpath,${prefix}/lib " +
				"-lonnxruntime -ltokenizers -framework CoreML\n",
		},
		{
			name: "static C++ link",
			pkgConfigContents: "Description: statically linked ONNX Runtime\n" +
				"Libs: -L${prefix}/lib -Wl,-rpath,${prefix}/lib " +
				"-lonnxruntime -ltokenizers -lc++\n",
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			installer := dependencyInstaller{
				prefix: t.TempDir(),
				target: buildTarget{
					goos:   operatingSystemDarwin,
					goarch: architectureARM64,
				},
			}
			stageDependencyCache(
				t,
				installer,
				dependencyCacheSentinel(),
				testCase.pkgConfigContents,
			)

			cached, err := installer.isCached()
			if err != nil {
				t.Fatalf("isCached() error = %v", err)
			}
			if cached {
				t.Fatal("isCached() = true for stale Darwin pkg-config contents")
			}
		})
	}
}

func stageDependencyCache(
	t *testing.T,
	installer dependencyInstaller,
	versionContents string,
	pkgConfigContents string,
) {
	t.Helper()

	files := map[string]string{
		installer.versionFile():   versionContents + "\n",
		installer.pkgConfigFile(): pkgConfigContents,
	}
	for _, libraryPath := range installer.cachedLibraryPaths() {
		files[libraryPath] = "library"
	}
	for path, contents := range files {
		if err := os.MkdirAll(filepath.Dir(path), defaultDirectoryMode); err != nil {
			t.Fatalf("create cache directory: %v", err)
		}
		if err := os.WriteFile(path, []byte(contents), defaultFileMode); err != nil {
			t.Fatalf("write cache file: %v", err)
		}
	}
}

func makeLinuxArchive(t *testing.T, archiveName string) []byte {
	t.Helper()

	var compressed bytes.Buffer
	gzipWriter := gzip.NewWriter(&compressed)
	tarWriter := tar.NewWriter(gzipWriter)
	entries := map[string]string{
		archiveName + "/lib/libonnxruntime.so." + onnxruntimedist.Version: "shared-library",
		archiveName + "/include/onnxruntime_c_api.h":                      "header",
	}
	for name, contents := range entries {
		header := &tar.Header{
			Name: name,
			Mode: defaultFileMode,
			Size: int64(len(contents)),
		}
		if err := tarWriter.WriteHeader(header); err != nil {
			t.Fatalf("write tar header: %v", err)
		}
		if _, err := io.WriteString(tarWriter, contents); err != nil {
			t.Fatalf("write tar contents: %v", err)
		}
	}
	if err := tarWriter.Close(); err != nil {
		t.Fatalf("close tar writer: %v", err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatalf("close gzip writer: %v", err)
	}
	return compressed.Bytes()
}

func makeDarwinArchive(t *testing.T, archiveName string) []byte {
	t.Helper()

	var compressed bytes.Buffer
	gzipWriter := gzip.NewWriter(&compressed)
	tarWriter := tar.NewWriter(gzipWriter)
	entries := map[string]string{
		archiveName + "/lib/libonnxruntime." + onnxruntimedist.Version + ".dylib": "shared-library",
		archiveName + "/lib/libonnxruntime." + onnxruntimedist.Version +
			".dylib.dSYM/Contents/Info.plist": "debug-symbols",
		archiveName + "/include/onnxruntime_c_api.h": "header",
	}
	for name, contents := range entries {
		header := &tar.Header{
			Name: name,
			Mode: defaultFileMode,
			Size: int64(len(contents)),
		}
		if err := tarWriter.WriteHeader(header); err != nil {
			t.Fatalf("write tar header: %v", err)
		}
		if _, err := io.WriteString(tarWriter, contents); err != nil {
			t.Fatalf("write tar contents: %v", err)
		}
	}
	if err := tarWriter.Close(); err != nil {
		t.Fatalf("close tar writer: %v", err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatalf("close gzip writer: %v", err)
	}
	return compressed.Bytes()
}
