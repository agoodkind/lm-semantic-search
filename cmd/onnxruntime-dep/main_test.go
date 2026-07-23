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
)

func TestLinuxArchivesUsePinnedOfficialReleases(t *testing.T) {
	testCases := []struct {
		architecture architecture
		archiveName  string
		url          string
		sha256       string
	}{
		{
			architecture: architectureAMD64,
			archiveName:  "onnxruntime-linux-x64-1.27.0",
			url: "https://github.com/microsoft/onnxruntime/releases/download/" +
				"v1.27.0/onnxruntime-linux-x64-1.27.0.tgz",
			sha256: "547e40a48f1fe73e3f812d7c88a948612c23f896b91e4e2ee1e232d7b468246f",
		},
		{
			architecture: architectureARM64,
			archiveName:  "onnxruntime-linux-aarch64-1.27.0",
			url: "https://github.com/microsoft/onnxruntime/releases/download/" +
				"v1.27.0/onnxruntime-linux-aarch64-1.27.0.tgz",
			sha256: "3e4d83ac06924a32a07b6d7f91ce6f852876153fc0bbdf931bf517a140bfbe48",
		},
	}

	for _, testCase := range testCases {
		t.Run(string(testCase.architecture), func(t *testing.T) {
			archive, ok := linuxArchives[testCase.architecture]
			if !ok {
				t.Fatalf("linuxArchives[%q] is missing", testCase.architecture)
			}
			if archive.archiveName != testCase.archiveName {
				t.Fatalf("archive name = %q, want %q", archive.archiveName, testCase.archiveName)
			}
			if archive.url != testCase.url {
				t.Fatalf("archive URL = %q, want %q", archive.url, testCase.url)
			}
			if archive.sha256 != testCase.sha256 {
				t.Fatalf("archive SHA-256 = %q, want %q", archive.sha256, testCase.sha256)
			}
		})
	}
}

func TestDarwinArchivesUsePinnedOfficialRelease(t *testing.T) {
	archive, ok := darwinArchives[architectureARM64]
	if !ok {
		t.Fatalf("darwinArchives[%q] is missing", architectureARM64)
	}
	if archive.archiveName != "onnxruntime-osx-arm64-1.27.0" {
		t.Fatalf("archive name = %q", archive.archiveName)
	}
	const expectedURL = "https://github.com/microsoft/onnxruntime/releases/download/" +
		"v1.27.0/onnxruntime-osx-arm64-1.27.0.tgz"
	if archive.url != expectedURL {
		t.Fatalf("archive URL = %q, want %q", archive.url, expectedURL)
	}
	const expectedSHA256 = "545e81c58152353acb0d1e8bd6ce4b62f830c0961f5b3acfedc790ffd76e477a"
	if archive.sha256 != expectedSHA256 {
		t.Fatalf("archive SHA-256 = %q, want %q", archive.sha256, expectedSHA256)
	}
	if _, found := darwinArchives[architectureAMD64]; found {
		t.Fatal("darwinArchives unexpectedly supports amd64")
	}
}

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
	archive := darwinArchive{
		archiveName: archiveName,
		url:         server.URL,
		sha256:      hex.EncodeToString(archiveDigest[:]),
	}
	if err := installer.installDarwinSharedArchive(
		context.Background(),
		t.TempDir(),
		archive,
	); err != nil {
		t.Fatalf("installDarwinSharedArchive() error = %v", err)
	}

	versionedLibrary := filepath.Join(
		prefix,
		"lib",
		"libonnxruntime."+onnxRuntimeVersion+".dylib",
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
		"libonnxruntime."+onnxRuntimeVersion+".dylib.dSYM",
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
	archive := linuxArchive{
		archiveName: archiveName,
		url:         server.URL,
		sha256:      hex.EncodeToString(archiveDigest[:]),
	}
	if err := installer.installLinuxSharedArchive(
		context.Background(),
		t.TempDir(),
		archive,
	); err != nil {
		t.Fatalf("installLinuxSharedArchive() error = %v", err)
	}

	versionedLibrary := filepath.Join(
		prefix,
		"lib",
		"libonnxruntime.so."+onnxRuntimeVersion,
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
		onnxRuntimeVersion,
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
		archiveName + "/lib/libonnxruntime.so." + onnxRuntimeVersion: "shared-library",
		archiveName + "/include/onnxruntime_c_api.h":                 "header",
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
		archiveName + "/lib/libonnxruntime." + onnxRuntimeVersion + ".dylib": "shared-library",
		archiveName + "/lib/libonnxruntime." + onnxRuntimeVersion +
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

func TestSafeArchivePathRejectsTraversal(t *testing.T) {
	root := t.TempDir()
	rejected := []string{
		"../escape",
		"../../escape",
		"lib/../../escape",
		"/etc/passwd",
	}
	for _, entry := range rejected {
		if _, err := safeArchivePath(root, entry); err == nil {
			t.Fatalf("safeArchivePath(%q) = nil error, want rejection", entry)
		}
	}

	accepted := filepath.Join("lib", "libonnxruntime.1.27.0.dylib")
	resolved, err := safeArchivePath(root, accepted)
	if err != nil {
		t.Fatalf("safeArchivePath(%q) returned error: %v", accepted, err)
	}
	want := filepath.Join(root, accepted)
	if resolved != want {
		t.Fatalf("safeArchivePath(%q) = %q, want %q", accepted, resolved, want)
	}
}

func TestExtractTarGzipRejectsSymlinkedParentEscape(t *testing.T) {
	// Plant a symlink inside the destination that points outside it, then
	// extract an archive that writes a file through that symlinked directory.
	// The entry name "linkdir/payload" is a perfectly local path, so a lexical
	// check would allow it; the real-path parent check must resolve linkdir to
	// its true location outside the root and refuse the write.
	outside := t.TempDir()
	destination := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(destination, "linkdir")); err != nil {
		t.Fatalf("plant symlinked parent: %v", err)
	}

	var compressed bytes.Buffer
	gzipWriter := gzip.NewWriter(&compressed)
	tarWriter := tar.NewWriter(gzipWriter)
	body := []byte("payload")
	header := &tar.Header{
		Name:     "linkdir/payload",
		Typeflag: tar.TypeReg,
		Mode:     defaultFileMode,
		Size:     int64(len(body)),
	}
	if err := tarWriter.WriteHeader(header); err != nil {
		t.Fatalf("WriteHeader returned error: %v", err)
	}
	if _, err := tarWriter.Write(body); err != nil {
		t.Fatalf("Write returned error: %v", err)
	}
	if err := tarWriter.Close(); err != nil {
		t.Fatalf("tar Close returned error: %v", err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatalf("gzip Close returned error: %v", err)
	}

	archivePath := filepath.Join(t.TempDir(), "escape.tgz")
	if err := os.WriteFile(archivePath, compressed.Bytes(), defaultFileMode); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	if err := extractTarGzip(archivePath, destination); err == nil {
		t.Fatal("extractTarGzip wrote through a symlinked parent, want rejection")
	}
	if _, err := os.Lstat(filepath.Join(outside, "payload")); !os.IsNotExist(err) {
		t.Fatalf("payload escaped into the outside directory: %v", err)
	}
}

func TestExtractTarGzipRejectsEscapingSymlink(t *testing.T) {
	var compressed bytes.Buffer
	gzipWriter := gzip.NewWriter(&compressed)
	tarWriter := tar.NewWriter(gzipWriter)
	header := &tar.Header{
		Name:     "onnxruntime/lib/evil",
		Linkname: "../../../../../../tmp/escape",
		Typeflag: tar.TypeSymlink,
		Mode:     defaultFileMode,
	}
	if err := tarWriter.WriteHeader(header); err != nil {
		t.Fatalf("WriteHeader returned error: %v", err)
	}
	if err := tarWriter.Close(); err != nil {
		t.Fatalf("tar Close returned error: %v", err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatalf("gzip Close returned error: %v", err)
	}

	archiveDirectory := t.TempDir()
	archivePath := filepath.Join(archiveDirectory, "malicious.tgz")
	if err := os.WriteFile(archivePath, compressed.Bytes(), defaultFileMode); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	destination := t.TempDir()
	if err := extractTarGzip(archivePath, destination); err == nil {
		t.Fatal("extractTarGzip accepted an escaping symlink, want rejection")
	}
	if _, err := os.Lstat(filepath.Join(destination, "lib", "evil")); !os.IsNotExist(err) {
		t.Fatalf("escaping symlink was created despite rejection: %v", err)
	}
}
