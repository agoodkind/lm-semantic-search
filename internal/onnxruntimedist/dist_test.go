package onnxruntimedist

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
	"testing"
)

const testFileMode = 0o644

func TestLinuxArchivesUsePinnedOfficialReleases(t *testing.T) {
	testCases := []struct {
		architecture string
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
		t.Run(testCase.architecture, func(t *testing.T) {
			archive, err := ArchiveFor(operatingSystemLinux, testCase.architecture)
			if err != nil {
				t.Fatalf("ArchiveFor(linux, %q) error = %v", testCase.architecture, err)
			}
			if archive.Name != testCase.archiveName {
				t.Fatalf("archive name = %q, want %q", archive.Name, testCase.archiveName)
			}
			if archive.URL != testCase.url {
				t.Fatalf("archive URL = %q, want %q", archive.URL, testCase.url)
			}
			if archive.SHA256 != testCase.sha256 {
				t.Fatalf("archive SHA-256 = %q, want %q", archive.SHA256, testCase.sha256)
			}
		})
	}
}

func TestDarwinArchivesUsePinnedOfficialRelease(t *testing.T) {
	archive, err := ArchiveFor(operatingSystemDarwin, architectureARM64)
	if err != nil {
		t.Fatalf("ArchiveFor(darwin, arm64) error = %v", err)
	}
	if archive.Name != "onnxruntime-osx-arm64-1.27.0" {
		t.Fatalf("archive name = %q", archive.Name)
	}
	const expectedURL = "https://github.com/microsoft/onnxruntime/releases/download/" +
		"v1.27.0/onnxruntime-osx-arm64-1.27.0.tgz"
	if archive.URL != expectedURL {
		t.Fatalf("archive URL = %q, want %q", archive.URL, expectedURL)
	}
	const expectedSHA256 = "545e81c58152353acb0d1e8bd6ce4b62f830c0961f5b3acfedc790ffd76e477a"
	if archive.SHA256 != expectedSHA256 {
		t.Fatalf("archive SHA-256 = %q, want %q", archive.SHA256, expectedSHA256)
	}
	if _, err := ArchiveFor(operatingSystemDarwin, architectureAMD64); err == nil {
		t.Fatal("ArchiveFor(darwin, amd64) unexpectedly succeeded")
	}
}

func TestInstallSharedLibraryPlacesVersionedLibraryAndRelativeSymlinks(t *testing.T) {
	const archiveName = "onnxruntime-test"
	names := LibraryNames{
		Versioned:   "libonnxruntime.so." + Version,
		SONAME:      "libonnxruntime.so.1",
		Unversioned: "libonnxruntime.so",
	}
	archiveBytes := makeArchive(t, map[string]string{
		archiveName + "/lib/" + names.Versioned:      "shared-library",
		archiveName + "/include/onnxruntime_c_api.h": "header",
	})
	archiveDigest := sha256.Sum256(archiveBytes)
	server := httptest.NewServer(http.HandlerFunc(func(
		responseWriter http.ResponseWriter,
		_ *http.Request,
	) {
		if _, err := responseWriter.Write(archiveBytes); err != nil {
			t.Errorf("write archive response: %v", err)
		}
	}))
	defer server.Close()
	archive := Archive{
		Name:   archiveName,
		URL:    server.URL,
		SHA256: hex.EncodeToString(archiveDigest[:]),
	}

	binDirectory := t.TempDir()
	// The second install proves a rerun replaces the existing library and
	// symlinks instead of failing on them.
	for attempt := 1; attempt <= 2; attempt++ {
		if err := InstallSharedLibrary(
			context.Background(),
			server.Client(),
			archive,
			names,
			binDirectory,
		); err != nil {
			t.Fatalf("InstallSharedLibrary() attempt %d error = %v", attempt, err)
		}
	}

	libraryContents, err := os.ReadFile(filepath.Join(binDirectory, names.Versioned))
	if err != nil {
		t.Fatalf("read versioned library: %v", err)
	}
	if string(libraryContents) != "shared-library" {
		t.Fatalf("versioned library contents = %q", libraryContents)
	}
	for _, linkName := range []string{names.SONAME, names.Unversioned} {
		linkTarget, linkErr := os.Readlink(filepath.Join(binDirectory, linkName))
		if linkErr != nil {
			t.Fatalf("read %s symlink: %v", linkName, linkErr)
		}
		if linkTarget != names.Versioned {
			t.Fatalf("%s symlink target = %q, want %q", linkName, linkTarget, names.Versioned)
		}
	}
	entries, err := os.ReadDir(binDirectory)
	if err != nil {
		t.Fatalf("read bin directory: %v", err)
	}
	if len(entries) != 3 {
		t.Fatalf("bin directory has %d entries, want only the library and two symlinks", len(entries))
	}
}

func TestInstallSharedLibraryRejectsChecksumMismatch(t *testing.T) {
	names, err := LibraryNamesFor(operatingSystemLinux)
	if err != nil {
		t.Fatalf("LibraryNamesFor(linux) error = %v", err)
	}
	archiveBytes := makeArchive(t, map[string]string{
		"onnxruntime-test/lib/" + names.Versioned: "tampered",
	})
	server := httptest.NewServer(http.HandlerFunc(func(
		responseWriter http.ResponseWriter,
		_ *http.Request,
	) {
		if _, err := responseWriter.Write(archiveBytes); err != nil {
			t.Errorf("write archive response: %v", err)
		}
	}))
	defer server.Close()
	archive := Archive{
		Name:   "onnxruntime-test",
		URL:    server.URL,
		SHA256: linuxAMD64ArchiveSHA256,
	}

	binDirectory := t.TempDir()
	if err := InstallSharedLibrary(
		context.Background(),
		server.Client(),
		archive,
		names,
		binDirectory,
	); err == nil {
		t.Fatal("InstallSharedLibrary() accepted an archive that does not match its pin")
	}
	if _, err := os.Lstat(filepath.Join(binDirectory, names.Versioned)); !os.IsNotExist(err) {
		t.Fatalf("library was installed despite checksum mismatch: %v", err)
	}
}

func makeArchive(t *testing.T, entries map[string]string) []byte {
	t.Helper()

	var compressed bytes.Buffer
	gzipWriter := gzip.NewWriter(&compressed)
	tarWriter := tar.NewWriter(gzipWriter)
	for name, contents := range entries {
		header := &tar.Header{
			Name: name,
			Mode: testFileMode,
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
		Mode:     testFileMode,
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
	if err := os.WriteFile(archivePath, compressed.Bytes(), testFileMode); err != nil {
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
		Mode:     testFileMode,
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
	if err := os.WriteFile(archivePath, compressed.Bytes(), testFileMode); err != nil {
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
