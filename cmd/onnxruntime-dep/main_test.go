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
