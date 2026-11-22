package onnxruntimedist

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"
)

func TestExtractPackageStagesTargetLibraryAndHeaders(t *testing.T) {
	archive, err := ArchiveFor("linux", "arm64")
	if err != nil {
		t.Fatal(err)
	}

	archivePath := filepath.Join(t.TempDir(), "runtime.nupkg")
	file, err := os.Create(archivePath)
	if err != nil {
		t.Fatal(err)
	}

	defer file.Close()

	writer := zip.NewWriter(file)

	for name, content := range map[string]string{
		"runtimes/linux-arm64/native/libonnxruntime.so": "arm64",
		"runtimes/linux-x64/native/libonnxruntime.so":   "amd64",
		"build/native/include/onnxruntime_c_api.h":      "header",
	} {
		entry, err := writer.Create(name)
		if err != nil {
			t.Fatal(err)
		}

		if _, err := entry.Write([]byte(content)); err != nil {
			t.Fatal(err)
		}
	}

	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	if err := file.Close(); err != nil {
		t.Fatal(err)
	}

	extracted := t.TempDir()
	if err := extractPackage(archivePath, extracted, archive); err != nil {
		t.Fatal(err)
	}

	for name, want := range map[string]string{
		filepath.Join("lib", linuxVersionedLibraryName): "arm64",
		filepath.Join("include", "onnxruntime_c_api.h"): "header",
	} {
		content, err := os.ReadFile(filepath.Join(extracted, name))
		if err != nil {
			t.Fatal(err)
		}

		if string(content) != want {
			t.Errorf("%s contains %q, want %q", name, content, want)
		}
	}
}
