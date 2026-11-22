package onnxruntimedist

import (
	"archive/zip"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
)

const packageHeaderRoot = "build/native/include"

func extractPackage(archivePath string, directory string, archive Archive) error {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer reader.Close()

	libraryDirectory := filepath.Join(directory, "lib")
	includeDirectory := filepath.Join(directory, "include")
	for _, subdirectory := range []string{libraryDirectory, includeDirectory} {
		if err := os.Mkdir(subdirectory, defaultDirectoryMode); err != nil {
			return err
		}
	}

	libraryDestination := filepath.Join(libraryDirectory, archive.libraryName)
	if err := extractPackageFile(reader, archive.libraryPath, libraryDestination); err != nil {
		return err
	}

	headers, err := fs.Glob(reader, packageHeaderRoot+"/*.h")
	if err != nil {
		return err
	}
	if len(headers) == 0 {
		return fmt.Errorf("ONNX Runtime package has no headers in %s", packageHeaderRoot)
	}

	for _, header := range headers {
		destination := filepath.Join(includeDirectory, path.Base(header))
		if err := extractPackageFile(reader, header, destination); err != nil {
			return err
		}
	}

	return nil
}

// extractPackageFile copies a selected file to a caller-supplied path.
// The archive never controls the destination directory or creates links.
func extractPackageFile(reader fs.FS, entry string, destinationPath string) error {
	source, err := reader.Open(entry)
	if err != nil {
		return err
	}
	defer source.Close()

	info, err := source.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() < 0 || info.Size() > maxExtractedFileSize {
		return fmt.Errorf("unsupported ONNX Runtime package entry %q", entry)
	}

	destination, err := os.OpenFile(
		destinationPath,
		os.O_CREATE|os.O_EXCL|os.O_WRONLY,
		info.Mode().Perm(),
	)
	if err != nil {
		return err
	}

	written, copyErr := io.Copy(destination, io.LimitReader(source, info.Size()+1))
	closeErr := destination.Close()
	if copyErr != nil {
		return copyErr
	}
	if closeErr != nil {
		return closeErr
	}
	if written != info.Size() {
		return fmt.Errorf("ONNX Runtime package entry %q has size %d, want %d", entry, written, info.Size())
	}

	return nil
}
