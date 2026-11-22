package onnxruntimedist

import (
	"archive/zip"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
)

const packageHeaderRoot = "build/native/include"

var (
	errMissingPackageHeaders = errors.New("ONNX Runtime package has no headers")
	errUnsupportedEntry      = errors.New("unsupported ONNX Runtime package entry")
	errPackageEntrySize      = errors.New("ONNX Runtime package entry size mismatch")
)

func extractPackage(archivePath string, directory string, archive Archive) error {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return wrapError("open ONNX Runtime package", err)
	}

	defer reader.Close()

	libraryDirectory := filepath.Join(directory, "lib")
	includeDirectory := filepath.Join(directory, "include")

	for _, subdirectory := range []string{libraryDirectory, includeDirectory} {
		if err := os.Mkdir(subdirectory, defaultDirectoryMode); err != nil {
			return wrapError("create ONNX Runtime extraction subdirectory", err)
		}
	}

	libraryDestination := filepath.Join(libraryDirectory, archive.libraryName)
	if err := extractPackageFile(reader, archive.libraryPath, libraryDestination); err != nil {
		return err
	}

	headers, err := fs.Glob(reader, packageHeaderRoot+"/*.h")
	if err != nil {
		return wrapError("list ONNX Runtime package headers", err)
	}

	if len(headers) == 0 {
		return wrapError("find headers in "+packageHeaderRoot, errMissingPackageHeaders)
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
// It accepts regular files and creates them exclusively at that path.
func extractPackageFile(reader fs.FS, entry string, destinationPath string) error {
	source, err := reader.Open(entry)
	if err != nil {
		return wrapError("open package entry "+entry, err)
	}

	defer source.Close()

	info, err := source.Stat()
	if err != nil {
		return wrapError("stat package entry "+entry, err)
	}

	if !info.Mode().IsRegular() || info.Size() < 0 || info.Size() > maxExtractedFileSize {
		return wrapError("extract package entry "+entry, errUnsupportedEntry)
	}

	flags := os.O_CREATE | os.O_EXCL | os.O_WRONLY
	destination, err := os.OpenFile(destinationPath, flags, info.Mode().Perm())
	if err != nil {
		return wrapError("create extracted ONNX Runtime file", err)
	}

	written, copyErr := io.Copy(destination, io.LimitReader(source, info.Size()+1))
	closeErr := destination.Close()
	if copyErr != nil {
		return wrapError("copy package entry "+entry, copyErr)
	}

	if closeErr != nil {
		return wrapError("close extracted ONNX Runtime file", closeErr)
	}

	if written != info.Size() {
		operation := fmt.Sprintf("copy %q (%d of %d bytes)", entry, written, info.Size())

		return wrapError(operation, errPackageEntrySize)
	}

	return nil
}
