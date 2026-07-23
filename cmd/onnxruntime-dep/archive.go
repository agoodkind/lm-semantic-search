package main

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
)

type pendingHardLink struct {
	path       string
	targetPath string
}

func extractZip(archivePath string, destinationDirectory string) error {
	slog.Debug("extract zip archive", "path", archivePath)
	archive, err := zip.OpenReader(archivePath)
	if err != nil {
		return wrapError("open zip archive", err)
	}
	defer archive.Close()

	if err := os.MkdirAll(destinationDirectory, defaultDirectoryMode); err != nil {
		return wrapError("create zip destination", err)
	}
	for _, entry := range archive.File {
		destinationPath, err := safeArchivePath(
			destinationDirectory,
			entry.Name,
		)
		if err != nil {
			return err
		}
		if entry.FileInfo().IsDir() {
			mode := entry.Mode().Perm()
			if mode == 0 {
				mode = defaultDirectoryMode
			}
			if err := os.MkdirAll(destinationPath, mode); err != nil {
				return wrapError("create zip directory", err)
			}
			continue
		}
		if entry.Mode()&os.ModeSymlink != 0 {
			if err := extractZipSymlink(entry, destinationPath); err != nil {
				return err
			}
			continue
		}
		if err := extractZipFile(entry, destinationPath); err != nil {
			return err
		}
	}
	return nil
}

func extractZipSymlink(entry *zip.File, destinationPath string) error {
	slog.Debug("extract zip symlink", "path", destinationPath)
	reader, err := entry.Open()
	if err != nil {
		return wrapError("open zip symlink", err)
	}
	targetReader := io.LimitReader(reader, maxSymlinkTargetSize+1)
	targetBytes, readErr := io.ReadAll(targetReader)
	closeErr := reader.Close()
	if readErr != nil {
		return wrapError("read zip symlink", readErr)
	}
	if closeErr != nil {
		return wrapError("close zip symlink", closeErr)
	}
	if len(targetBytes) > maxSymlinkTargetSize {
		return fmt.Errorf("zip symlink target exceeds %d bytes", maxSymlinkTargetSize)
	}
	if err := os.MkdirAll(
		filepath.Dir(destinationPath),
		defaultDirectoryMode,
	); err != nil {
		return wrapError("create zip symlink parent", err)
	}
	if err := os.Symlink(string(targetBytes), destinationPath); err != nil {
		return wrapError("create zip symlink", err)
	}
	return nil
}

func extractZipFile(entry *zip.File, destinationPath string) error {
	slog.Debug("extract zip file", "path", destinationPath)
	if entry.UncompressedSize64 > maxExtractedFileSize {
		return fmt.Errorf(
			"zip entry %s exceeds %d bytes",
			entry.Name,
			maxExtractedFileSize,
		)
	}
	expectedSize := int64(entry.UncompressedSize64)
	reader, err := entry.Open()
	if err != nil {
		return wrapError("open zip file", err)
	}
	if err := os.MkdirAll(
		filepath.Dir(destinationPath),
		defaultDirectoryMode,
	); err != nil {
		_ = reader.Close()
		return wrapError("create zip file parent", err)
	}
	mode := entry.Mode().Perm()
	if mode == 0 {
		mode = defaultFileMode
	}
	destination, err := os.OpenFile(
		destinationPath,
		os.O_CREATE|os.O_WRONLY|os.O_TRUNC,
		mode,
	)
	if err != nil {
		_ = reader.Close()
		return wrapError("create extracted zip file", err)
	}
	copyErr := copyExactSize(destination, reader, expectedSize)
	destinationCloseErr := destination.Close()
	readerCloseErr := reader.Close()
	if copyErr != nil {
		return copyErr
	}
	if destinationCloseErr != nil {
		return wrapError("close extracted zip file", destinationCloseErr)
	}
	if readerCloseErr != nil {
		return wrapError("close zip file", readerCloseErr)
	}
	return nil
}

func extractTarGzip(archivePath string, destinationDirectory string) error {
	slog.Debug("extract tar gzip archive", "path", archivePath)
	archiveFile, err := os.Open(archivePath)
	if err != nil {
		return wrapError("open tar gzip archive", err)
	}
	gzipReader, err := gzip.NewReader(archiveFile)
	if err != nil {
		_ = archiveFile.Close()
		return wrapError("open gzip stream", err)
	}

	pendingLinks := make([]pendingHardLink, 0)
	tarReader := tar.NewReader(gzipReader)
	for {
		header, nextErr := tarReader.Next()
		if errors.Is(nextErr, io.EOF) {
			break
		}
		if nextErr != nil {
			_ = gzipReader.Close()
			_ = archiveFile.Close()
			return wrapError("read tar entry", nextErr)
		}
		if err := extractTarEntry(
			tarReader,
			header,
			destinationDirectory,
			&pendingLinks,
		); err != nil {
			_ = gzipReader.Close()
			_ = archiveFile.Close()
			return err
		}
	}

	for _, link := range pendingLinks {
		if err := os.Link(link.targetPath, link.path); err != nil {
			_ = gzipReader.Close()
			_ = archiveFile.Close()
			return wrapError("create tar hard link", err)
		}
	}
	gzipCloseErr := gzipReader.Close()
	archiveCloseErr := archiveFile.Close()
	if gzipCloseErr != nil {
		return wrapError("close gzip stream", gzipCloseErr)
	}
	if archiveCloseErr != nil {
		return wrapError("close tar gzip archive", archiveCloseErr)
	}
	return nil
}

func extractTarEntry(
	tarReader *tar.Reader,
	header *tar.Header,
	destinationDirectory string,
	pendingLinks *[]pendingHardLink,
) error {
	slog.Debug("extract tar entry", "name", header.Name)
	destinationPath, err := safeArchivePath(
		destinationDirectory,
		header.Name,
	)
	if err != nil {
		return err
	}
	mode := header.FileInfo().Mode().Perm()

	switch header.Typeflag {
	case tar.TypeDir:
		if err := os.MkdirAll(destinationPath, mode); err != nil {
			return wrapError("create tar directory", err)
		}
		return nil
	case tar.TypeReg, 0:
		return extractTarFile(tarReader, destinationPath, mode, header.Size)
	case tar.TypeSymlink:
		if err := os.MkdirAll(
			filepath.Dir(destinationPath),
			defaultDirectoryMode,
		); err != nil {
			return wrapError("create tar symlink parent", err)
		}
		if err := os.Symlink(header.Linkname, destinationPath); err != nil {
			return wrapError("create tar symlink", err)
		}
		return nil
	case tar.TypeLink:
		targetPath, pathErr := safeArchivePath(
			destinationDirectory,
			header.Linkname,
		)
		if pathErr != nil {
			return pathErr
		}
		if err := os.MkdirAll(
			filepath.Dir(destinationPath),
			defaultDirectoryMode,
		); err != nil {
			return wrapError("create tar hard link parent", err)
		}
		*pendingLinks = append(*pendingLinks, pendingHardLink{
			path:       destinationPath,
			targetPath: targetPath,
		})
		return nil
	case tar.TypeXGlobalHeader:
		return nil
	default:
		return fmt.Errorf(
			"unsupported tar entry type %d for %s",
			header.Typeflag,
			header.Name,
		)
	}
}

func extractTarFile(
	tarReader *tar.Reader,
	destinationPath string,
	mode os.FileMode,
	expectedSize int64,
) error {
	slog.Debug("extract tar file", "path", destinationPath)
	if expectedSize < 0 || expectedSize > maxExtractedFileSize {
		return fmt.Errorf(
			"tar entry %s has unsupported size %d",
			destinationPath,
			expectedSize,
		)
	}
	if err := os.MkdirAll(
		filepath.Dir(destinationPath),
		defaultDirectoryMode,
	); err != nil {
		return wrapError("create tar file parent", err)
	}
	destination, err := os.OpenFile(
		destinationPath,
		os.O_CREATE|os.O_WRONLY|os.O_TRUNC,
		mode,
	)
	if err != nil {
		return wrapError("create extracted tar file", err)
	}
	copyErr := copyExactSize(destination, tarReader, expectedSize)
	closeErr := destination.Close()
	if copyErr != nil {
		return copyErr
	}
	if closeErr != nil {
		return wrapError("close extracted tar file", closeErr)
	}
	return nil
}

func copyExactSize(
	destination io.Writer,
	source io.Reader,
	expectedSize int64,
) error {
	slog.Debug("copy fixed-size archive entry", "bytes", expectedSize)
	writtenSize, err := io.CopyN(destination, source, expectedSize)
	if err != nil {
		return wrapError("copy archive entry", err)
	}
	if writtenSize != expectedSize {
		return fmt.Errorf(
			"archive entry copied %d bytes, want %d",
			writtenSize,
			expectedSize,
		)
	}
	return nil
}
