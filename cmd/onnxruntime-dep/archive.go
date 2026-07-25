package main

import (
	"archive/tar"
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

func extractTarGzip(archivePath string, destinationDirectory string) error {
	slog.Debug("extract tar gzip archive", "path", archivePath)
	// Resolve the extraction root to its real path once. Every entry's real
	// parent directory is later checked against this, so a symlink planted by an
	// earlier entry cannot redirect a later write outside the root.
	if err := os.MkdirAll(destinationDirectory, defaultDirectoryMode); err != nil {
		return wrapError("create extraction root", err)
	}
	realRoot, err := filepath.EvalSymlinks(destinationDirectory)
	if err != nil {
		return wrapError("resolve extraction root", err)
	}

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
			realRoot,
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
	realRoot string,
	pendingLinks *[]pendingHardLink,
) error {
	slog.Debug("extract tar entry", "name", header.Name)
	destinationPath, err := safeArchivePath(destinationDirectory, header.Name)
	if err != nil {
		return err
	}
	mode := header.FileInfo().Mode().Perm()

	switch header.Typeflag {
	case tar.TypeDir:
		if err := os.MkdirAll(destinationPath, mode); err != nil {
			return wrapError("create tar directory", err)
		}
		return verifyRealPathWithinRoot(realRoot, destinationPath)
	case tar.TypeReg, 0:
		return extractTarFile(
			tarReader,
			realRoot,
			destinationPath,
			mode,
			header.Size,
		)
	case tar.TypeSymlink:
		// The target must be a local relative path, so following the link
		// cannot leave the directory the link lives in. Combined with the real
		// parent check below and the same rule on every other entry name, no
		// symlink in any parent chain can point outside the extraction root.
		linkTarget := filepath.FromSlash(header.Linkname)
		if !filepath.IsLocal(linkTarget) {
			return fmt.Errorf("unsafe symlink target %q", header.Linkname)
		}
		if err := createParentWithinRoot(realRoot, destinationPath); err != nil {
			return err
		}
		// Resolve the real directory the target points into and confirm it stays
		// within the root. Passing a path derived from the target through
		// [filepath.EvalSymlinks] follows any symlink an earlier entry planted,
		// so the finished link cannot resolve out of the extraction root.
		targetDirectory := filepath.Dir(
			filepath.Join(filepath.Dir(destinationPath), linkTarget),
		)
		realTargetDirectory, resolveErr := filepath.EvalSymlinks(targetDirectory)
		if resolveErr != nil {
			return wrapError("resolve tar symlink target", resolveErr)
		}
		if !pathWithinRoot(realRoot, realTargetDirectory) {
			return fmt.Errorf(
				"symlink target %q escapes destination",
				header.Linkname,
			)
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
		if err := createParentWithinRoot(realRoot, destinationPath); err != nil {
			return err
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
	realRoot string,
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
	if err := createParentWithinRoot(realRoot, destinationPath); err != nil {
		return err
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

// createParentWithinRoot creates the parent directory of destinationPath and
// confirms its real, symlink-resolved location stays inside realRoot. Resolving
// the parent with [filepath.EvalSymlinks] follows any symlink an earlier archive
// entry planted, so a crafted archive cannot redirect this write outside the
// extraction root.
func createParentWithinRoot(realRoot string, destinationPath string) error {
	parent := filepath.Dir(destinationPath)
	if err := os.MkdirAll(parent, defaultDirectoryMode); err != nil {
		return wrapError("create tar entry parent", err)
	}
	return verifyRealPathWithinRoot(realRoot, parent)
}

// verifyRealPathWithinRoot resolves candidatePath to its real location and
// rejects it when that location is not realRoot or a descendant of realRoot.
func verifyRealPathWithinRoot(realRoot string, candidatePath string) error {
	realCandidate, err := filepath.EvalSymlinks(candidatePath)
	if err != nil {
		return wrapError("resolve archive entry path", err)
	}
	if !pathWithinRoot(realRoot, realCandidate) {
		return fmt.Errorf("archive entry %q escapes destination", candidatePath)
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
