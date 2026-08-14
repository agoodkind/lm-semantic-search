//go:build restartacceptance

package sandboxharness

import (
	"archive/tar"
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"time"
)

const cleanupTimeout = 2 * time.Minute

// RequiredBytes returns the 125 percent free-space reserve for source sizes.
func RequiredBytes(sizes []int64) int64 {
	var total int64
	for _, size := range sizes {
		total += size
	}
	return (total*5 + 3) / 4
}

// RequireFreeSpace rejects available space below the restore reserve.
func RequireFreeSpace(available int64, sizes []int64) error {
	required := RequiredBytes(sizes)
	if available < required {
		return fmt.Errorf("free space %d bytes is less than required %d bytes", available, required)
	}
	return nil
}

// VerifyChecksums verifies an exact checksum manifest and every expected file.
func VerifyChecksums(root string, manifestName string, expected map[string]string) error {
	manifest, err := openRegular(filepath.Join(root, manifestName))
	if err != nil {
		return fmt.Errorf("open checksum manifest: %w", err)
	}
	defer func() { _ = manifest.Close() }()
	seen := make(map[string]struct{}, len(expected))
	scanner := bufio.NewScanner(manifest)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != 2 {
			return fmt.Errorf("invalid checksum manifest entry %q", scanner.Text())
		}
		name := strings.TrimPrefix(fields[1], "*")
		if filepath.IsAbs(name) || filepath.Clean(name) != name || strings.Contains(name, string(filepath.Separator)) {
			return fmt.Errorf("checksum entry %q is not a literal file name", name)
		}
		want, exists := expected[name]
		if !exists {
			return fmt.Errorf("checksum manifest contains unexpected entry %q", name)
		}
		if _, duplicate := seen[name]; duplicate {
			return fmt.Errorf("checksum manifest contains duplicate entry %q", name)
		}
		manifestHash := strings.ToLower(fields[0])
		if manifestHash != want {
			return fmt.Errorf("manifest checksum for %q is %s, want %s", name, manifestHash, want)
		}
		got, hashErr := hashFile(filepath.Join(root, name))
		if hashErr != nil {
			return hashErr
		}
		if got != want {
			return fmt.Errorf("checksum for %q is %s, want %s", name, got, want)
		}
		seen[name] = struct{}{}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read checksum manifest: %w", err)
	}
	missing := make([]string, 0)
	for name := range expected {
		if _, exists := seen[name]; !exists {
			missing = append(missing, name)
		}
	}
	if len(missing) != 0 {
		slices.Sort(missing)
		return fmt.Errorf("checksum manifest is missing entries: %s", strings.Join(missing, ", "))
	}
	return nil
}

// RestoreArchiveReader extracts a regular-file tar tree and makes it read-only.
func RestoreArchiveReader(ctx context.Context, archive io.Reader, destination string) error {
	if err := context.Cause(ctx); err != nil {
		return fmt.Errorf("restore archive: %w", err)
	}
	if _, err := os.Lstat(destination); err == nil {
		entries, readErr := os.ReadDir(destination)
		if readErr != nil {
			return fmt.Errorf("inspect restore destination: %w", readErr)
		}
		if len(entries) != 0 {
			return fmt.Errorf("restore destination %q is not empty", destination)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("inspect restore destination: %w", err)
	}
	if err := os.MkdirAll(destination, 0o700); err != nil {
		return fmt.Errorf("create restore destination: %w", err)
	}
	reader := tar.NewReader(&contextReader{ctx: ctx, reader: archive})
	for {
		header, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("read restore archive: %w", err)
		}
		name := filepath.Clean(header.Name)
		if name == "." {
			if header.Typeflag != tar.TypeDir {
				return fmt.Errorf("restore archive root entry %q is not a directory", header.Name)
			}
			continue
		}
		if filepath.IsAbs(header.Name) || name == ".." || strings.HasPrefix(name, ".."+string(filepath.Separator)) {
			return fmt.Errorf("restore archive entry %q escapes destination", header.Name)
		}
		target := filepath.Join(destination, name)
		if !pathWithin(destination, target) {
			return fmt.Errorf("restore archive entry %q escapes destination", header.Name)
		}
		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o700); err != nil {
				return fmt.Errorf("create restored directory: %w", err)
			}
		case tar.TypeReg, tar.TypeRegA:
			if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
				return fmt.Errorf("create restored parent: %w", err)
			}
			file, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
			if err != nil {
				return fmt.Errorf("create restored file: %w", err)
			}
			_, copyErr := io.CopyN(file, &contextReader{ctx: ctx, reader: reader}, header.Size)
			closeErr := file.Close()
			if copyErr != nil {
				return fmt.Errorf("extract restored file: %w", copyErr)
			}
			if closeErr != nil {
				return fmt.Errorf("close restored file: %w", closeErr)
			}
		default:
			return fmt.Errorf("restore archive entry %q has unsupported type %d", header.Name, header.Typeflag)
		}
	}
	if err := filepath.WalkDir(destination, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := context.Cause(ctx); err != nil {
			return err
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("restored source contains symlink %q", path)
		}
		mode := fs.FileMode(0o400)
		if entry.IsDir() {
			mode = 0o500
		}
		return os.Chmod(path, mode)
	}); err != nil {
		return fmt.Errorf("make restored source immutable: %w", err)
	}
	return nil
}

// CloneTree creates a writable copy-on-write clone of an immutable source tree.
func CloneTree(source string, destination string) error {
	if _, err := os.Lstat(destination); !errors.Is(err, os.ErrNotExist) {
		if err == nil {
			return fmt.Errorf("clone destination %q already exists", destination)
		}
		return fmt.Errorf("inspect clone destination: %w", err)
	}
	return filepath.WalkDir(source, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return fmt.Errorf("resolve clone path: %w", err)
		}
		target := filepath.Join(destination, relative)
		if entry.IsDir() {
			return os.MkdirAll(target, 0o700)
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("immutable source contains symlink %q", path)
		}
		if err := cloneFile(path, target); err != nil {
			return fmt.Errorf("clone %q: %w", path, err)
		}
		return os.Chmod(target, 0o600)
	})
}

// RemoveTree removes one exact child of parent with a bounded retry loop.
func RemoveTree(ctx context.Context, parent string, root string, requiredName string) error {
	cleanParent := filepath.Clean(parent)
	cleanRoot := filepath.Clean(root)
	if filepath.Base(cleanRoot) != requiredName || requiredName == "" {
		return fmt.Errorf("cleanup root %q does not match required name %q", cleanRoot, requiredName)
	}
	if filepath.Dir(cleanRoot) != cleanParent {
		return fmt.Errorf("cleanup root %q is not an exact child of %q", cleanRoot, cleanParent)
	}
	cleanupContext, cancel := context.WithTimeout(ctx, cleanupTimeout)
	defer cancel()
	if err := filepath.WalkDir(cleanRoot, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return nil
		}
		return os.Chmod(path, 0o700)
	}); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("make cleanup tree writable: %w", err)
	}
	for {
		if err := os.RemoveAll(cleanRoot); err == nil {
			return nil
		} else if !errors.Is(err, syscall.ENOTEMPTY) {
			return fmt.Errorf("remove cleanup tree: %w", err)
		}
		select {
		case <-cleanupContext.Done():
			return fmt.Errorf("remove cleanup tree: %w", context.Cause(cleanupContext))
		case <-time.After(100 * time.Millisecond):
		}
	}
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (reader *contextReader) Read(body []byte) (int, error) {
	if err := context.Cause(reader.ctx); err != nil {
		return 0, err
	}
	return reader.reader.Read(body)
}

func pathWithin(root string, path string) bool {
	relative, err := filepath.Rel(filepath.Clean(root), filepath.Clean(path))
	if err != nil {
		return false
	}
	return relative != "." && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator))
}

func openRegular(path string) (*os.File, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return nil, fmt.Errorf("path %q is not a regular file", path)
	}
	return os.Open(path)
}

func hashFile(path string) (string, error) {
	file, err := openRegular(path)
	if err != nil {
		return "", fmt.Errorf("open checksum input %q: %w", path, err)
	}
	defer func() { _ = file.Close() }()
	digest := sha256.New()
	if _, err := io.Copy(digest, file); err != nil {
		return "", fmt.Errorf("hash checksum input %q: %w", path, err)
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}
