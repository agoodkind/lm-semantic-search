package merkle

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
)

// errInodeUnsupported reports that the platform does not expose a (device,
// inode) pair for a file.
var errInodeUnsupported = errors.New("inode identity is not supported on this platform")

// StatInode returns the device and inode numbers for path as an InodeRef.
// It calls [os.Stat], which follows symbolic links. It returns an error if
// [os.Stat] fails. It returns an error wrapping errInodeUnsupported if
// inodeFromFileInfo returns a zero InodeRef.
func StatInode(path string) (InodeRef, error) {
	info, err := os.Stat(path)
	if err != nil {
		slog.Error("stat file for inode failed", "path", path, "err", err)
		return InodeRef{Device: "", Inode: 0}, fmt.Errorf("stat %s for inode: %w", path, err)
	}
	ref := inodeFromFileInfo(info)
	if ref.IsZero() {
		slog.Warn("inode identity unsupported; inode tracking falls back to path-only identity", "path", path)
		return InodeRef{Device: "", Inode: 0}, fmt.Errorf("stat %s for inode: %w", path, errInodeUnsupported)
	}
	return ref, nil
}
