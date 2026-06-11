package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
)

// sourceDirMissing reports whether a codebase's source directory is absent.
// It distinguishes a vanished-source failure (a removed worktree or deleted
// directory) from a real build failure so the codebase reads as missing.
func sourceDirMissing(canonicalPath string) bool {
	_, err := os.Stat(canonicalPath)
	return errors.Is(err, os.ErrNotExist)
}

// guardStateRoot rejects any registration whose canonical path covers the
// daemon's StateRoot. Indexing StateRoot would re-enter the daemon's own
// registry and snapshot files; the resulting feedback loop is hard to
// reason about, so we hard-reject at the boundary.
func (manager *Manager) guardStateRoot(canonicalPath string) error {
	stateRoot := strings.TrimRight(filepath.Clean(manager.config.StateRoot), string(filepath.Separator))
	if stateRoot == "" {
		return nil
	}
	if pathCovers(canonicalPath, stateRoot) {
		err := fmt.Errorf("refusing to index %s because it covers daemon state root %s", canonicalPath, stateRoot)
		slog.Error("state-root guard rejected registration", "path", canonicalPath, "state_root", stateRoot, "err", err)
		return err
	}
	return nil
}

// guardFilesystemRoot rejects a registration rooted at the filesystem root.
// Indexing "/" swallows every mount on the host and is never intentional;
// the daemon-resolved-relative-path incident registered "/" exactly this way.
func (manager *Manager) guardFilesystemRoot(canonicalPath string) error {
	_ = manager
	if filepath.Clean(canonicalPath) == string(filepath.Separator) {
		err := fmt.Errorf("refusing to index filesystem root %s", canonicalPath)
		slog.Error("filesystem-root guard rejected registration", "path", canonicalPath, "err", err)
		return err
	}
	return nil
}

// guardDirectory rejects any registration whose canonical path is not a
// directory. Files, sockets, FIFOs, and devices are not indexable
// codebases. A non-existent path is allowed so a future periodic sync can
// converge once the directory appears.
func (manager *Manager) guardDirectory(canonicalPath string) error {
	_ = manager
	info, err := os.Stat(canonicalPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		slog.Error("stat codebase root failed", "path", canonicalPath, "err", err)
		return fmt.Errorf("stat codebase root %s: %w", canonicalPath, err)
	}
	if !info.IsDir() {
		err := fmt.Errorf("codebase root %s is not a directory", canonicalPath)
		slog.Error("directory guard rejected registration", "path", canonicalPath, "err", err)
		return err
	}
	return nil
}

// detectInodeTrackingDisabled returns true when the filesystem under
// canonicalPath looks unsuitable for (device, inode) tracking. The check is
// best-effort: two stats of the same path returning different inode
// numbers indicate an unstable inode source (some NFS or FUSE backends).
// A false positive only forces path-only tracking; correctness does not
// depend on this signal.
func detectInodeTrackingDisabled(ctx context.Context, canonicalPath string) bool {
	first, err := statInode(canonicalPath)
	if err != nil {
		slog.WarnContext(ctx, "stat codebase root for inode check failed", "path", canonicalPath, "err", err)
		return true
	}
	second, err := statInode(canonicalPath)
	if err != nil {
		slog.WarnContext(ctx, "second stat for inode stability failed", "path", canonicalPath, "err", err)
		return true
	}
	if first != second {
		slog.WarnContext(ctx, "inode unstable across consecutive stats; disabling inode tracking", "path", canonicalPath)
		return true
	}
	return false
}

// inodeIdentity is the platform-uniform pair of values that identifies one
// inode for the convergence decision table. The device token is the host's
// device id rendered as a base-10 string so the type stays comparable
// without forcing a signed-to-unsigned cast that the integer-overflow lint
// rejects. The inode field stays uint64 because every supported platform
// reports Stat_t.Ino in that width.
type inodeIdentity struct {
	device string
	inode  uint64
}
