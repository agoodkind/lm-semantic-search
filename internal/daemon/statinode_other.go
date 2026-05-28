//go:build !darwin && !linux

package daemon

import (
	"errors"
	"log/slog"
)

// statInode is a stub for unsupported platforms. The Manager's inode
// detection routes through it; the routine returns an error so callers
// disable inode tracking gracefully on platforms without a unix.Stat_t.
func statInode(path string) (inodeIdentity, error) {
	slog.Warn("statInode unsupported on this platform; inode tracking will fall back to path-only", "path", path)
	return inodeIdentity{}, errors.New("statInode is not supported on this platform")
}
