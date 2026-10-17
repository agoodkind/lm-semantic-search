package daemon

import (
	"fmt"
	"log/slog"

	"goodkind.io/lm-semantic-search/internal/merkle"
)

// statInode calls merkle.StatInode to read the device and inode numbers for
// path and returns them as an inodeIdentity. It returns an error if
// merkle.StatInode fails.
func statInode(path string) (inodeIdentity, error) {
	ref, err := merkle.StatInode(path)
	if err != nil {
		slog.Warn("stat inode for daemon identity failed", "path", path, "err", err)
		return inodeIdentity{device: "", inode: 0}, fmt.Errorf("stat inode for %s: %w", path, err)
	}
	return inodeIdentity{device: ref.Device, inode: ref.Inode}, nil
}
