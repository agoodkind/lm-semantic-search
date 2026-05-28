//go:build linux

package daemon

import (
	"fmt"
	"log/slog"
	"strconv"

	"golang.org/x/sys/unix"
)

// statInode returns the (device, inode) identifier for the file at path.
// Linux reports Stat_t.Dev as uint64; the value is rendered as a base-10
// string for the same comparable-token reason as the darwin variant.
func statInode(path string) (inodeIdentity, error) {
	var stat unix.Stat_t
	if err := unix.Stat(path, &stat); err != nil {
		slog.Error("unix.Stat failed", "path", path, "err", err)
		return inodeIdentity{}, fmt.Errorf("unix.Stat %s: %w", path, err)
	}
	device := strconv.FormatUint(stat.Dev, 10)
	return inodeIdentity{device: device, inode: stat.Ino}, nil
}
