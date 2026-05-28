//go:build darwin

package daemon

import (
	"fmt"
	"log/slog"
	"strconv"

	"golang.org/x/sys/unix"
)

// statInode returns the (device, inode) identifier for the file at path.
// macOS reports Stat_t.Dev as a signed 32-bit value; widening through
// int64 and rendering as a string keeps the comparison stable without a
// signed-to-unsigned cast.
func statInode(path string) (inodeIdentity, error) {
	var stat unix.Stat_t
	if err := unix.Stat(path, &stat); err != nil {
		slog.Error("unix.Stat failed", "path", path, "err", err)
		return inodeIdentity{}, fmt.Errorf("unix.Stat %s: %w", path, err)
	}
	device := strconv.FormatInt(int64(stat.Dev), 10)
	return inodeIdentity{device: device, inode: stat.Ino}, nil
}
