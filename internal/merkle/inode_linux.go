//go:build linux

package merkle

import (
	"os"
	"strconv"
	"syscall"
)

// inodeFromFileInfo returns the device and inode numbers from
// info.Sys().(*[syscall.Stat_t]). It formats stat.Dev as an unsigned base-10
// string. It returns a zero InodeRef when info.Sys() does not return a
// *[syscall.Stat_t].
func inodeFromFileInfo(info os.FileInfo) InodeRef {
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return InodeRef{Device: "", Inode: 0}
	}
	return InodeRef{Device: strconv.FormatUint(stat.Dev, 10), Inode: stat.Ino}
}
