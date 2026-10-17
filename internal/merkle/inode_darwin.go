//go:build darwin

package merkle

import (
	"os"
	"strconv"
	"syscall"
)

func inodeFromFileInfo(info os.FileInfo) InodeRef {
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return InodeRef{Device: "", Inode: 0}
	}
	return InodeRef{Device: strconv.FormatInt(int64(stat.Dev), 10), Inode: stat.Ino}
}
