//go:build !darwin && !linux

package merkle

import "os"

func inodeFromFileInfo(_ os.FileInfo) InodeRef {
	return InodeRef{Device: "", Inode: 0}
}
