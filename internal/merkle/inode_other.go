//go:build !darwin && !linux

package merkle

import "os"

// inodeFromFileInfo ignores its argument and returns a zero InodeRef.
func inodeFromFileInfo(_ os.FileInfo) InodeRef {
	return InodeRef{Device: "", Inode: 0}
}
