//go:build restartacceptance && darwin

package restartacceptance

import "golang.org/x/sys/unix"

func cloneFile(source string, destination string) error {
	return unix.Clonefile(source, destination, 0)
}
