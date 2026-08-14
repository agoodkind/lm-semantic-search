//go:build restartacceptance && darwin

package sandboxharness

import "golang.org/x/sys/unix"

func cloneFile(source string, destination string) error {
	return unix.Clonefile(source, destination, 0)
}
