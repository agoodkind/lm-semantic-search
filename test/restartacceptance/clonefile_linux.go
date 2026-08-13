//go:build restartacceptance && linux

package restartacceptance

import (
	"errors"
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

func cloneFile(source string, destination string) (cloneErr error) {
	sourceFile, err := os.Open(source)
	if err != nil {
		return fmt.Errorf("open clone source: %w", err)
	}
	defer func() {
		cloneErr = errors.Join(cloneErr, sourceFile.Close())
	}()
	destinationFile, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return fmt.Errorf("create clone destination: %w", err)
	}
	defer func() {
		cloneErr = errors.Join(cloneErr, destinationFile.Close())
	}()
	if err := unix.IoctlFileClone(int(destinationFile.Fd()), int(sourceFile.Fd())); err != nil {
		return fmt.Errorf("clone file with FICLONE: %w", err)
	}
	return nil
}
