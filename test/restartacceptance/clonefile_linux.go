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
	destinationFile, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		openErr := fmt.Errorf("create clone destination: %w", err)
		if closeErr := sourceFile.Close(); closeErr != nil {
			return errors.Join(openErr, fmt.Errorf("close clone source: %w", closeErr))
		}
		return openErr
	}
	defer func() {
		if err := destinationFile.Close(); err != nil {
			cloneErr = errors.Join(cloneErr, fmt.Errorf("close clone destination: %w", err))
		}
		if err := sourceFile.Close(); err != nil {
			cloneErr = errors.Join(cloneErr, fmt.Errorf("close clone source: %w", err))
		}
		if cloneErr == nil {
			return
		}
		if err := os.Remove(destination); err != nil && !errors.Is(err, os.ErrNotExist) {
			cloneErr = errors.Join(cloneErr, fmt.Errorf("remove failed clone destination: %w", err))
		}
	}()
	if err := unix.IoctlFileClone(int(destinationFile.Fd()), int(sourceFile.Fd())); err != nil {
		return fmt.Errorf("clone with FICLONE from %q to %q: %w", source, destination, err)
	}
	return nil
}
