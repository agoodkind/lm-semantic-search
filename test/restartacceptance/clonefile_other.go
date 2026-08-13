//go:build restartacceptance && !darwin && !linux

package restartacceptance

import "fmt"

func cloneFile(_, _ string) error {
	return fmt.Errorf("copy-on-write file cloning is unavailable on this platform")
}
