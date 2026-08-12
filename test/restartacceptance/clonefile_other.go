//go:build restartacceptance && !darwin

package restartacceptance

import "fmt"

func cloneFile(_, _ string) error {
	return fmt.Errorf("APFS clonefile semantics are unavailable on this platform")
}
