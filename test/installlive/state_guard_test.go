//go:build installlive

package installlive

import (
	"os"

	"goodkind.io/lm-semantic-search/internal/sandbox"
)

// isolatedEnvironment returns the process environment with every sandbox.Env
// root, including the state root, set under root. Callers create root in /tmp.
// The kernel caps a socket path near 104 bytes, and the platform temp
// directory is longer.
func isolatedEnvironment(root string) []string {
	environment := os.Environ()
	for _, variable := range sandbox.Env(root) {
		if variable.Value != "" {
			environment = append(environment, variable.Name+"="+variable.Value)
		}
	}
	return environment
}
