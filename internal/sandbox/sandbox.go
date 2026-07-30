// Package sandbox roots a daemon inside one throwaway directory so it cannot
// reach the operator's state, the operator's advisory lock, or the shared
// backends.
//
// Every value is a default: a name already set in the environment is left
// alone, so a caller can ask for a real backend and keep the rest of the
// isolation. The sandbox command and the live suites all resolve through here.
package sandbox

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"goodkind.io/lm-semantic-search/internal/config"
)

// Var is one environment default the sandbox supplies.
type Var struct {
	Name  string
	Value string
}

const (
	stateDirName   = "state"
	configDirName  = "config"
	contextDirName = "context"
	logsDirName    = "logs"

	socketFileName = "daemon.sock"
	logFileName    = "daemon.log"

	// Port 0 asks the kernel for a free port, since the installed daemon
	// already holds the fixed one. The host is an address literal because the
	// debug listener parses it as an IP and rejects a name.
	ephemeralDebugListenAddr = "[::1]:0"
)

// Env returns the defaults that root a daemon inside dir, in a fixed order.
// Returning the table rather than applying it lets a test install the same
// values through t.Setenv, which unwinds when the test ends.
func Env(dir string) []Var {
	return []Var{
		{Name: "CLAUDE_CONTEXTD_STATE_ROOT", Value: filepath.Join(dir, stateDirName)},
		{Name: "CLAUDE_CONTEXTD_CONFIG_ROOT", Value: filepath.Join(dir, configDirName)},
		{Name: "CLAUDE_CONTEXTD_CONTEXT_ROOT", Value: filepath.Join(dir, contextDirName)},
		{Name: "CLAUDE_CONTEXTD_SOCKET_PATH", Value: filepath.Join(dir, socketFileName)},
		{Name: "CLAUDE_CONTEXTD_LOG_PATH", Value: filepath.Join(dir, logsDirName, logFileName)},
		// Deliberately outside dir: a checksum-verified model is identical for
		// every daemon, so a sandbox reuses it instead of downloading it again.
		{Name: "CLAUDE_CONTEXTD_MODEL_CACHE_ROOT", Value: realModelCacheRoot()},
		// Offline swaps in an on-disk index and an in-process embedder, so a
		// sandbox reaches neither shared backend.
		{Name: "CLAUDE_CONTEXT_PROFILE", Value: config.ProfileOffline},
		{Name: "CLAUDE_CONTEXT_DEBUG_LISTEN_ADDR", Value: ephemeralDebugListenAddr},
	}
}

// realModelCacheRoot reads the machine's cache before any sandbox value is
// applied, so the answer is the operator's root rather than one derived from
// dir. An empty result is skipped by Apply and falls back to the state root.
func realModelCacheRoot() string {
	resolved, err := config.Default()
	if err != nil {
		return ""
	}
	return resolved.ModelCacheRoot
}

// Apply sets each default not already in the environment and reports the values
// now in effect, whether this call set them or the caller did.
func Apply(dir string) ([]Var, error) {
	effective := make([]Var, 0, len(Env(dir)))
	for _, variable := range Env(dir) {
		if variable.Value == "" {
			continue
		}
		if existing, found := os.LookupEnv(variable.Name); found {
			effective = append(effective, Var{Name: variable.Name, Value: existing})
			continue
		}
		if err := os.Setenv(variable.Name, variable.Value); err != nil {
			slog.Error("set sandbox environment default failed", "name", variable.Name, "err", err)
			return nil, fmt.Errorf("set %s: %w", variable.Name, err)
		}
		effective = append(effective, variable)
	}
	return effective, nil
}

// Resolve applies the defaults for dir and returns the configuration through
// the same config.Default the installed daemon uses, so only the starting point
// differs.
func Resolve(dir string) (config.Config, error) {
	if _, err := Apply(dir); err != nil {
		return config.Config{}, err
	}
	resolved, err := config.Default()
	if err != nil {
		slog.Error("resolve sandbox config failed", "root", dir, "err", err)
		return config.Config{}, fmt.Errorf("resolve sandbox config: %w", err)
	}
	return resolved, nil
}

// Directories returns the paths a daemon needs to exist before it serves.
func Directories(resolved config.Config) []string {
	return []string{
		resolved.StateRoot,
		resolved.ConfigRoot,
		resolved.ContextRoot,
		resolved.LogsDir,
		resolved.SocketsDir,
		resolved.MerkleDir,
		resolved.LocksDir,
		resolved.ChunksDir,
		resolved.GraphDir,
	}
}
