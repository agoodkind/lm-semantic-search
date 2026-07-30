// Package sandbox roots a daemon inside one throwaway directory so it cannot
// reach the operator's state, the operator's advisory lock, or the shared
// backends.
//
// It is the single place that decides what isolation means. The sandbox command
// and the live test harnesses all resolve their configuration through it, so
// there is one answer to "what does an isolated daemon read and write" rather
// than one answer per caller.
//
// Every value it supplies is a default. Resolve skips any variable the caller
// already set, so a caller that wants a real backend asks for it and keeps the
// rest of the isolation.
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

// Directory names created under a sandbox root. They mirror the layout the
// daemon builds under its real state root, so a person reading a sandbox root
// sees the same shape they already know.
const (
	stateDirName   = "state"
	configDirName  = "config"
	contextDirName = "context"
	logsDirName    = "logs"

	socketFileName = "daemon.sock"
	logFileName    = "daemon.log"

	// ephemeralDebugListenAddr keeps the debug listener off the fixed port the
	// installed daemon binds. A port of 0 means the kernel assigns a free one,
	// so several sandboxes can run at once. The host is the IPv6 loopback
	// literal because the debug listener parses its host as an IP address and
	// rejects a name, so "localhost" cannot be used here.
	ephemeralDebugListenAddr = "[::1]:0"
)

// Env returns the environment defaults that root a daemon inside dir, in a
// fixed order so a caller can print or apply them reproducibly.
//
// It reads the ambient environment once, to find the real model cache root, and
// otherwise computes only from dir. Returning the table rather than applying it
// lets a test install the same values through t.Setenv, which unwinds per test,
// instead of mutating the process for good.
func Env(dir string) []Var {
	return []Var{
		{Name: "CLAUDE_CONTEXTD_STATE_ROOT", Value: filepath.Join(dir, stateDirName)},
		{Name: "CLAUDE_CONTEXTD_CONFIG_ROOT", Value: filepath.Join(dir, configDirName)},
		{Name: "CLAUDE_CONTEXTD_CONTEXT_ROOT", Value: filepath.Join(dir, contextDirName)},
		{Name: "CLAUDE_CONTEXTD_SOCKET_PATH", Value: filepath.Join(dir, socketFileName)},
		{Name: "CLAUDE_CONTEXTD_LOG_PATH", Value: filepath.Join(dir, logsDirName, logFileName)},
		// The one value that deliberately points outside dir. A downloaded model
		// is checksum-verified and identical for every daemon, so re-fetching it
		// per sandbox would cost a large download and require network on a run
		// that is otherwise fully local.
		{Name: "CLAUDE_CONTEXTD_MODEL_CACHE_ROOT", Value: realModelCacheRoot()},
		// Offline replaces the shared vector store with an on-disk index and the
		// hosted embedder with an in-process one, so a sandbox reaches neither.
		{Name: "CLAUDE_CONTEXT_PROFILE", Value: config.ProfileOffline},
		// The debug listener binds a fixed loopback port, which the installed
		// daemon already holds. Port 0 asks the kernel for a free one, so a
		// sandbox never competes for it and several can run at once.
		{Name: "CLAUDE_CONTEXT_DEBUG_LISTEN_ADDR", Value: ephemeralDebugListenAddr},
	}
}

// realModelCacheRoot resolves the machine's model cache before any sandbox
// value is applied, so the answer is the operator's root rather than one derived
// from a sandbox root. A failure to resolve returns the empty string, which Apply
// then skips, leaving the cache to fall back to the sandbox state root.
func realModelCacheRoot() string {
	resolved, err := config.Default()
	if err != nil {
		return ""
	}
	return resolved.ModelCacheRoot
}

// Apply sets each default that is not already present in the environment and
// reports the values now in effect, whether this call set them or the caller
// did. Skipping a name that is already set is what makes every value an
// override point rather than a forced setting.
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

// Resolve applies the defaults for dir and returns the configuration a daemon
// rooted there runs with, through the same config.Default the installed daemon
// uses. Every flag and variable behaves as it always does; only the starting
// point moved.
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

// Directories returns the paths a sandbox daemon needs to exist before it
// serves, so a caller creates them in one place rather than rediscovering the
// list.
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
