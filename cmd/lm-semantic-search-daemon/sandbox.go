package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/orphanguard"
	"goodkind.io/lm-semantic-search/internal/sandbox"
	"goodkind.io/lm-semantic-search/internal/store"
)

// sandboxCommandName is the argument that selects a throwaway daemon instead of
// the installed one.
const sandboxCommandName = "sandbox"

// sandboxRootFlag names the flag that keeps a chosen root rather than making and
// removing a temporary one.
const sandboxRootFlag = "--root"

// sandboxRootPattern prefixes the temporary root. Temporary directories on macOS
// live under a long path, and a unix socket path is capped near 104 bytes, so the
// root is created under /tmp where the socket beneath it still fits.
const (
	sandboxRootPattern = "lms-sandbox-"
	sandboxRootParent  = "/tmp"
	sandboxDirMode     = 0o700
)

// runSandbox serves a throwaway daemon rooted in its own directory and returns
// when the process is interrupted. It is one process: it starts no child, writes
// no pid file, and watches nothing, so the daemon cannot outlive this call.
//
// Isolation comes from where the config is rooted, not from a reduced daemon.
// Everything below the resolve is the same code path the installed daemon runs.
func runSandbox(rootContext context.Context, arguments []string) error {
	root, keepRoot, err := resolveSandboxRoot(arguments)
	if err != nil {
		return err
	}
	if !keepRoot {
		defer removeSandboxRoot(root)
	}

	resolved, err := sandbox.Resolve(root)
	if err != nil {
		slog.ErrorContext(rootContext, "resolve sandbox configuration failed", "root", root, "err", err)
		return fmt.Errorf("resolve sandbox configuration: %w", err)
	}
	for _, directory := range sandbox.Directories(resolved) {
		if err := store.EnsureDir(directory); err != nil {
			slog.ErrorContext(rootContext, "create sandbox directory failed", "path", directory, "err", err)
			return fmt.Errorf("create sandbox directory %s: %w", directory, err)
		}
	}

	if err := writeSandboxBanner(os.Stdout, root, resolved, keepRoot); err != nil {
		return err
	}

	// Ctrl-C, a kill, and a closed terminal all signal this process, and serve
	// already returns on those. A launcher that dies without signalling leaves
	// nothing to notice, so watch for that case too and stop rather than keep
	// serving with no one left to stop us. The installed daemon is meant to
	// outlive its launcher and never arms this.
	serveContext, stopServing := context.WithCancel(rootContext)
	defer stopServing()
	goSafe(serveContext, func() { orphanguard.Watch(serveContext, stopServing) })

	return serve(serveContext, resolved)
}

// resolveSandboxRoot returns the directory to root the daemon in and whether it
// belongs to the caller. A caller-supplied root is left in place on exit, since
// removing a directory the caller named would destroy state they chose to keep.
func resolveSandboxRoot(arguments []string) (root string, keepRoot bool, err error) {
	supplied, err := parseSandboxRootFlag(arguments)
	if err != nil {
		return "", false, err
	}
	if supplied != "" {
		absolute, absErr := filepath.Abs(supplied)
		if absErr != nil {
			slog.Error("resolve sandbox root failed", "path", supplied, "err", absErr)
			return "", false, fmt.Errorf("resolve sandbox root %s: %w", supplied, absErr)
		}
		if mkErr := store.EnsureDir(absolute); mkErr != nil {
			slog.Error("create sandbox root failed", "path", absolute, "err", mkErr)
			return "", false, fmt.Errorf("create sandbox root %s: %w", absolute, mkErr)
		}
		return absolute, true, nil
	}
	created, tempErr := os.MkdirTemp(sandboxRootParent, sandboxRootPattern)
	if tempErr != nil {
		slog.Error("create sandbox root failed", "parent", sandboxRootParent, "err", tempErr)
		return "", false, fmt.Errorf("create sandbox root under %s: %w", sandboxRootParent, tempErr)
	}
	if chmodErr := os.Chmod(created, sandboxDirMode); chmodErr != nil {
		slog.Warn("restrict sandbox root permissions failed", "path", created, "err", chmodErr)
	}
	return created, false, nil
}

// parseSandboxRootFlag reads the one flag this command owns. Every other setting
// reaches the daemon through the environment and the flags the daemon already
// parses, so this deliberately understands nothing else.
func parseSandboxRootFlag(arguments []string) (string, error) {
	for index := range arguments {
		argument := arguments[index]
		if value, found := trimFlagPrefix(argument, sandboxRootFlag+"="); found {
			return value, nil
		}
		if argument != sandboxRootFlag {
			continue
		}
		if index+1 >= len(arguments) {
			err := fmt.Errorf("%s requires a directory path", sandboxRootFlag)
			slog.Error("parse sandbox arguments failed", "err", err)
			return "", err
		}
		return arguments[index+1], nil
	}
	return "", nil
}

func trimFlagPrefix(argument string, prefix string) (string, bool) {
	if len(argument) <= len(prefix) || argument[:len(prefix)] != prefix {
		return "", false
	}
	return argument[len(prefix):], true
}

// removeSandboxRoot deletes a root this command created. It re-derives that the
// path is one of ours rather than trusting the caller, because this is the only
// recursive delete in the daemon and a wrong path here would remove a directory
// somebody cares about. A failure is reported and not returned, since the daemon
// has already stopped and a leftover temporary directory is not worth a nonzero
// exit.
func removeSandboxRoot(root string) {
	cleaned := filepath.Clean(root)
	if filepath.Dir(cleaned) != sandboxRootParent ||
		!strings.HasPrefix(filepath.Base(cleaned), sandboxRootPattern) {
		slog.Warn("refusing to remove a path this sandbox did not create", "path", cleaned)
		return
	}
	if err := os.RemoveAll(cleaned); err != nil {
		slog.Warn("remove sandbox root failed", "path", cleaned, "err", err)
	}
}

// writeSandboxBanner prints where this daemon put everything, so a second
// terminal can reach it and a reader can confirm at a glance that nothing points
// at the installed daemon's state.
func writeSandboxBanner(writer io.Writer, root string, resolved config.Config, keepRoot bool) error {
	rootNote := "removed on exit"
	if keepRoot {
		rootNote = "kept"
	}
	lines := []struct {
		label string
		value string
	}{
		{label: "root", value: root + "  (" + rootNote + ")"},
		{label: "socket", value: resolved.SocketPath},
		{label: "state", value: resolved.StateRoot},
		{label: "context", value: resolved.ContextRoot},
		{label: "models", value: resolved.ModelCacheRoot},
		{label: "profile", value: resolved.Profile},
		{label: "store", value: string(resolved.IndexBackend)},
		{label: "embedder", value: string(resolved.EmbeddingProvider)},
	}
	for _, line := range lines {
		if _, err := fmt.Fprintf(writer, "%-9s %s\n", line.label, line.value); err != nil {
			slog.Error("write sandbox banner failed", "err", err)
			return fmt.Errorf("write sandbox banner: %w", err)
		}
	}
	if _, err := fmt.Fprint(writer, "serving. ctrl-c to stop.\n"); err != nil {
		slog.Error("write sandbox banner failed", "err", err)
		return fmt.Errorf("write sandbox banner: %w", err)
	}
	return nil
}
