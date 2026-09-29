//go:build installlive || updatelive

// Package operatorstate fails a live test run that adds, removes, or changes a
// file in the state roots of an installed LMS daemon.
package operatorstate

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// stateDirectories are the state roots of an installed LMS daemon, relative to
// the home directory. The live tests use isolated roots and must add or change
// no file in these.
var stateDirectories = []string{
	".local/state/lm-semantic-search",
	".local/state/lm-semantic-search-daemon",
}

type stateFile struct {
	size    int64
	modTime time.Time
}

// snapshot records every file under the operator state roots, with its size
// and modification time. A missing root records no file.
func snapshot() (map[string]stateFile, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("resolve home: %w", err)
	}
	files := map[string]stateFile{}
	for _, relative := range stateDirectories {
		root := filepath.Join(home, relative)
		walkErr := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
			if errors.Is(err, fs.ErrNotExist) {
				return nil
			}
			if err != nil {
				return err
			}
			info, err := entry.Info()
			if err != nil {
				return err
			}
			files[path] = stateFile{size: info.Size(), modTime: info.ModTime()}
			return nil
		})
		if walkErr != nil {
			return nil, fmt.Errorf("walk %s: %w", root, walkErr)
		}
	}
	return files, nil
}

// changes lists every path added, removed, or changed between two snapshots.
func changes(before map[string]stateFile, after map[string]stateFile) []string {
	found := []string{}
	for path, file := range after {
		previous, ok := before[path]
		switch {
		case !ok:
			found = append(found, "added "+path)
		case previous.size != file.size || !previous.modTime.Equal(file.modTime):
			found = append(found, "changed "+path)
		}
	}
	for path := range before {
		if _, ok := after[path]; !ok {
			found = append(found, "removed "+path)
		}
	}
	sort.Strings(found)
	return found
}

// RunWithGuard runs the tests and returns exit code 1 when the operator state
// roots differ afterward. Otherwise it returns the exit code of run.
func RunWithGuard(run func() int) int {
	before, err := snapshot()
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "snapshot operator state: %v\n", err)
		return 1
	}
	exitCode := run()
	after, err := snapshot()
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "snapshot operator state: %v\n", err)
		return 1
	}
	if found := changes(before, after); len(found) > 0 {
		_, _ = fmt.Fprintf(os.Stderr, "live tests changed the operator LMS state:\n%s\n", strings.Join(found, "\n"))
		return 1
	}
	return exitCode
}
