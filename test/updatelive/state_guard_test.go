//go:build updatelive

package updatelive

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

// operatorStateDirectories are the state roots of an installed LMS daemon.
// The live tests use isolated roots and must add or change no file in these.
var operatorStateDirectories = []string{
	".local/state/lm-semantic-search",
	".local/state/lm-semantic-search-daemon",
}

type stateFile struct {
	size    int64
	modTime time.Time
}

// snapshotOperatorState records every file under the operator state roots,
// with its size and modification time. A missing root records no file.
func snapshotOperatorState() (map[string]stateFile, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("resolve home: %w", err)
	}
	files := map[string]stateFile{}
	for _, relative := range operatorStateDirectories {
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

// operatorStateChanges lists every path added, removed, or changed between two
// snapshots.
func operatorStateChanges(before map[string]stateFile, after map[string]stateFile) []string {
	changes := []string{}
	for path, file := range after {
		previous, found := before[path]
		switch {
		case !found:
			changes = append(changes, "added "+path)
		case previous.size != file.size || !previous.modTime.Equal(file.modTime):
			changes = append(changes, "changed "+path)
		}
	}
	for path := range before {
		if _, found := after[path]; !found {
			changes = append(changes, "removed "+path)
		}
	}
	sort.Strings(changes)
	return changes
}

// runWithOperatorStateGuard runs the tests and fails the run when the operator
// state roots differ afterward.
func runWithOperatorStateGuard(run func() int) int {
	before, err := snapshotOperatorState()
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "snapshot operator state: %v\n", err)
		return 1
	}
	exitCode := run()
	after, err := snapshotOperatorState()
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "snapshot operator state: %v\n", err)
		return 1
	}
	if changes := operatorStateChanges(before, after); len(changes) > 0 {
		_, _ = fmt.Fprintf(os.Stderr, "live tests changed the operator LMS state:\n%s\n", strings.Join(changes, "\n"))
		return 1
	}
	return exitCode
}
