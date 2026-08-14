//go:build restartacceptance

package sandboxharness

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"time"
)

var composeProjectPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{2,62}$`)

// CommandRunner runs one command with explicit environment overrides.
type CommandRunner interface {
	Run(context.Context, map[string]string, string, ...string) ([]byte, error)
}

// ComposeProject owns one tagged Docker Compose lifecycle.
type ComposeProject struct {
	Runner         CommandRunner
	Name           string
	File           string
	Environment    map[string]string
	CleanupTimeout time.Duration
}

// Run starts the project, waits for health, runs work, and always cleans it.
func (project ComposeProject) Run(ctx context.Context, work func(context.Context) error) (runErr error) {
	if project.Runner == nil || work == nil {
		return fmt.Errorf("compose project operations are incomplete")
	}
	if !composeProjectPattern.MatchString(project.Name) {
		return fmt.Errorf("compose project name %q is unsafe", project.Name)
	}
	if !filepath.IsAbs(project.File) {
		return fmt.Errorf("compose file %q is not absolute", project.File)
	}
	cleanupTimeout := project.CleanupTimeout
	if cleanupTimeout <= 0 {
		cleanupTimeout = 2 * time.Minute
	}
	defer func() {
		cleanupContext, cancel := context.WithTimeout(context.Background(), cleanupTimeout)
		defer cancel()
		_, err := project.Runner.Run(
			cleanupContext,
			project.Environment,
			"docker",
			"compose",
			"-p",
			project.Name,
			"-f",
			project.File,
			"down",
			"--volumes",
			"--remove-orphans",
		)
		if err != nil {
			runErr = errors.Join(runErr, fmt.Errorf("clean compose project %q: %w", project.Name, err))
		}
	}()
	if _, err := project.Runner.Run(
		ctx,
		project.Environment,
		"docker",
		"compose",
		"-p",
		project.Name,
		"-f",
		project.File,
		"up",
		"-d",
		"--wait",
	); err != nil {
		return fmt.Errorf("start compose project %q: %w", project.Name, err)
	}
	if err := work(ctx); err != nil {
		return fmt.Errorf("run compose project %q: %w", project.Name, err)
	}
	return nil
}
