package gitworktree_test

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/internal/gitworktree"
)

const (
	headTestMainBranch    = "main"
	headTestLinkedBranch  = "linked"
	headTestTrackedFile   = "tracked.txt"
	headTestDirectoryMode = 0o755
	headTestFileMode      = 0o644
)

func headTestGit(t *testing.T, directory string, arguments ...string) string {
	t.Helper()

	command := exec.CommandContext(
		context.Background(),
		"git",
		append([]string{"-C", directory}, arguments...)...,
	)
	command.Env = append(
		os.Environ(),
		"GIT_AUTHOR_NAME=head-test",
		"GIT_AUTHOR_EMAIL=head-test@example.invalid",
		"GIT_COMMITTER_NAME=head-test",
		"GIT_COMMITTER_EMAIL=head-test@example.invalid",
	)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("The git command with arguments %s in directory %s failed with error %v and output %s.", strings.Join(arguments, " "), directory, err, output)
	}
	return strings.TrimSpace(string(output))
}

func headTestCommit(t *testing.T, directory string, content string) string {
	t.Helper()

	path := filepath.Join(directory, headTestTrackedFile)
	if err := os.WriteFile(path, []byte(content), headTestFileMode); err != nil {
		t.Fatalf("Writing the fixture file at %s failed with error %v.", path, err)
	}
	headTestGit(t, directory, "add", "--all")
	headTestGit(t, directory, "-c", "commit.gpgsign=false", "commit", "--quiet", "--message", content)
	return headTestGit(t, directory, "rev-parse", "HEAD")
}

func newHeadTestRepository(t *testing.T) (string, string) {
	t.Helper()

	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("Resolving symlinks of the temporary directory failed with error %v.", err)
	}
	repository := filepath.Join(base, "repository")
	if err := os.Mkdir(repository, headTestDirectoryMode); err != nil {
		t.Fatalf("Creating the repository directory %s failed with error %v.", repository, err)
	}
	headTestGit(t, repository, "init", "--quiet", "--initial-branch", headTestMainBranch)
	headTestCommit(t, repository, "first")
	worktree := filepath.Join(base, headTestLinkedBranch)
	headTestGit(t, repository, "worktree", "add", "--quiet", "-b", headTestLinkedBranch, worktree)
	return repository, worktree
}

func requireHeadCommit(t *testing.T, worktreeRoot string) {
	t.Helper()

	want := headTestGit(t, worktreeRoot, "rev-parse", "HEAD")
	got, resolved := gitworktree.HeadCommit(worktreeRoot)
	if !resolved || got != want {
		t.Fatalf("HeadCommit for worktree %s returned commit %q with resolved flag %t instead of the resolved commit %q from git rev-parse HEAD.", worktreeRoot, got, resolved, want)
	}
}

func requireNoLooseRef(t *testing.T, repository string, branch string) {
	t.Helper()

	looseRef := filepath.Join(repository, ".git", "refs", "heads", branch)
	if _, err := os.Stat(looseRef); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("The loose ref file %s exists after git pack-refs --all. The stat error was %v.", looseRef, err)
	}
}

func TestHeadCommitResolvesLooseBranchRef(t *testing.T) {
	repository, worktree := newHeadTestRepository(t)
	mainCommit := headTestCommit(t, repository, "second on main")
	linkedCommit := headTestCommit(t, worktree, "second on linked")
	if mainCommit == linkedCommit {
		t.Fatalf("The main branch and the linked branch both have commit %q, but the fixture requires different commits.", mainCommit)
	}

	requireHeadCommit(t, repository)
	requireHeadCommit(t, worktree)
}

func TestHeadCommitResolvesDetachedHead(t *testing.T) {
	repository, worktree := newHeadTestRepository(t)
	headTestCommit(t, worktree, "second on linked")
	headTestGit(t, worktree, "checkout", "--quiet", "--detach")
	headTestGit(t, repository, "checkout", "--quiet", "--detach")

	requireHeadCommit(t, worktree)
	requireHeadCommit(t, repository)
}

func TestHeadCommitResolvesPackedRef(t *testing.T) {
	repository, worktree := newHeadTestRepository(t)
	headTestCommit(t, repository, "second on main")
	headTestCommit(t, worktree, "second on linked")
	headTestGit(t, repository, "pack-refs", "--all")
	requireNoLooseRef(t, repository, headTestMainBranch)
	requireNoLooseRef(t, repository, headTestLinkedBranch)

	requireHeadCommit(t, repository)
	requireHeadCommit(t, worktree)
}

func TestHeadCommitReportsUnresolvedHead(t *testing.T) {
	plainDirectory := t.TempDir()
	if commit, resolved := gitworktree.HeadCommit(plainDirectory); resolved {
		t.Fatalf("HeadCommit for directory %s outside a Git repository resolved commit %q.", plainDirectory, commit)
	}

	repository := filepath.Join(t.TempDir(), "unborn")
	if err := os.Mkdir(repository, headTestDirectoryMode); err != nil {
		t.Fatalf("Creating the repository directory %s failed with error %v.", repository, err)
	}
	headTestGit(t, repository, "init", "--quiet", "--initial-branch", headTestMainBranch)
	if commit, resolved := gitworktree.HeadCommit(repository); resolved {
		t.Fatalf("HeadCommit for repository %s with no commit resolved commit %q.", repository, commit)
	}
}
