package daemon_test

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	sourceTestStartingCommit = "1f2e3d4c5b6a79880716253443526170deadbeef"
	sourceTestCloseTimeout   = 10 * time.Second
	sourceTestSyncIntervalMS = 300000
	sourceTestStatusTimeout  = 30 * time.Second
	sourceTestPollInterval   = 50 * time.Millisecond
)

type worktreeSourceScene struct {
	config     config.Config
	manager    *daemon.Manager
	server     *daemon.GRPCServer
	repository string
	worktreeA  string
	worktreeB  string
}

func newSourceTestConfig(t *testing.T) config.Config {
	t.Helper()

	stateRoot := t.TempDir()
	cfg := config.Config{
		StateRoot:         stateRoot,
		SocketPath:        filepath.Join(stateRoot, "sockets", "lm-semantic-search-daemon.sock"),
		RegistryPath:      filepath.Join(stateRoot, "registry.json"),
		JobsPath:          filepath.Join(stateRoot, "jobs.jsonl"),
		EventsPath:        filepath.Join(stateRoot, "events.jsonl"),
		LogsDir:           filepath.Join(stateRoot, "logs"),
		LogPath:           filepath.Join(stateRoot, "logs", "lm-semantic-search-daemon.log"),
		MerkleDir:         filepath.Join(stateRoot, "merkle"),
		LocksDir:          filepath.Join(stateRoot, "locks"),
		SocketsDir:        filepath.Join(stateRoot, "sockets"),
		ChunksDir:         filepath.Join(stateRoot, "chunks"),
		GraphDir:          filepath.Join(stateRoot, "graph"),
		ContextRoot:       filepath.Join(stateRoot, "context"),
		EmbeddingProvider: "OpenAI",
		EmbeddingModel:    "nvidia/NV-EmbedCode-7b-v1",
		HybridMode:        true,
		SyncIntervalMS:    sourceTestSyncIntervalMS,
	}
	for _, path := range []string{cfg.StateRoot, cfg.LogsDir, cfg.MerkleDir, cfg.LocksDir, cfg.SocketsDir, cfg.ChunksDir, cfg.GraphDir, cfg.ContextRoot} {
		if err := store.EnsureDir(path); err != nil {
			t.Fatalf("State directory creation failed for %s with error %v.", path, err)
		}
	}
	if err := store.WriteRegistry(cfg.RegistryPath, model.RegistryFile{}); err != nil {
		t.Fatalf("The empty registry write to %s failed with error %v.", cfg.RegistryPath, err)
	}
	return cfg
}

func runSourceTestGit(t *testing.T, directory string, arguments ...string) {
	t.Helper()

	command := exec.CommandContext(
		context.Background(),
		"git",
		append([]string{"-C", directory}, arguments...)...,
	)
	command.Env = append(
		os.Environ(),
		"GIT_AUTHOR_NAME=source-test",
		"GIT_AUTHOR_EMAIL=source-test@example.invalid",
		"GIT_COMMITTER_NAME=source-test",
		"GIT_COMMITTER_EMAIL=source-test@example.invalid",
	)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("The git command with arguments %s failed in directory %s with error %v and command output %s.", strings.Join(arguments, " "), directory, err, output)
	}
}

func newSourceTestRepository(t *testing.T, worktreeNames ...string) (string, []string) {
	t.Helper()

	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("Temporary directory resolution failed with error %v.", err)
	}
	repository := filepath.Join(base, "repository")
	if err := os.MkdirAll(repository, 0o755); err != nil {
		t.Fatalf("The test could not create directory %s because of error %v.", repository, err)
	}
	if err := os.WriteFile(filepath.Join(repository, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("The test could not write the fixture file in directory %s because of error %v.", repository, err)
	}
	runSourceTestGit(t, repository, "init", "--quiet", "--initial-branch", "main")
	runSourceTestGit(t, repository, "add", "--all")
	runSourceTestGit(t, repository, "-c", "commit.gpgsign=false", "commit", "--quiet", "--message", "initial")
	worktrees := make([]string, 0, len(worktreeNames))
	for _, name := range worktreeNames {
		worktree := filepath.Join(base, name)
		runSourceTestGit(t, repository, "worktree", "add", "--quiet", "-b", name, worktree)
		worktrees = append(worktrees, worktree)
	}
	return repository, worktrees
}

func newWorktreeSourceScene(t *testing.T) *worktreeSourceScene {
	t.Helper()

	cfg := newSourceTestConfig(t)
	repository, worktrees := newSourceTestRepository(t, "feature-a", "feature-b")
	scene := &worktreeSourceScene{
		config:     cfg,
		manager:    nil,
		server:     nil,
		repository: repository,
		worktreeA:  worktrees[0],
		worktreeB:  worktrees[1],
	}
	scene.start(t)
	t.Cleanup(func() { scene.stop(t) })
	return scene
}

func (scene *worktreeSourceScene) start(t *testing.T) {
	t.Helper()

	manager, err := daemon.NewManager(context.Background(), scene.config)
	if err != nil {
		t.Fatalf("NewManager returned error %v.", err)
	}
	scene.manager = manager
	scene.server = daemon.NewGRPCServer(manager, nil)
}

func (scene *worktreeSourceScene) stop(t *testing.T) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), sourceTestCloseTimeout)
	defer cancel()
	if err := scene.manager.Close(ctx); err != nil {
		t.Errorf("Manager.Close returned error %v.", err)
	}
}

func (scene *worktreeSourceScene) startIndex(path string, sourcePath string, startingCommit string) (*pb.StartIndexResponse, error) {
	return scene.server.StartIndex(context.Background(), &pb.StartIndexRequest{
		Path:           path,
		Client:         &pb.ClientInfo{Name: "source-test"},
		SourcePath:     sourcePath,
		StartingCommit: startingCommit,
	})
}

func (scene *worktreeSourceScene) mustStartIndex(t *testing.T, path string, sourcePath string, startingCommit string) *pb.StartIndexResponse {
	t.Helper()

	response, err := scene.startIndex(path, sourcePath, startingCommit)
	if err != nil {
		t.Fatalf("StartIndex failed for target %s using source %q and starting commit %q with error %v.", path, sourcePath, startingCommit, err)
	}
	return response
}

func (scene *worktreeSourceScene) status(t *testing.T, path string) *pb.GetIndexResponse {
	t.Helper()

	response, err := scene.server.GetIndex(context.Background(), &pb.GetIndexRequest{
		Path:   path,
		Client: &pb.ClientInfo{Name: "source-test"},
	})
	if err != nil {
		t.Fatalf("GetIndex failed for path %s with error %v.", path, err)
	}
	return response
}

func (scene *worktreeSourceScene) requireSource(t *testing.T, path string, wantSourcePath string, wantSourceID string, wantStartingCommit string) {
	t.Helper()

	codebase := scene.status(t, path).GetCodebase()
	if codebase.GetWorktreeSourcePath() != wantSourcePath ||
		codebase.GetWorktreeSourceId() != wantSourceID ||
		codebase.GetWorktreeStartingCommit() != wantStartingCommit {
		t.Fatalf(
			"The stored worktree source for %s has source path %q, source identifier %q, and starting commit %q instead of expected source path %q, source identifier %q, and starting commit %q.",
			path,
			codebase.GetWorktreeSourcePath(),
			codebase.GetWorktreeSourceId(),
			codebase.GetWorktreeStartingCommit(),
			wantSourcePath,
			wantSourceID,
			wantStartingCommit,
		)
	}
}

func (scene *worktreeSourceScene) requireRejected(t *testing.T, path string, sourcePath string, startingCommit string) {
	t.Helper()

	response, err := scene.startIndex(path, sourcePath, startingCommit)
	if err == nil {
		t.Fatalf("StartIndex accepted an invalid worktree source request for %s with source %q, starting commit %q, and response %v.", path, sourcePath, startingCommit, response)
	}
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("StartIndex rejected the request for %s with source %q and starting commit %q with unexpected gRPC code %v instead of the expected %v.", path, sourcePath, startingCommit, status.Code(err), codes.InvalidArgument)
	}
}

func (scene *worktreeSourceScene) requireUnregistered(t *testing.T, path string) {
	t.Helper()

	listed, err := scene.server.ListIndexes(context.Background(), &pb.ListIndexesRequest{})
	if err != nil {
		t.Fatalf("ListIndexes returned error %v.", err)
	}
	for _, codebase := range listed.GetIndexes() {
		if codebase.GetCanonicalPath() == path {
			t.Fatalf("A rejected request registered %s as a codebase.", path)
		}
	}
}

func (scene *worktreeSourceScene) waitForRedirectingStatus(t *testing.T, path string) {
	t.Helper()

	deadline := time.Now().Add(sourceTestStatusTimeout)
	lastStatus := ""
	for time.Now().Before(deadline) {
		lastStatus = scene.status(t, path).GetCodebase().GetStatus()
		if lastStatus == string(model.CodebaseStatusIndexed) {
			return
		}
		time.Sleep(sourceTestPollInterval)
	}
	t.Fatalf("Codebase %s with last status %q did not have status indexed within %s.", path, lastStatus, sourceTestStatusTimeout)
}

func TestStartIndexRecordsWorktreeSourceAcrossRestart(t *testing.T) {
	scene := newWorktreeSourceScene(t)

	scene.mustStartIndex(t, scene.worktreeA, scene.repository, sourceTestStartingCommit)
	scene.requireSource(t, scene.worktreeA, scene.repository, "", sourceTestStartingCommit)

	listed, err := scene.server.ListIndexes(context.Background(), &pb.ListIndexesRequest{})
	if err != nil {
		t.Fatalf("ListIndexes returned error %v.", err)
	}
	if len(listed.GetIndexes()) != 1 || listed.GetIndexes()[0].GetCanonicalPath() != scene.worktreeA {
		t.Fatalf("The registry contains codebases %v instead of only target codebase %s.", listed.GetIndexes(), scene.worktreeA)
	}

	wantLine := "worktree_source_path=" + scene.repository +
		" worktree_source_id= worktree_starting_commit=" + sourceTestStartingCommit
	if displayText := scene.status(t, scene.worktreeA).GetDisplayText(); !strings.Contains(displayText, wantLine) {
		t.Fatalf("The status text %q does not contain the expected worktree source line %q.", displayText, wantLine)
	}

	scene.stop(t)
	scene.start(t)

	scene.requireSource(t, scene.worktreeA, scene.repository, "", sourceTestStartingCommit)
}

func TestStartIndexWithoutSourceFieldsPreservesWorktreeSource(t *testing.T) {
	scene := newWorktreeSourceScene(t)

	scene.mustStartIndex(t, scene.worktreeA, "", "")
	scene.mustStartIndex(t, scene.worktreeA, scene.repository, sourceTestStartingCommit)
	scene.requireSource(t, scene.worktreeA, scene.repository, "", sourceTestStartingCommit)

	scene.mustStartIndex(t, scene.worktreeA, "", "")
	scene.requireSource(t, scene.worktreeA, scene.repository, "", sourceTestStartingCommit)

	scene.stop(t)
	scene.start(t)

	scene.mustStartIndex(t, scene.worktreeA, "", "")
	scene.requireSource(t, scene.worktreeA, scene.repository, "", sourceTestStartingCommit)
}

func TestStartIndexReplacesWorktreeSourceWithRegisteredSource(t *testing.T) {
	scene := newWorktreeSourceScene(t)

	scene.mustStartIndex(t, scene.worktreeA, scene.repository, sourceTestStartingCommit)
	registeredSource := scene.mustStartIndex(t, scene.worktreeB, "", "")
	if registeredSource.GetCodebaseId() == "" {
		t.Fatalf("StartIndex returned no codebase identifier for source worktree %s.", scene.worktreeB)
	}

	scene.mustStartIndex(t, scene.worktreeA, scene.worktreeB, "")
	scene.requireSource(t, scene.worktreeA, scene.worktreeB, registeredSource.GetCodebaseId(), "")
}

func TestStartIndexRejectsInvalidWorktreeSource(t *testing.T) {
	scene := newWorktreeSourceScene(t)
	otherRepository, _ := newSourceTestRepository(t)

	scene.mustStartIndex(t, scene.worktreeA, scene.repository, sourceTestStartingCommit)

	scene.requireRejected(t, scene.worktreeA, scene.worktreeA, "")
	scene.requireRejected(t, scene.worktreeA, otherRepository, "")
	scene.requireRejected(t, scene.worktreeA, filepath.Join(scene.repository, "absent"), "")
	scene.requireSource(t, scene.worktreeA, scene.repository, "", sourceTestStartingCommit)

	scene.requireRejected(t, scene.repository, scene.worktreeA, "")

	scene.requireRejected(t, scene.worktreeB, "", sourceTestStartingCommit)
	scene.requireUnregistered(t, scene.worktreeB)

	scene.mustStartIndex(t, scene.worktreeB, scene.worktreeA, "")
	scene.requireRejected(t, scene.repository, scene.worktreeB, "")
	scene.requireUnregistered(t, scene.repository)
}

func TestStartIndexRejectsWorktreeSourceOnRedirectedPath(t *testing.T) {
	scene := newWorktreeSourceScene(t)
	nestedDirectory := filepath.Join(scene.worktreeA, "nested")
	if err := os.MkdirAll(nestedDirectory, 0o755); err != nil {
		t.Fatalf("The test could not create directory %s because of error %v.", nestedDirectory, err)
	}
	if err := os.WriteFile(filepath.Join(nestedDirectory, "nested.go"), []byte("package nested\n"), 0o644); err != nil {
		t.Fatalf("The test could not write the fixture file in directory %s because of error %v.", nestedDirectory, err)
	}

	scene.mustStartIndex(t, scene.worktreeA, "", "")
	scene.waitForRedirectingStatus(t, scene.worktreeA)

	scene.requireRejected(t, nestedDirectory, scene.repository, "")
	scene.requireRejected(t, nestedDirectory, scene.repository, sourceTestStartingCommit)
	scene.requireUnregistered(t, nestedDirectory)
	scene.requireSource(t, scene.worktreeA, "", "", "")
}
