package main_test

import (
	"bytes"
	"context"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/encoding/protojson"
)

const (
	indexSourceWorktreeBranch = "feature"
	indexSourceCloseTimeout   = 10 * time.Second
	indexSourceSyncIntervalMS = 300000
)

func newIndexSourceDaemonConfig(t *testing.T) config.Config {
	t.Helper()

	stateRoot := t.TempDir()
	socketDirectory, err := os.MkdirTemp("/tmp", "lms-cli-source-")
	if err != nil {
		t.Fatalf("Socket directory creation under /tmp failed with error %v.", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(socketDirectory) })
	daemonConfig := config.Config{
		StateRoot:         stateRoot,
		SocketPath:        filepath.Join(socketDirectory, "daemon.sock"),
		RegistryPath:      filepath.Join(stateRoot, "registry.json"),
		JobsPath:          filepath.Join(stateRoot, "jobs.jsonl"),
		EventsPath:        filepath.Join(stateRoot, "events.jsonl"),
		LogsDir:           filepath.Join(stateRoot, "logs"),
		LogPath:           filepath.Join(stateRoot, "logs", "lm-semantic-search-daemon.log"),
		MerkleDir:         filepath.Join(stateRoot, "merkle"),
		LocksDir:          filepath.Join(stateRoot, "locks"),
		SocketsDir:        socketDirectory,
		ChunksDir:         filepath.Join(stateRoot, "chunks"),
		GraphDir:          filepath.Join(stateRoot, "graph"),
		ContextRoot:       filepath.Join(stateRoot, "context"),
		EmbeddingProvider: "OpenAI",
		EmbeddingModel:    "nvidia/NV-EmbedCode-7b-v1",
		HybridMode:        true,
		SyncIntervalMS:    indexSourceSyncIntervalMS,
	}
	stateDirectories := []string{
		daemonConfig.StateRoot,
		daemonConfig.LogsDir,
		daemonConfig.MerkleDir,
		daemonConfig.LocksDir,
		daemonConfig.ChunksDir,
		daemonConfig.GraphDir,
		daemonConfig.ContextRoot,
	}
	for _, directory := range stateDirectories {
		if err := store.EnsureDir(directory); err != nil {
			t.Fatalf("Daemon state directory creation at %s failed with error %v.", directory, err)
		}
	}
	if err := store.WriteRegistry(daemonConfig.RegistryPath, model.RegistryFile{}); err != nil {
		t.Fatalf("The empty registry write at %s failed with error %v.", daemonConfig.RegistryPath, err)
	}
	return daemonConfig
}

func startIndexSourceDaemon(t *testing.T) string {
	t.Helper()

	daemonConfig := newIndexSourceDaemonConfig(t)
	manager, err := daemon.NewManager(context.Background(), daemonConfig)
	if err != nil {
		t.Fatalf("daemon.NewManager returned error %v.", err)
	}
	listener, err := (&net.ListenConfig{}).Listen(context.Background(), "unix", daemonConfig.SocketPath)
	if err != nil {
		t.Fatalf("The daemon socket listener at %s failed to start with error %v.", daemonConfig.SocketPath, err)
	}
	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(grpcutil.MaxMessageBytes),
		grpc.MaxSendMsgSize(grpcutil.MaxMessageBytes),
	)
	pb.RegisterSemanticSearchDaemonServiceServer(server, daemon.NewGRPCServer(manager, nil))
	go func() {
		_ = server.Serve(listener)
	}()
	t.Cleanup(func() {
		server.GracefulStop()
		closeContext, cancel := context.WithTimeout(context.Background(), indexSourceCloseTimeout)
		defer cancel()
		if err := manager.Close(closeContext); err != nil {
			t.Errorf("Manager.Close returned error %v.", err)
		}
	})
	return daemonConfig.SocketPath
}

func runIndexSourceGit(t *testing.T, directory string, arguments ...string) string {
	t.Helper()

	command := exec.CommandContext(
		context.Background(),
		"git",
		append([]string{"-C", directory}, arguments...)...,
	)
	command.Env = append(
		os.Environ(),
		"GIT_AUTHOR_NAME=cli-source-test",
		"GIT_AUTHOR_EMAIL=cli-source-test@example.invalid",
		"GIT_COMMITTER_NAME=cli-source-test",
		"GIT_COMMITTER_EMAIL=cli-source-test@example.invalid",
	)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("The git command with arguments %s in directory %s failed with error %v and output %s.", strings.Join(arguments, " "), directory, err, output)
	}
	return strings.TrimSpace(string(output))
}

func newIndexSourceRepository(t *testing.T) (string, string) {
	t.Helper()

	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("Temporary directory resolution failed with error %v.", err)
	}
	repository := filepath.Join(base, "repository")
	if err := os.MkdirAll(repository, 0o755); err != nil {
		t.Fatalf("Repository directory creation at %s failed with error %v.", repository, err)
	}
	if err := os.WriteFile(filepath.Join(repository, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("The fixture file write in directory %s failed with error %v.", repository, err)
	}
	runIndexSourceGit(t, repository, "init", "--quiet", "--initial-branch", "main")
	runIndexSourceGit(t, repository, "add", "--all")
	runIndexSourceGit(t, repository, "-c", "commit.gpgsign=false", "commit", "--quiet", "--message", "initial")
	worktree := filepath.Join(base, indexSourceWorktreeBranch)
	runIndexSourceGit(t, repository, "worktree", "add", "--quiet", "-b", indexSourceWorktreeBranch, worktree)
	return repository, worktree
}

func runIndexSourceCLI(t *testing.T, binaryPath string, socketPath string, arguments ...string) []byte {
	t.Helper()

	command := exec.CommandContext(
		context.Background(),
		binaryPath,
		append([]string{"--socket", socketPath}, arguments...)...,
	)
	command.Env = os.Environ()
	var standardError bytes.Buffer
	command.Stderr = &standardError
	output, err := command.Output()
	if err != nil {
		t.Fatalf("The lm-semantic-search command with arguments %s failed with error %v, standard output %s, and standard error %s.", strings.Join(arguments, " "), err, output, standardError.String())
	}
	return output
}

func TestCodebaseIndexRecordsSourcePathAndStartingCommit(t *testing.T) {
	binaryPath := buildProfileCLI(t)
	repository, worktree := newIndexSourceRepository(t)
	startingCommit := runIndexSourceGit(t, repository, "rev-parse", "HEAD")
	isolatedProfileConfigPath(t)
	socketPath := startIndexSourceDaemon(t)

	runIndexSourceCLI(
		t,
		binaryPath,
		socketPath,
		"codebase", "index", worktree,
		"--source-path", repository,
		"--starting-commit", startingCommit,
	)

	statusOutput := runIndexSourceCLI(t, binaryPath, socketPath, "--json", "codebase", "status", worktree)
	status := &pb.GetIndexResponse{}
	if err := protojson.Unmarshal(statusOutput, status); err != nil {
		t.Fatalf("The parser failed to parse the codebase status output as a GetIndexResponse with error %v. The output was %s.", err, statusOutput)
	}
	codebase := status.GetCodebase()
	if codebase.GetCanonicalPath() != worktree {
		t.Fatalf("The codebase status command reports canonical path %q instead of the worktree %q in output %s.", codebase.GetCanonicalPath(), worktree, statusOutput)
	}
	if codebase.GetWorktreeSourcePath() != repository {
		t.Fatalf("The codebase records source path %q instead of the --source-path value %q.", codebase.GetWorktreeSourcePath(), repository)
	}
	if codebase.GetWorktreeStartingCommit() != startingCommit {
		t.Fatalf("The codebase records starting commit %q instead of the --starting-commit value %q.", codebase.GetWorktreeStartingCommit(), startingCommit)
	}
}
