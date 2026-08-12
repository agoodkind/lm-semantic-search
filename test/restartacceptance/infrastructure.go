//go:build restartacceptance

package restartacceptance

import (
	"archive/tar"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	etcdImage   = "quay.io/coreos/etcd@sha256:b2317748a97a43c82040cdfacb68041b83bb6a01072a7da564a4c2562b3a3bcb"
	minioImage  = "minio/minio@sha256:8ce3a2b6c83521fabbde58b4f06e5fb7f95e64bfc05bc10def264479327c407b"
	milvusImage = "milvusdb/milvus@sha256:fea7ed1d226c93292f476e4283584ce01f0ff01e45762206b9cd7486f9c68c5d"

	minioUserEnvironment     = "LMS_RESTART_MINIO_USER"
	minioPasswordEnvironment = "LMS_RESTART_MINIO_PASSWORD"
)

type runPaths struct {
	RunRoot             string
	Restore             string
	Source              string
	SourceEtcd          string
	SourceMilvus        string
	SourceMinIO         string
	SourceMinIODefault  string
	Cases               string
	LMSState            string
	LMSContext          string
	LMSSocket           string
	LMSLogs             string
	ClydeConfig         string
	ClydeData           string
	ClydeState          string
	ClydeCache          string
	ClydeRuntime        string
	ClydeHome           string
	ClydeConfigFile     string
	Artifacts           string
	EventsJSONL         string
	ResultJSON          string
	ResultMarkdown      string
	ProductionInventory string
	ComposeFile         string
}

func pathsForRun(runRoot string) runPaths {
	restore := filepath.Join(runRoot, "restore")
	source := filepath.Join(restore, "source")
	lms := filepath.Join(runRoot, "lms")
	clyde := filepath.Join(runRoot, "clyde")
	artifacts := filepath.Join(runRoot, "artifacts")
	return runPaths{
		RunRoot:             runRoot,
		Restore:             restore,
		Source:              source,
		SourceEtcd:          filepath.Join(source, "etcd"),
		SourceMilvus:        filepath.Join(source, "milvus"),
		SourceMinIO:         filepath.Join(source, "minio"),
		SourceMinIODefault:  filepath.Join(source, "minio-default"),
		Cases:               filepath.Join(restore, "cases"),
		LMSState:            filepath.Join(lms, "state"),
		LMSContext:          filepath.Join(lms, "context"),
		LMSSocket:           filepath.Join(lms, "daemon.sock"),
		LMSLogs:             filepath.Join(lms, "logs"),
		ClydeConfig:         filepath.Join(clyde, "config"),
		ClydeData:           filepath.Join(clyde, "data"),
		ClydeState:          filepath.Join(clyde, "state"),
		ClydeCache:          filepath.Join(clyde, "cache"),
		ClydeRuntime:        filepath.Join(clyde, "runtime"),
		ClydeHome:           filepath.Join(clyde, "home"),
		ClydeConfigFile:     filepath.Join(clyde, "config", "clyde", "config.toml"),
		Artifacts:           artifacts,
		EventsJSONL:         filepath.Join(artifacts, "events.jsonl"),
		ResultJSON:          filepath.Join(artifacts, "result.json"),
		ResultMarkdown:      filepath.Join(artifacts, "result.md"),
		ProductionInventory: filepath.Join(artifacts, "production-inventory.json"),
		ComposeFile:         filepath.Join(runRoot, "compose.yaml"),
	}
}

func createIsolationLayout(paths runPaths) error {
	directories := []string{
		paths.SourceEtcd,
		paths.SourceMilvus,
		paths.SourceMinIO,
		paths.SourceMinIODefault,
		paths.Cases,
		paths.LMSState,
		paths.LMSContext,
		paths.LMSLogs,
		filepath.Dir(paths.ClydeConfigFile),
		paths.ClydeData,
		paths.ClydeState,
		paths.ClydeCache,
		paths.ClydeRuntime,
		paths.ClydeHome,
		paths.Artifacts,
	}
	for _, directory := range directories {
		if !pathWithin(paths.RunRoot, directory) {
			return fmt.Errorf("layout path %q escapes run root %q", directory, paths.RunRoot)
		}
		if err := os.MkdirAll(directory, 0o700); err != nil {
			return fmt.Errorf("create isolation directory %q: %w", directory, err)
		}
	}
	collectionID := "restart-" + filepath.Base(paths.RunRoot)
	configuration := fmt.Sprintf("[conversation.semantic]\nenabled = true\nsearch_enabled = true\nsocket_path = %q\ncollection_id = %q\n\n[adapter]\nenabled = false\n\n[mitm]\nenabled_default = false\n", paths.LMSSocket, collectionID)
	if err := os.WriteFile(paths.ClydeConfigFile, []byte(configuration), 0o600); err != nil {
		return fmt.Errorf("write isolated Clyde configuration: %w", err)
	}
	return nil
}

func pathWithin(root string, path string) bool {
	relative, err := filepath.Rel(filepath.Clean(root), filepath.Clean(path))
	if err != nil {
		return false
	}
	return relative != "." && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator))
}

func restoreImmutableArchive(ctx context.Context, archivePath string, destination string) error {
	archive, err := openNoFollowRegular(archivePath)
	if err != nil {
		return fmt.Errorf("open restore archive: %w", err)
	}
	defer func() { _ = archive.Close() }()
	return restoreImmutableArchiveReader(ctx, archive, destination)
}

func restoreImmutableArchiveReader(ctx context.Context, archive io.Reader, destination string) error {
	if err := context.Cause(ctx); err != nil {
		return fmt.Errorf("restore archive: %w", err)
	}
	if _, err := os.Lstat(destination); err == nil {
		entries, readErr := os.ReadDir(destination)
		if readErr != nil {
			return fmt.Errorf("inspect restore destination: %w", readErr)
		}
		if len(entries) != 0 {
			return fmt.Errorf("restore destination %q is not empty", destination)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("inspect restore destination: %w", err)
	}
	if err := os.MkdirAll(destination, 0o700); err != nil {
		return fmt.Errorf("create restore destination: %w", err)
	}
	reader := tar.NewReader(&contextReader{ctx: ctx, reader: archive})
	for {
		if err := context.Cause(ctx); err != nil {
			return fmt.Errorf("restore archive: %w", err)
		}
		header, readErr := reader.Next()
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return fmt.Errorf("read restore archive: %w", readErr)
		}
		cleanName := filepath.Clean(header.Name)
		if cleanName == "." {
			if header.Typeflag != tar.TypeDir {
				return fmt.Errorf("restore archive root entry %q is not a directory", header.Name)
			}
			continue
		}
		if filepath.IsAbs(header.Name) || cleanName == ".." || strings.HasPrefix(cleanName, ".."+string(filepath.Separator)) {
			return fmt.Errorf("restore archive entry %q escapes destination", header.Name)
		}
		target := filepath.Join(destination, cleanName)
		if !pathWithin(destination, target) {
			return fmt.Errorf("restore archive entry %q escapes destination", header.Name)
		}
		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o700); err != nil {
				return fmt.Errorf("create restored directory: %w", err)
			}
		case tar.TypeReg, tar.TypeRegA:
			if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
				return fmt.Errorf("create restored parent: %w", err)
			}
			file, openErr := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
			if openErr != nil {
				return fmt.Errorf("create restored file: %w", openErr)
			}
			_, copyErr := io.CopyN(file, &contextReader{ctx: ctx, reader: reader}, header.Size)
			closeErr := file.Close()
			if copyErr != nil {
				return fmt.Errorf("extract restored file: %w", copyErr)
			}
			if closeErr != nil {
				return fmt.Errorf("close restored file: %w", closeErr)
			}
		default:
			return fmt.Errorf("restore archive entry %q has unsupported type %d", header.Name, header.Typeflag)
		}
	}
	if err := filepath.WalkDir(destination, func(path string, entry fs.DirEntry, walkErr error) error {
		if err := context.Cause(ctx); err != nil {
			return err
		}
		if walkErr != nil {
			return walkErr
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("restored source contains symlink %q", path)
		}
		mode := fs.FileMode(0o400)
		if entry.IsDir() {
			mode = 0o500
		}
		return os.Chmod(path, mode)
	}); err != nil {
		return fmt.Errorf("make restored source immutable: %w", err)
	}
	return nil
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (reader *contextReader) Read(body []byte) (int, error) {
	if err := context.Cause(reader.ctx); err != nil {
		return 0, err
	}
	return reader.reader.Read(body)
}

func cloneWritableTree(source string, destination string) error {
	if _, err := os.Lstat(destination); !errors.Is(err, os.ErrNotExist) {
		if err == nil {
			return fmt.Errorf("case destination %q already exists", destination)
		}
		return fmt.Errorf("inspect case destination: %w", err)
	}
	return filepath.WalkDir(source, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		target := filepath.Join(destination, relative)
		if entry.IsDir() {
			return os.MkdirAll(target, 0o700)
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("immutable source contains symlink %q", path)
		}
		if err := cloneFile(path, target); err != nil {
			return fmt.Errorf("clone %q: %w", path, err)
		}
		return os.Chmod(target, 0o600)
	})
}

func removeTree(ctx context.Context, root string) error {
	if err := context.Cause(ctx); err != nil {
		return err
	}
	if err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := context.Cause(ctx); err != nil {
			return err
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return nil
		}
		return os.Chmod(path, 0o700)
	}); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("make cleanup tree writable: %w", err)
	}
	if err := os.RemoveAll(root); err != nil {
		return fmt.Errorf("remove cleanup tree: %w", err)
	}
	return nil
}

func renderCompose(paths runPaths, caseName string) string {
	caseRoot := filepath.Join(paths.Cases, caseName)
	return fmt.Sprintf(`services:
  etcd:
    image: %s
    command: etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls=http://0.0.0.0:2379 --data-dir=/etcd
    ports:
      - "127.0.0.1:%d:2379"
    volumes:
      - "%s/etcd:/etcd"
    healthcheck:
      test: ["CMD-SHELL", "etcdctl endpoint health"]
      interval: 5s
      timeout: 5s
      retries: 12
  minio:
    image: %s
    command: minio server /minio_data --console-address :9001
    environment:
      MINIO_ROOT_USER: ${%s:?required}
      MINIO_ROOT_PASSWORD: ${%s:?required}
    ports:
      - "127.0.0.1:%d:9000"
      - "127.0.0.1:%d:9001"
    volumes:
      - "%s/minio:/minio_data"
      - "%s/minio-default:/data"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 5s
      retries: 12
  standalone:
    image: %s
    command: ["milvus", "run", "standalone"]
    security_opt:
      - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY_ID: ${%s:?required}
      MINIO_SECRET_ACCESS_KEY: ${%s:?required}
    ports:
      - "127.0.0.1:%d:19530"
      - "127.0.0.1:%d:9091"
    volumes:
      - "%s/milvus:/var/lib/milvus"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 5s
      timeout: 5s
      retries: 30
    depends_on:
      etcd:
        condition: service_healthy
      minio:
        condition: service_healthy
`, etcdImage, etcdClientPort, caseRoot, minioImage, minioUserEnvironment,
		minioPasswordEnvironment, minioAPIPort, minioConsolePort, caseRoot, caseRoot,
		milvusImage, minioUserEnvironment, minioPasswordEnvironment, milvusGRPCPort,
		milvusHealthPort, caseRoot)
}

type evidenceEvent struct {
	RecordedAt time.Time         `json:"recorded_at"`
	Phase      string            `json:"phase"`
	Status     string            `json:"status"`
	Details    map[string]string `json:"details,omitempty"`
}

type acceptanceResult struct {
	RunID  string `json:"run_id"`
	Status string `json:"status"`
	Error  string `json:"error,omitempty"`
}

type evidenceRecorder struct {
	paths runPaths
	now   func() time.Time
}

func newEvidenceRecorder(paths runPaths, now func() time.Time) *evidenceRecorder {
	return &evidenceRecorder{paths: paths, now: now}
}

func (recorder *evidenceRecorder) Record(phase string, status string, details map[string]string) error {
	event := evidenceEvent{RecordedAt: recorder.now().UTC(), Phase: phase, Status: status, Details: details}
	body, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("encode evidence event: %w", err)
	}
	file, err := os.OpenFile(recorder.paths.EventsJSONL, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return fmt.Errorf("open evidence events: %w", err)
	}
	if _, err := file.Write(append(body, '\n')); err != nil {
		_ = file.Close()
		return fmt.Errorf("write evidence event: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close evidence events: %w", err)
	}
	return nil
}

func (recorder *evidenceRecorder) Finish(result acceptanceResult) error {
	body, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("encode acceptance result: %w", err)
	}
	body = append(body, '\n')
	if err := os.WriteFile(recorder.paths.ResultJSON, body, 0o600); err != nil {
		return fmt.Errorf("write JSON acceptance result: %w", err)
	}
	markdown := fmt.Sprintf("# Restart acceptance\n\nRun: %s\n\nStatus: %s\n", result.RunID, result.Status)
	if result.Error != "" {
		markdown += "\nError: " + result.Error + "\n"
	}
	if err := os.WriteFile(recorder.paths.ResultMarkdown, []byte(markdown), 0o600); err != nil {
		return fmt.Errorf("write Markdown acceptance result: %w", err)
	}
	return nil
}

type commandRunner interface {
	Run(context.Context, map[string]string, string, ...string) ([]byte, error)
}

type harness struct {
	paths                runPaths
	composeProject       string
	runner               commandRunner
	valueEntropy         io.Reader
	valueMutex           sync.Mutex
	runtimeValues        composeRuntimeValues
	archiveSizes         []int64
	availableBytes       func(string) (int64, error)
	inventory            inventoryToken
	now                  func() time.Time
	census               collectionCensusFunc
	proxyCalls           func() []milvusProxyCall
	readiness            func(context.Context) error
	protectedCollections map[collectionIdentity]struct{}
}

type composeRuntimeValues struct {
	userValue string
	keyValue  string
}

func newComposeRuntimeValues(entropy io.Reader) (composeRuntimeValues, error) {
	var randomBytes [32]byte
	if _, err := io.ReadFull(entropy, randomBytes[:]); err != nil {
		return composeRuntimeValues{}, fmt.Errorf("generate acceptance runtime values: %w", err)
	}
	return composeRuntimeValues{
		userValue: hex.EncodeToString(randomBytes[:8]),
		keyValue:  hex.EncodeToString(randomBytes[8:]),
	}, nil
}

func (h *harness) setRuntimeValues(values composeRuntimeValues) {
	h.runtimeValues = values
}

func (h *harness) composeEnvironment() (map[string]string, error) {
	h.valueMutex.Lock()
	defer h.valueMutex.Unlock()
	if h.runtimeValues.userValue == "" || h.runtimeValues.keyValue == "" {
		entropy := h.valueEntropy
		if entropy == nil {
			entropy = rand.Reader
		}
		values, err := newComposeRuntimeValues(entropy)
		if err != nil {
			return nil, err
		}
		h.setRuntimeValues(values)
	}
	return map[string]string{
		minioUserEnvironment:     h.runtimeValues.userValue,
		minioPasswordEnvironment: h.runtimeValues.keyValue,
	}, nil
}

func (h *harness) runCompose(ctx context.Context, caseName string) (runErr error) {
	return h.withCompose(ctx, caseName, func(context.Context) error { return nil })
}

func (h *harness) withCompose(
	ctx context.Context,
	caseName string,
	scenario func(context.Context) error,
) (runErr error) {
	if !validComposeProject(h.composeProject, filepath.Base(h.paths.RunRoot)) {
		return fmt.Errorf("compose project %q is not tagged for run %q", h.composeProject, filepath.Base(h.paths.RunRoot))
	}
	now := time.Now().UTC()
	if h.now != nil {
		now = h.now().UTC()
	}
	if err := validateInventoryToken(h.inventory, filepath.Base(h.paths.RunRoot), now); err != nil {
		return err
	}
	if h.census == nil {
		return fmt.Errorf("production collection census is unavailable for post-run audit")
	}
	if h.readiness == nil {
		return fmt.Errorf("production readiness check is unavailable")
	}
	composeEnvironment, err := h.composeEnvironment()
	if err != nil {
		return err
	}
	caseRoot, err := resolveCaseRoot(h.paths.Cases, caseName)
	if err != nil {
		return err
	}
	spaceReader := h.availableBytes
	if spaceReader == nil {
		spaceReader = availableDiskBytes
	}
	available, err := spaceReader(h.paths.RunRoot)
	if err != nil {
		return err
	}
	if err := validateFreeSpace(available, h.archiveSizes); err != nil {
		return fmt.Errorf("recheck case free space: %w", err)
	}
	defer func() {
		cleanupContext, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		if removeErr := removeTree(cleanupContext, caseRoot); removeErr != nil {
			runErr = errors.Join(runErr, fmt.Errorf("clean case tree: %w", removeErr))
		}
	}()
	for _, name := range []string{"etcd", "milvus", "minio", "minio-default"} {
		if err := cloneWritableTree(filepath.Join(h.paths.Source, name), filepath.Join(caseRoot, name)); err != nil {
			return fmt.Errorf("create writable %s copy: %w", name, err)
		}
	}
	if err := os.WriteFile(h.paths.ComposeFile, []byte(renderCompose(h.paths, caseName)), 0o600); err != nil {
		return fmt.Errorf("write compose file: %w", err)
	}
	defer func() {
		cleanupContext, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		_, cleanupErr := h.runner.Run(cleanupContext, composeEnvironment, "docker", "compose", "-p", h.composeProject, "-f", h.paths.ComposeFile, "down", "--volumes", "--remove-orphans")
		if cleanupErr != nil {
			runErr = errors.Join(runErr, fmt.Errorf("clean compose project %q: %w", h.composeProject, cleanupErr))
		}
		milvusSnapshot, censusErr := h.census(cleanupContext)
		calls := []milvusProxyCall(nil)
		if h.proxyCalls != nil {
			calls = h.proxyCalls()
		}
		if censusErr == nil {
			censusErr = auditProductionMutation(h.inventory.Inventory, productionInventory{
				Databases:   slices.Clone(milvusSnapshot.Databases),
				Collections: cloneCollectionCensus(milvusSnapshot.Collections),
				Samples:     cloneCollectionCensus(milvusSnapshot.Samples),
			}, calls, h.protectedCollections)
		}
		if censusErr != nil {
			runErr = errors.Join(runErr, fmt.Errorf("audit production after case: %w", censusErr))
		}
	}()
	if err := h.readiness(ctx); err != nil {
		return fmt.Errorf("recheck production readiness before clone startup: %w", err)
	}
	if _, err := h.runner.Run(ctx, composeEnvironment, "docker", "compose", "-p", h.composeProject, "-f", h.paths.ComposeFile, "up", "-d", "--wait"); err != nil {
		return fmt.Errorf("start compose project %q: %w", h.composeProject, err)
	}
	if err := scenario(ctx); err != nil {
		return fmt.Errorf("run restart acceptance scenario: %w", err)
	}
	return nil
}

var caseNamePattern = regexp.MustCompile(`^[a-h](-[a-z0-9]+)?$`)

func resolveCaseRoot(casesRoot string, caseName string) (string, error) {
	if !caseNamePattern.MatchString(caseName) {
		return "", fmt.Errorf("case name %q is invalid", caseName)
	}
	resolvedCases, err := filepath.EvalSymlinks(casesRoot)
	if err != nil {
		return "", fmt.Errorf("resolve cases root: %w", err)
	}
	caseRoot := filepath.Join(resolvedCases, caseName)
	if !pathWithin(resolvedCases, caseRoot) {
		return "", fmt.Errorf("case root %q escapes cases root", caseRoot)
	}
	if _, err := os.Lstat(caseRoot); !errors.Is(err, os.ErrNotExist) {
		if err == nil {
			return "", fmt.Errorf("case root %q already exists", caseRoot)
		}
		return "", fmt.Errorf("inspect case root: %w", err)
	}
	return caseRoot, nil
}

func validComposeProject(project string, runID string) bool {
	if !runIDPattern.MatchString(runID) {
		return false
	}
	suffix := runID[len(runID)-8:]
	return project == "lms-restart-"+suffix
}

func isolatedLMSEnvironment(paths runPaths) map[string]string {
	return map[string]string{
		"XDG_STATE_HOME":               paths.LMSState,
		"CLAUDE_CONTEXTD_CONTEXT_ROOT": paths.LMSContext,
		"CLAUDE_CONTEXTD_SOCKET_PATH":  paths.LMSSocket,
		"MILVUS_ADDRESS":               fmt.Sprintf("127.0.0.1:%d", milvusProxyPort),
		"MILVUS_DATABASE":              cloneMilvusDatabase,
		"OPENAI_BASE_URL":              fmt.Sprintf("http://127.0.0.1:%d", embeddingProxyPort),
	}
}

func isolatedClydeEnvironment(paths runPaths) map[string]string {
	return map[string]string{
		"HOME":                        paths.ClydeHome,
		"XDG_CONFIG_HOME":             paths.ClydeConfig,
		"XDG_DATA_HOME":               paths.ClydeData,
		"XDG_STATE_HOME":              paths.ClydeState,
		"XDG_CACHE_HOME":              paths.ClydeCache,
		"XDG_RUNTIME_DIR":             paths.ClydeRuntime,
		"CLAUDE_CONTEXTD_SOCKET_PATH": paths.LMSSocket,
	}
}

func captureProductionInventory(
	ctx context.Context,
	paths runPaths,
	binaries installedBinaries,
	runner commandRunner,
	runID string,
	capturedAt time.Time,
	census collectionCensusFunc,
) (inventoryToken, error) {
	if census == nil {
		return inventoryToken{}, fmt.Errorf("production collection census is unavailable")
	}
	commands := [][]string{
		{"--json", "codebase", "list"},
		{"--json", "job", "list"},
		{"--json", "daemon", "status"},
	}
	outputs := make([]json.RawMessage, len(commands))
	for index, arguments := range commands {
		body, err := runner.Run(ctx, nil, binaries.CLI, arguments...)
		if err != nil {
			return inventoryToken{}, fmt.Errorf("capture production inventory command %q: %w", strings.Join(arguments, " "), err)
		}
		if !json.Valid(body) {
			return inventoryToken{}, fmt.Errorf("production inventory command %q returned invalid JSON", strings.Join(arguments, " "))
		}
		outputs[index] = append(json.RawMessage(nil), body...)
	}
	if err := validateProductionReadiness(outputs[0], outputs[1]); err != nil {
		return inventoryToken{}, err
	}
	milvusSnapshot, err := census(ctx)
	if err != nil {
		return inventoryToken{}, fmt.Errorf("capture production collection census: %w", err)
	}
	if len(milvusSnapshot.Databases) == 0 || len(milvusSnapshot.Collections) == 0 {
		return inventoryToken{}, fmt.Errorf("production collection census is empty")
	}
	inventory := productionInventory{
		Databases:   slices.Clone(milvusSnapshot.Databases),
		Collections: cloneCollectionCensus(milvusSnapshot.Collections),
		Samples:     cloneCollectionCensus(milvusSnapshot.Samples),
		Codebases:   outputs[0],
		Jobs:        outputs[1],
		Daemon:      outputs[2],
	}
	token, err := newInventoryToken(runID, capturedAt, inventory)
	if err != nil {
		return inventoryToken{}, err
	}
	body, err := json.MarshalIndent(token, "", "  ")
	if err != nil {
		return inventoryToken{}, fmt.Errorf("encode production inventory: %w", err)
	}
	if err := os.WriteFile(paths.ProductionInventory, append(body, '\n'), 0o600); err != nil {
		return inventoryToken{}, fmt.Errorf("write production inventory: %w", err)
	}
	return token, nil
}

func captureProductionReadiness(
	ctx context.Context,
	binaries installedBinaries,
	runner commandRunner,
) error {
	commands := [][]string{{"--json", "codebase", "list"}, {"--json", "job", "list"}}
	outputs := make([]json.RawMessage, len(commands))
	for index, arguments := range commands {
		body, err := runner.Run(ctx, nil, binaries.CLI, arguments...)
		if err != nil {
			return fmt.Errorf("recheck production command %q: %w", strings.Join(arguments, " "), err)
		}
		if !json.Valid(body) {
			return fmt.Errorf("production command %q returned invalid JSON", strings.Join(arguments, " "))
		}
		outputs[index] = append(json.RawMessage(nil), body...)
	}
	return validateProductionReadiness(outputs[0], outputs[1])
}

type collectionCensusFunc func(context.Context) (productionMilvusCensus, error)

type productionMilvusCensus struct {
	Databases   []string         `json:"databases"`
	Collections collectionCensus `json:"collections"`
	Samples     collectionCensus `json:"samples,omitempty"`
}

type collectionIdentity struct {
	Database   string `json:"database"`
	Collection string `json:"collection"`
}

type collectionCensus map[collectionIdentity]string

type collectionCensusEntry struct {
	Database   string `json:"database"`
	Collection string `json:"collection"`
	Hash       string `json:"hash"`
}

func (census collectionCensus) MarshalJSON() ([]byte, error) {
	entries := make([]collectionCensusEntry, 0, len(census))
	for identity, hash := range census {
		entries = append(entries, collectionCensusEntry{
			Database: identity.Database, Collection: identity.Collection, Hash: hash,
		})
	}
	sort.Slice(entries, func(left int, right int) bool {
		if entries[left].Database != entries[right].Database {
			return entries[left].Database < entries[right].Database
		}
		return entries[left].Collection < entries[right].Collection
	})
	return json.Marshal(entries)
}

type inventoryToken struct {
	RunID       string              `json:"run_id"`
	CapturedAt  time.Time           `json:"captured_at"`
	ContentHash string              `json:"content_hash"`
	Inventory   productionInventory `json:"inventory"`
}

func newInventoryToken(runID string, capturedAt time.Time, inventory productionInventory) (inventoryToken, error) {
	body, err := json.Marshal(inventory)
	if err != nil {
		return inventoryToken{}, fmt.Errorf("hash production inventory: %w", err)
	}
	digest := sha256.Sum256(body)
	return inventoryToken{
		RunID:       runID,
		CapturedAt:  capturedAt.UTC(),
		ContentHash: hex.EncodeToString(digest[:]),
		Inventory:   inventory,
	}, nil
}

func validateInventoryToken(token inventoryToken, runID string, now time.Time) error {
	if token.RunID != runID {
		return fmt.Errorf("production inventory token belongs to run %q, not %q", token.RunID, runID)
	}
	if token.CapturedAt.IsZero() || token.CapturedAt.After(now) || now.Sub(token.CapturedAt) > 5*time.Minute {
		return fmt.Errorf("production inventory token timestamp is invalid or stale")
	}
	if token.ContentHash == "" || len(token.Inventory.Databases) == 0 || len(token.Inventory.Collections) == 0 {
		return fmt.Errorf("production inventory token content is empty")
	}
	expected, err := newInventoryToken(token.RunID, token.CapturedAt, token.Inventory)
	if err != nil {
		return err
	}
	if token.ContentHash != expected.ContentHash {
		return fmt.Errorf("production inventory token content hash is invalid")
	}
	return nil
}

func cloneCollectionCensus(collections collectionCensus) collectionCensus {
	result := make(collectionCensus, len(collections))
	for identity, hash := range collections {
		result[identity] = hash
	}
	return result
}

type dependencyHealthJSON struct {
	Degraded bool   `json:"degraded"`
	Mode     string `json:"mode"`
}

type indexesInventoryJSON struct {
	DependencyHealth *dependencyHealthJSON `json:"dependencyHealth"`
}

type jobsInventoryJSON struct {
	Jobs []struct {
		ID    string `json:"id"`
		State string `json:"state"`
	} `json:"jobs"`
	DependencyHealth *dependencyHealthJSON `json:"dependencyHealth"`
}

func validateProductionReadiness(indexesBody json.RawMessage, jobsBody json.RawMessage) error {
	var indexes indexesInventoryJSON
	if err := json.Unmarshal(indexesBody, &indexes); err != nil {
		return fmt.Errorf("decode production codebase inventory: %w", err)
	}
	if indexes.DependencyHealth == nil {
		return fmt.Errorf("production codebase inventory is missing dependency health")
	}
	if indexes.DependencyHealth.Degraded || indexes.DependencyHealth.Mode != "" {
		return fmt.Errorf("production dependencies are degraded: %s", indexes.DependencyHealth.Mode)
	}
	var jobs jobsInventoryJSON
	if err := json.Unmarshal(jobsBody, &jobs); err != nil {
		return fmt.Errorf("decode production job inventory: %w", err)
	}
	if jobs.DependencyHealth == nil {
		return fmt.Errorf("production job inventory is missing dependency health")
	}
	if jobs.DependencyHealth.Degraded || jobs.DependencyHealth.Mode != "" {
		return fmt.Errorf("production dependencies are degraded: %s", jobs.DependencyHealth.Mode)
	}
	for _, job := range jobs.Jobs {
		switch job.State {
		case "completed", "failed", "cancelled":
		case "queued", "running", "cancelling":
			return fmt.Errorf("production job %q is active in state %q", job.ID, job.State)
		default:
			return fmt.Errorf("production job %q has unknown state %q", job.ID, job.State)
		}
	}
	return nil
}

type productionInventory struct {
	Databases   []string         `json:"databases"`
	Collections collectionCensus `json:"collections"`
	Samples     collectionCensus `json:"samples,omitempty"`
	Codebases   json.RawMessage  `json:"codebases,omitempty"`
	Jobs        json.RawMessage  `json:"jobs,omitempty"`
	Daemon      json.RawMessage  `json:"daemon,omitempty"`
}

type milvusProxyCall struct {
	Database   string `json:"database"`
	Collection string `json:"collection"`
	Method     string `json:"method"`
}

func auditProductionMutation(
	before productionInventory,
	after productionInventory,
	calls []milvusProxyCall,
	protectedCollections map[collectionIdentity]struct{},
) error {
	beforeDatabases := slices.Clone(before.Databases)
	afterDatabases := slices.Clone(after.Databases)
	slices.Sort(beforeDatabases)
	slices.Sort(afterDatabases)
	if !slices.Equal(beforeDatabases, afterDatabases) {
		return fmt.Errorf("production database set changed from %v to %v", beforeDatabases, afterDatabases)
	}
	for identity, beforeHash := range before.Collections {
		afterHash, exists := after.Collections[identity]
		if !exists {
			return fmt.Errorf("production collection %q/%q was removed", identity.Database, identity.Collection)
		}
		if afterHash != beforeHash {
			return fmt.Errorf("production collection %q/%q changed", identity.Database, identity.Collection)
		}
		beforeSample, sampledBefore := before.Samples[identity]
		afterSample, sampledAfter := after.Samples[identity]
		if sampledBefore && sampledAfter && beforeSample != afterSample {
			return fmt.Errorf("production collection %q/%q sampled rows changed", identity.Database, identity.Collection)
		}
	}
	for identity := range after.Collections {
		if _, exists := before.Collections[identity]; exists {
			continue
		}
		if _, protected := protectedCollections[identity]; protected {
			return fmt.Errorf("acceptance collection %q/%q appeared in production", identity.Database, identity.Collection)
		}
		for _, call := range calls {
			if call.Database == identity.Database && call.Collection == identity.Collection {
				return fmt.Errorf("production collection %q/%q appeared with harness mutation %s", identity.Database, identity.Collection, call.Method)
			}
		}
	}
	return nil
}
