//go:build restartacceptance

package restartacceptance

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestComposeFilePinsImagesPortsAndWritableCaseData(t *testing.T) {
	t.Parallel()

	paths := pathsForRun(filepath.Join(t.TempDir(), "20260812T010203Z-abcdef01"))
	content := renderCompose(paths, "g-restore")
	for _, literal := range []string{
		etcdImage.Tag,
		minioImage.Tag,
		milvusImage.Tag,
		"pull_policy: never",
		"127.0.0.1:22379:2379",
		"127.0.0.1:29000:9000",
		"127.0.0.1:29001:9001",
		"127.0.0.1:29530:19530",
		"127.0.0.1:29091:9091",
		paths.Cases + "/g-restore/etcd:/etcd",
		paths.Cases + "/g-restore/milvus:/var/lib/milvus",
		paths.Cases + "/g-restore/minio:/minio_data",
		paths.Cases + "/g-restore/minio-default:/data",
		`test: ["CMD", "etcdctl", "endpoint", "health"]`,
		"http://localhost:9000/minio/health/live",
		"http://localhost:9091/healthz",
		"condition: service_healthy",
	} {
		if !strings.Contains(content, literal) {
			t.Fatalf("compose file missing %q:\n%s", literal, content)
		}
	}
	for _, image := range []recordedImage{etcdImage, minioImage, milvusImage} {
		if strings.Contains(content, image.ID) {
			t.Fatalf("compose file uses local image ID as a registry digest: %s", image.ID)
		}
	}
	if strings.Contains(content, "restart: unless-stopped") {
		t.Fatal("compose file inherited production restart policy")
	}
	if strings.Contains(content, `test: ["CMD-SHELL", "etcdctl endpoint health"]`) {
		t.Fatal("etcd health check requires a shell that the recorded image does not contain")
	}
	for _, line := range strings.Split(content, "\n") {
		isCredential := strings.Contains(line, "ACCESS_KEY") ||
			strings.Contains(line, "SECRET") ||
			strings.Contains(line, "ROOT_USER")
		if isCredential && !strings.Contains(line, "${") {
			t.Fatal("compose file contains a credential literal")
		}
	}
}

func TestValidateRecordedImagesRequiresExactLocalImageIDs(t *testing.T) {
	runner := &recordingRunner{outputs: [][]byte{
		[]byte(etcdImage.ID + "\n"),
		[]byte(minioImage.ID + "\n"),
		[]byte(milvusImage.ID + "\n"),
	}}
	if err := validateRecordedImages(context.Background(), runner); err != nil {
		t.Fatalf("validate recorded images: %v", err)
	}
	if len(runner.calls) != 3 {
		t.Fatalf("image inspection calls = %d, want 3", len(runner.calls))
	}
	for index, image := range []recordedImage{etcdImage, minioImage, milvusImage} {
		want := []string{"docker", "image", "inspect", "--format={{.Id}}", image.Tag}
		if !slices.Equal(runner.calls[index], want) {
			t.Fatalf("image inspection call %d = %v, want %v", index, runner.calls[index], want)
		}
	}

	mismatch := &recordingRunner{outputs: [][]byte{[]byte("sha256:wrong\n")}}
	if err := validateRecordedImages(context.Background(), mismatch); err == nil {
		t.Fatal("mismatched local image ID was accepted")
	}
}

func TestRemoveTreeDoesNotFollowRuntimeSymlinks(t *testing.T) {
	root := t.TempDir()
	external := filepath.Join(t.TempDir(), "external")
	if err := os.WriteFile(external, []byte("preserved"), 0o400); err != nil {
		t.Fatalf("write external file: %v", err)
	}
	link := filepath.Join(root, "runtime-link")
	if err := os.Symlink(external, link); err != nil {
		t.Fatalf("create runtime symlink: %v", err)
	}

	if err := removeTree(context.Background(), root); err != nil {
		t.Fatalf("remove tree: %v", err)
	}
	info, err := os.Stat(external)
	if err != nil {
		t.Fatalf("external file was removed: %v", err)
	}
	if info.Mode().Perm() != 0o400 {
		t.Fatalf("external mode = %o, want 400", info.Mode().Perm())
	}
}

func TestRemoveTreeRetriesDirectoryNotEmpty(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, ".DS_Store"), []byte("finder"), 0o600); err != nil {
		t.Fatalf("write Finder metadata: %v", err)
	}
	attempts := 0
	removeAll := func(path string) error {
		attempts++
		if attempts == 1 {
			return &os.PathError{Op: "unlinkat", Path: path, Err: syscall.ENOTEMPTY}
		}
		return os.RemoveAll(path)
	}

	if err := removeTreeWith(context.Background(), root, removeAll, 0, time.Second); err != nil {
		t.Fatalf("remove tree after transient directory recreation: %v", err)
	}
	if attempts != 2 {
		t.Fatalf("remove attempts = %d, want 2", attempts)
	}
	if _, err := os.Stat(root); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("cleanup root still exists: %v", err)
	}
}

func TestRemoveTreeStopsAfterCleanupDeadline(t *testing.T) {
	root := t.TempDir()
	attempts := 0
	removeAll := func(path string) error {
		attempts++
		return &os.PathError{Op: "unlinkat", Path: path, Err: syscall.ENOTEMPTY}
	}

	err := removeTreeWith(context.Background(), root, removeAll, time.Millisecond, 10*time.Millisecond)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("remove tree error = %v, want deadline exceeded", err)
	}
	if attempts < 1 {
		t.Fatal("remove tree did not attempt removal")
	}
}

func TestRemoveTreeChecksCancellationBeforeRetry(t *testing.T) {
	root := t.TempDir()
	ctx, cancel := context.WithCancel(context.Background())
	attempts := 0
	removeAll := func(path string) error {
		attempts++
		cancel()
		return &os.PathError{Op: "unlinkat", Path: path, Err: syscall.ENOTEMPTY}
	}

	err := removeTreeWith(ctx, root, removeAll, 0, time.Second)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("remove tree error = %v, want context canceled", err)
	}
	if attempts != 1 {
		t.Fatalf("remove attempts = %d, want 1", attempts)
	}
}

func TestRemoveTreeStopsAfterNonRetryableError(t *testing.T) {
	root := t.TempDir()
	attempts := 0
	wantErr := syscall.EACCES
	removeAll := func(path string) error {
		attempts++
		return &os.PathError{Op: "unlinkat", Path: path, Err: wantErr}
	}

	err := removeTreeWith(context.Background(), root, removeAll, 0, time.Second)
	if !errors.Is(err, wantErr) {
		t.Fatalf("remove tree error = %v, want permission denied", err)
	}
	if attempts != 1 {
		t.Fatalf("remove attempts = %d, want 1", attempts)
	}
}

func TestHarnessPassesGeneratedCredentialsOnlyThroughComposeEnvironment(t *testing.T) {
	paths := pathsForRun(filepath.Join(t.TempDir(), "lms-restart-acceptance", "20260812T010203Z-abcdef01"))
	for _, path := range []string{paths.SourceEtcd, paths.SourceMilvus, paths.SourceMinIO, paths.SourceMinIODefault, paths.Artifacts} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatalf("create harness path %s: %v", path, err)
		}
	}
	if err := os.WriteFile(paths.ProductionInventory, []byte("{}\n"), 0o600); err != nil {
		t.Fatalf("write production inventory: %v", err)
	}
	runner := &recordingRunner{}
	harness := configuredTestHarness(t, paths, runner)
	harness.valueEntropy = bytes.NewReader(bytes.Repeat([]byte{0xab}, 32))
	if err := harness.runCompose(context.Background(), "a-restore"); err != nil {
		t.Fatalf("run compose: %v", err)
	}
	if len(runner.environments) != 2 {
		t.Fatalf("compose environment count = %d, want 2", len(runner.environments))
	}
	for _, environment := range runner.environments {
		if environment[minioUserEnvironment] == "" || environment[minioPasswordEnvironment] == "" {
			t.Fatal("compose credential environment contains an empty value")
		}
	}
	composeBody, err := os.ReadFile(paths.ComposeFile)
	if err != nil {
		t.Fatalf("read compose file: %v", err)
	}
	for _, value := range runner.environments[0] {
		if strings.Contains(string(composeBody), value) {
			t.Fatal("compose file contains generated credential value")
		}
	}
}

func TestCreateIsolationLayoutUsesOnlyRunRoot(t *testing.T) {
	runRoot := filepath.Join(t.TempDir(), "20260812T010203Z-abcdef01")
	paths := pathsForRun(runRoot)
	if err := createIsolationLayout(paths); err != nil {
		t.Fatalf("create isolation layout: %v", err)
	}
	for _, path := range []string{
		paths.SourceEtcd,
		paths.SourceMilvus,
		paths.SourceMinIO,
		paths.SourceMinIODefault,
		paths.LMSState,
		paths.LMSContext,
		paths.LMSLogs,
		paths.ClydeConfig,
		paths.ClydeData,
		paths.ClydeState,
		paths.ClydeCache,
		paths.ClydeRuntime,
		paths.Artifacts,
	} {
		info, err := os.Stat(path)
		if err != nil || !info.IsDir() {
			t.Fatalf("directory %q: info=%v error=%v", path, info, err)
		}
		if !strings.HasPrefix(path, runRoot+string(filepath.Separator)) {
			t.Fatalf("path escaped run root: %q", path)
		}
	}
	configBody, err := os.ReadFile(paths.ClydeConfigFile)
	if err != nil {
		t.Fatalf("read Clyde config: %v", err)
	}
	want := "socket_path = \"" + paths.LMSSocket + "\""
	if !strings.Contains(string(configBody), want) {
		t.Fatalf("Clyde config = %q, want %q", configBody, want)
	}
}

func TestRestoreArchiveMakesImmutableSourceAndWritableCaseCopy(t *testing.T) {
	var archive bytes.Buffer
	writer := tar.NewWriter(&archive)
	if err := writer.WriteHeader(&tar.Header{Name: "./", Typeflag: tar.TypeDir, Mode: 0o755}); err != nil {
		t.Fatalf("write root directory header: %v", err)
	}
	if err := writer.WriteHeader(&tar.Header{Name: "nested", Typeflag: tar.TypeDir, Mode: 0o755}); err != nil {
		t.Fatalf("write directory header: %v", err)
	}
	body := []byte("restored")
	if err := writer.WriteHeader(&tar.Header{Name: "nested/data", Size: int64(len(body)), Mode: 0o644}); err != nil {
		t.Fatalf("write file header: %v", err)
	}
	if _, err := writer.Write(body); err != nil {
		t.Fatalf("write archive body: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close archive: %v", err)
	}
	archivePath := filepath.Join(t.TempDir(), "data.tar")
	if err := os.WriteFile(archivePath, archive.Bytes(), 0o600); err != nil {
		t.Fatalf("write archive: %v", err)
	}
	source := filepath.Join(t.TempDir(), "source")
	t.Cleanup(func() {
		_ = filepath.WalkDir(source, func(path string, _ os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			return os.Chmod(path, 0o700)
		})
	})
	if err := restoreImmutableArchive(context.Background(), archivePath, source); err != nil {
		t.Fatalf("restore immutable archive: %v", err)
	}
	sourceInfo, err := os.Stat(filepath.Join(source, "nested", "data"))
	if err != nil {
		t.Fatalf("stat source: %v", err)
	}
	if sourceInfo.Mode().Perm()&0o222 != 0 {
		t.Fatalf("source mode = %o, want immutable", sourceInfo.Mode().Perm())
	}
	caseRoot := filepath.Join(t.TempDir(), "case")
	if err := cloneWritableTree(source, caseRoot); err != nil {
		t.Fatalf("copy writable tree: %v", err)
	}
	casePath := filepath.Join(caseRoot, "nested", "data")
	if err := os.WriteFile(casePath, []byte("changed"), 0o600); err != nil {
		t.Fatalf("case copy is not writable: %v", err)
	}
	sourceBody, err := os.ReadFile(filepath.Join(source, "nested", "data"))
	if err != nil {
		t.Fatalf("read source: %v", err)
	}
	if string(sourceBody) != "restored" {
		t.Fatalf("source changed through case copy: %q", sourceBody)
	}
}

func TestRestoreArchiveRejectsUnsafeRootEntries(t *testing.T) {
	testCases := []struct {
		name     string
		header   tar.Header
		contents []byte
	}{
		{name: "root file", header: tar.Header{Name: ".", Typeflag: tar.TypeReg, Size: 1}, contents: []byte("x")},
		{name: "parent traversal", header: tar.Header{Name: "../outside", Typeflag: tar.TypeReg}},
		{name: "absolute path", header: tar.Header{Name: "/outside", Typeflag: tar.TypeReg}},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			var archive bytes.Buffer
			writer := tar.NewWriter(&archive)
			if err := writer.WriteHeader(&testCase.header); err != nil {
				t.Fatalf("write unsafe header: %v", err)
			}
			if _, err := writer.Write(testCase.contents); err != nil {
				t.Fatalf("write unsafe contents: %v", err)
			}
			if err := writer.Close(); err != nil {
				t.Fatalf("close unsafe archive: %v", err)
			}
			err := restoreImmutableArchiveReader(
				context.Background(),
				bytes.NewReader(archive.Bytes()),
				filepath.Join(t.TempDir(), "restored"),
			)
			if err == nil {
				t.Fatal("unsafe archive entry was accepted")
			}
		})
	}
}

func TestEvidenceWritesEventsAndBothResultFormats(t *testing.T) {
	paths := pathsForRun(filepath.Join(t.TempDir(), "20260812T010203Z-abcdef01"))
	if err := os.MkdirAll(paths.Artifacts, 0o755); err != nil {
		t.Fatalf("create artifacts: %v", err)
	}
	clock := func() time.Time { return time.Date(2026, 8, 12, 1, 2, 3, 0, time.UTC) }
	recorder := newEvidenceRecorder(paths, clock)
	if err := recorder.Record("preflight", "passed", map[string]string{"guard": "ports"}); err != nil {
		t.Fatalf("record event: %v", err)
	}
	result := acceptanceResult{RunID: "20260812T010203Z-abcdef01", Status: "passed"}
	if err := recorder.Finish(result); err != nil {
		t.Fatalf("finish evidence: %v", err)
	}

	eventsBody, err := os.ReadFile(paths.EventsJSONL)
	if err != nil {
		t.Fatalf("read events: %v", err)
	}
	var event evidenceEvent
	if err := json.Unmarshal(eventsBody, &event); err != nil {
		t.Fatalf("decode event: %v", err)
	}
	if event.Phase != "preflight" || event.Status != "passed" {
		t.Fatalf("event = %+v", event)
	}
	for _, path := range []string{paths.ResultJSON, paths.ResultMarkdown} {
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		if !strings.Contains(string(body), result.RunID) || !strings.Contains(string(body), result.Status) {
			t.Fatalf("result %s = %q", path, body)
		}
	}
}

func TestHarnessCleansTaggedResourcesAfterPartialStartup(t *testing.T) {
	runRoot := filepath.Join(t.TempDir(), "lms-restart-acceptance", "20260812T010203Z-abcdef01")
	paths := pathsForRun(runRoot)
	for _, path := range []string{paths.SourceEtcd, paths.SourceMilvus, paths.SourceMinIO, paths.SourceMinIODefault} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatalf("create source %s: %v", path, err)
		}
	}
	if err := os.MkdirAll(paths.Artifacts, 0o755); err != nil {
		t.Fatalf("create artifacts: %v", err)
	}
	if err := os.WriteFile(paths.ProductionInventory, []byte("{}\n"), 0o600); err != nil {
		t.Fatalf("write production inventory: %v", err)
	}
	runner := &recordingRunner{failAt: 1}
	harness := configuredTestHarness(t, paths, runner)
	err := harness.runCompose(context.Background(), "a-restore")
	if err == nil {
		t.Fatal("partial startup returned no error")
	}
	wantLast := []string{"docker", "compose", "-p", "lms-restart-abcdef01", "-f", paths.ComposeFile, "down", "--volumes", "--remove-orphans"}
	if !reflect.DeepEqual(runner.calls[len(runner.calls)-1], wantLast) {
		t.Fatalf("cleanup call = %v, want %v", runner.calls[len(runner.calls)-1], wantLast)
	}
}

func TestHarnessRechecksProductionImmediatelyBeforeComposeStartup(t *testing.T) {
	paths := pathsForRun(filepath.Join(t.TempDir(), "lms-restart-acceptance", "20260812T010203Z-abcdef01"))
	for _, path := range []string{paths.SourceEtcd, paths.SourceMilvus, paths.SourceMinIO, paths.SourceMinIODefault, paths.Artifacts} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatalf("create source: %v", err)
		}
	}
	runner := &recordingRunner{}
	harness := configuredTestHarness(t, paths, runner)
	readinessCalls := 0
	harness.readiness = func(context.Context) error {
		if len(runner.calls) != 0 {
			t.Fatalf("readiness ran after compose command %v", runner.calls)
		}
		readinessCalls++
		return nil
	}
	if err := harness.runCompose(context.Background(), "a"); err != nil {
		t.Fatalf("run compose: %v", err)
	}
	if readinessCalls != 1 {
		t.Fatalf("readiness calls = %d, want 1", readinessCalls)
	}
}

func TestHarnessCleansTaggedResourcesAfterSuccessfulStartup(t *testing.T) {
	paths := pathsForRun(filepath.Join(t.TempDir(), "lms-restart-acceptance", "20260812T010203Z-abcdef01"))
	for _, path := range []string{paths.SourceEtcd, paths.SourceMilvus, paths.SourceMinIO, paths.SourceMinIODefault} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatalf("create source %s: %v", path, err)
		}
	}
	if err := os.MkdirAll(paths.Artifacts, 0o755); err != nil {
		t.Fatalf("create artifacts: %v", err)
	}
	if err := os.WriteFile(paths.ProductionInventory, []byte("{}\n"), 0o600); err != nil {
		t.Fatalf("write production inventory: %v", err)
	}
	runner := &recordingRunner{}
	harness := configuredTestHarness(t, paths, runner)
	if err := harness.runCompose(context.Background(), "a-restore"); err != nil {
		t.Fatalf("run compose: %v", err)
	}
	if len(runner.calls) != 2 {
		t.Fatalf("compose calls = %v, want up and down", runner.calls)
	}
	if runner.calls[0][len(runner.calls[0])-1] != "--wait" {
		t.Fatalf("startup did not wait for healthy services: %v", runner.calls[0])
	}
	if runner.calls[1][len(runner.calls[1])-2] != "--volumes" {
		t.Fatalf("cleanup call = %v", runner.calls[1])
	}
}

func TestHarnessRefusesStartupBeforeProductionInventoryCapture(t *testing.T) {
	paths := pathsForRun(filepath.Join(t.TempDir(), "lms-restart-acceptance", "20260812T010203Z-abcdef01"))
	for _, path := range []string{paths.SourceEtcd, paths.SourceMilvus, paths.SourceMinIO, paths.SourceMinIODefault} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatalf("create source %s: %v", path, err)
		}
	}
	runner := &recordingRunner{}
	harness := configuredTestHarness(t, paths, runner)
	harness.inventory = inventoryToken{}
	if err := harness.runCompose(context.Background(), "a-restore"); err == nil {
		t.Fatal("compose started without production inventory")
	}
	if len(runner.calls) != 0 {
		t.Fatalf("commands ran without production inventory: %v", runner.calls)
	}
}

func TestHarnessRunsScenarioBetweenStartupAndCleanup(t *testing.T) {
	paths := pathsForRun(filepath.Join(t.TempDir(), "lms-restart-acceptance", "20260812T010203Z-abcdef01"))
	for _, path := range []string{paths.SourceEtcd, paths.SourceMilvus, paths.SourceMinIO, paths.SourceMinIODefault, paths.Artifacts} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatalf("create harness path %s: %v", path, err)
		}
	}
	if err := os.WriteFile(paths.ProductionInventory, []byte("{}\n"), 0o600); err != nil {
		t.Fatalf("write production inventory: %v", err)
	}
	runner := &recordingRunner{}
	harness := configuredTestHarness(t, paths, runner)
	scenarioRan := false
	err := harness.withCompose(context.Background(), "a-restore", func(context.Context) error {
		scenarioRan = true
		if len(runner.calls) != 1 || runner.calls[0][len(runner.calls[0])-1] != "--wait" {
			t.Fatalf("startup calls before scenario = %v", runner.calls)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("run scenario: %v", err)
	}
	if !scenarioRan || len(runner.calls) != 2 {
		t.Fatalf("scenario ran=%v calls=%v", scenarioRan, runner.calls)
	}
}

func TestHarnessRefusesUntaggedComposeProject(t *testing.T) {
	paths := pathsForRun(filepath.Join(t.TempDir(), "lms-restart-acceptance", "20260812T010203Z-abcdef01"))
	runner := &recordingRunner{}
	harness := &harness{paths: paths, composeProject: "production", runner: runner}
	if err := harness.runCompose(context.Background(), "a-restore"); err == nil {
		t.Fatal("untagged compose project was accepted")
	}
	if len(runner.calls) != 0 {
		t.Fatalf("unsafe project executed commands: %v", runner.calls)
	}
}

func TestResolveCaseRootRejectsUnsafeAndExistingNames(t *testing.T) {
	casesRoot := t.TempDir()
	for _, name := range []string{"case-a", "../a", "/tmp/a", "i", "a_bad", "A"} {
		if _, err := resolveCaseRoot(casesRoot, name); err == nil {
			t.Fatalf("unsafe case name %q was accepted", name)
		}
	}
	if _, err := resolveCaseRoot(casesRoot, "a-restore"); err != nil {
		t.Fatalf("safe case name: %v", err)
	}
	if err := os.Mkdir(filepath.Join(casesRoot, "a-restore"), 0o700); err != nil {
		t.Fatalf("create existing case: %v", err)
	}
	if _, err := resolveCaseRoot(casesRoot, "a-restore"); err == nil {
		t.Fatal("existing case root was accepted")
	}
}

func TestHarnessRechecksSpaceAndDeletesCaseAfterCleanup(t *testing.T) {
	paths := pathsForRun(filepath.Join(t.TempDir(), "lms-restart-acceptance", "20260812T010203Z-abcdef01"))
	for _, path := range []string{paths.SourceEtcd, paths.SourceMilvus, paths.SourceMinIO, paths.SourceMinIODefault} {
		if err := os.MkdirAll(path, 0o700); err != nil {
			t.Fatalf("create source: %v", err)
		}
	}
	runner := &recordingRunner{}
	harness := configuredTestHarness(t, paths, runner)
	spaceChecks := 0
	harness.availableBytes = func(string) (int64, error) { spaceChecks++; return 1 << 30, nil }
	if err := harness.runCompose(context.Background(), "b-space"); err != nil {
		t.Fatalf("run compose: %v", err)
	}
	if spaceChecks != 1 {
		t.Fatalf("space checks = %d, want 1", spaceChecks)
	}
	if _, err := os.Lstat(filepath.Join(paths.Cases, "b-space")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("case tree remains: %v", err)
	}
	if len(runner.deadlines) != 2 || runner.deadlines[1].IsZero() {
		t.Fatalf("cleanup deadline was not bounded: %v", runner.deadlines)
	}
}

func TestHarnessRequiresFullWritableCaseReserveAfterRestore(t *testing.T) {
	paths := pathsForRun(filepath.Join(t.TempDir(), "lms-restart-acceptance", "20260812T010203Z-abcdef01"))
	for _, path := range []string{paths.SourceEtcd, paths.SourceMilvus, paths.SourceMinIO, paths.SourceMinIODefault} {
		if err := os.MkdirAll(path, 0o700); err != nil {
			t.Fatalf("create source: %v", err)
		}
	}
	harness := configuredTestHarness(t, paths, &recordingRunner{})
	harness.archiveSizes = []int64{100}
	harness.availableBytes = func(string) (int64, error) { return 125, nil }
	if err := harness.runCompose(context.Background(), "a-space"); err != nil {
		t.Fatalf("run compose with post-restore reserve: %v", err)
	}
	harness.availableBytes = func(string) (int64, error) { return 124, nil }
	if err := harness.runCompose(context.Background(), "b-space"); err == nil {
		t.Fatal("run compose without post-restore reserve succeeded")
	}
}

func TestInventoryTokenRejectsStaleMismatchedAndTamperedValues(t *testing.T) {
	now := time.Date(2026, 8, 12, 1, 7, 3, 0, time.UTC)
	inventory := productionInventory{Databases: []string{"default"}, Collections: collectionCensus{{Database: "default", Collection: "operator"}: "hash"}}
	token, err := newInventoryToken("20260812T010203Z-abcdef01", now.Add(-5*time.Minute), inventory)
	if err != nil {
		t.Fatalf("new token: %v", err)
	}
	if err := validateInventoryToken(token, token.RunID, now); err != nil {
		t.Fatalf("five-minute token: %v", err)
	}
	stale := token
	stale.CapturedAt = stale.CapturedAt.Add(-time.Nanosecond)
	if err := validateInventoryToken(stale, stale.RunID, now); err == nil {
		t.Fatal("stale token was accepted")
	}
	if err := validateInventoryToken(token, "20260812T010203Z-deadbeef", now); err == nil {
		t.Fatal("mismatched token was accepted")
	}
	tampered := token
	tampered.Inventory.Collections[collectionIdentity{Database: "default", Collection: "operator"}] = "changed"
	if err := validateInventoryToken(tampered, tampered.RunID, now); err == nil {
		t.Fatal("tampered token was accepted")
	}
}

func TestCollectionCensusSerializationIsDeterministic(t *testing.T) {
	first := collectionCensus{
		{Database: "z", Collection: "second"}: "hash-b",
		{Database: "a", Collection: "first"}:  "hash-a",
	}
	second := collectionCensus{
		{Database: "a", Collection: "first"}:  "hash-a",
		{Database: "z", Collection: "second"}: "hash-b",
	}
	firstBody, err := json.Marshal(first)
	if err != nil {
		t.Fatalf("marshal first census: %v", err)
	}
	secondBody, err := json.Marshal(second)
	if err != nil {
		t.Fatalf("marshal second census: %v", err)
	}
	if !bytes.Equal(firstBody, secondBody) {
		t.Fatalf("census serialization differs: %s != %s", firstBody, secondBody)
	}
}

func TestIsolatedEnvironmentRoutesOnlyCloneResources(t *testing.T) {
	t.Parallel()

	paths := pathsForRun(filepath.Join(t.TempDir(), "20260812T010203Z-abcdef01"))
	lmsEnvironment := isolatedLMSEnvironment(paths)
	wantLMS := map[string]string{
		"XDG_STATE_HOME":               paths.LMSState,
		"CLAUDE_CONTEXTD_CONTEXT_ROOT": paths.LMSContext,
		"CLAUDE_CONTEXTD_SOCKET_PATH":  paths.LMSSocket,
		"MILVUS_ADDRESS":               "127.0.0.1:39530",
		"MILVUS_DATABASE":              "default",
		"OPENAI_BASE_URL":              "http://127.0.0.1:35400",
	}
	if !reflect.DeepEqual(lmsEnvironment, wantLMS) {
		t.Fatalf("LMS environment = %#v, want %#v", lmsEnvironment, wantLMS)
	}
	clydeEnvironment := isolatedClydeEnvironment(paths)
	wantClyde := map[string]string{
		"HOME":                        paths.ClydeHome,
		"XDG_CONFIG_HOME":             paths.ClydeConfig,
		"XDG_DATA_HOME":               paths.ClydeData,
		"XDG_STATE_HOME":              paths.ClydeState,
		"XDG_CACHE_HOME":              paths.ClydeCache,
		"XDG_RUNTIME_DIR":             paths.ClydeRuntime,
		"CLAUDE_CONTEXTD_SOCKET_PATH": paths.LMSSocket,
	}
	if !reflect.DeepEqual(clydeEnvironment, wantClyde) {
		t.Fatalf("Clyde environment = %#v, want %#v", clydeEnvironment, wantClyde)
	}
}

func TestCaptureProductionInventoryWritesSeparateReadOnlyArtifact(t *testing.T) {
	paths := pathsForRun(filepath.Join(t.TempDir(), "20260812T010203Z-abcdef01"))
	if err := os.MkdirAll(paths.Artifacts, 0o755); err != nil {
		t.Fatalf("create artifacts: %v", err)
	}
	runner := &recordingRunner{outputs: [][]byte{
		[]byte(`{"indexes":[],"dependencyHealth":{}}`),
		[]byte(`{"jobs":[{"id":"operator","state":"running"}],"dependencyHealth":{}}`),
		[]byte(`{"commit":"abc"}`),
	}}
	binaries := installedBinaries{CLI: "/home/test/.local/bin/lm-semantic-search"}
	runID := "20260812T010203Z-abcdef01"
	_, err := captureProductionInventory(
		context.Background(),
		paths,
		binaries,
		runner,
		runID,
		time.Date(2026, 8, 12, 1, 2, 3, 0, time.UTC),
		func(context.Context) (productionMilvusCensus, error) {
			return productionMilvusCensus{
				Databases:   []string{"default"},
				Collections: collectionCensus{{Database: "default", Collection: "operator"}: "properties-loadstate-hash"},
			}, nil
		},
	)
	if err != nil {
		t.Fatalf("capture production inventory: %v", err)
	}
	wantCalls := [][]string{
		{binaries.CLI, "--json", "codebase", "list"},
		{binaries.CLI, "--json", "job", "list"},
		{binaries.CLI, "--json", "daemon", "status"},
	}
	if !reflect.DeepEqual(runner.calls, wantCalls) {
		t.Fatalf("inventory calls = %v, want %v", runner.calls, wantCalls)
	}
	body, err := os.ReadFile(paths.ProductionInventory)
	if err != nil {
		t.Fatalf("read production inventory: %v", err)
	}
	if !strings.Contains(string(body), `"commit": "abc"`) {
		t.Fatalf("production inventory = %s", body)
	}
}

func TestValidateProductionHealthAllowsOperatorJobsAfterInitialPreflight(t *testing.T) {
	t.Parallel()

	indexes := json.RawMessage(`{"indexes":[],"dependencyHealth":{}}`)
	jobs := json.RawMessage(`{"jobs":[{"id":"operator","state":"running"}],"dependencyHealth":{}}`)
	if err := validateProductionHealth(indexes, jobs); err != nil {
		t.Fatalf("healthy production with operator activity: %v", err)
	}
	if err := validateProductionReadiness(indexes, jobs); err == nil {
		t.Fatal("initial production readiness accepted an active job")
	}
	unknownJobs := json.RawMessage(`{"jobs":[{"id":"operator","state":"mystery"}],"dependencyHealth":{}}`)
	if err := validateProductionHealth(indexes, unknownJobs); err == nil {
		t.Fatal("production health accepted an unknown job state")
	}
}

func TestValidateProductionReadinessRequiresZeroActiveJobsAndHealthyDependencies(t *testing.T) {
	t.Parallel()

	healthyIndexes := json.RawMessage(`{"indexes":[],"dependencyHealth":{}}`)
	healthyJobs := json.RawMessage(`{"jobs":[{"id":"old","state":"completed"}],"dependencyHealth":{}}`)
	if err := validateProductionReadiness(healthyIndexes, healthyJobs); err != nil {
		t.Fatalf("healthy production readiness: %v", err)
	}
	tests := []struct {
		name    string
		indexes json.RawMessage
		jobs    json.RawMessage
	}{
		{
			name:    "active job",
			indexes: healthyIndexes,
			jobs:    json.RawMessage(`{"jobs":[{"id":"current","state":"running"}],"dependencyHealth":{}}`),
		},
		{
			name:    "degraded index dependency",
			indexes: json.RawMessage(`{"indexes":[],"dependencyHealth":{"degraded":true,"mode":"store_unavailable"}}`),
			jobs:    healthyJobs,
		},
		{
			name:    "degraded job dependency",
			indexes: healthyIndexes,
			jobs:    json.RawMessage(`{"jobs":[],"dependencyHealth":{"degraded":true,"mode":"embedder_unreachable"}}`),
		},
		{name: "missing index health", indexes: json.RawMessage(`{"indexes":[]}`), jobs: healthyJobs},
		{name: "null job health", indexes: healthyIndexes, jobs: json.RawMessage(`{"jobs":[],"dependencyHealth":null}`)},
		{name: "unknown index mode", indexes: json.RawMessage(`{"indexes":[],"dependencyHealth":{"mode":"ready"}}`), jobs: healthyJobs},
		{name: "contradictory job health", indexes: healthyIndexes, jobs: json.RawMessage(`{"jobs":[],"dependencyHealth":{"degraded":false,"mode":"store_unavailable"}}`)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if err := validateProductionReadiness(test.indexes, test.jobs); err == nil {
				t.Fatal("unsafe production readiness was accepted")
			}
		})
	}
}

func TestAuditProductionMutationRejectsRemovedChangedAndAttributedAdditions(t *testing.T) {
	t.Parallel()

	operator := collectionIdentity{Database: "default", Collection: "operator"}
	newCollection := collectionIdentity{Database: "default", Collection: "new"}
	otherCollection := collectionIdentity{Database: "default", Collection: "other"}
	before := productionInventory{Collections: collectionCensus{operator: "hash-a"}}
	tests := []struct {
		name  string
		after productionInventory
		calls []milvusProxyCall
	}{
		{name: "removed", after: productionInventory{Collections: collectionCensus{}}},
		{name: "changed", after: productionInventory{Collections: collectionCensus{operator: "hash-b"}}},
		{
			name:  "attributed addition",
			after: productionInventory{Collections: collectionCensus{operator: "hash-a", newCollection: "hash-c"}},
			calls: []milvusProxyCall{{Database: "default", Collection: "new", Method: "CreateCollection"}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if err := auditProductionMutation(before, test.after, test.calls, nil); err == nil {
				t.Fatal("production mutation was accepted")
			}
		})
	}
	concurrent := productionInventory{Collections: collectionCensus{operator: "hash-a", otherCollection: "hash-d"}}
	if err := auditProductionMutation(before, concurrent, nil, nil); err != nil {
		t.Fatalf("unattributed concurrent addition: %v", err)
	}
	protected := map[collectionIdentity]struct{}{otherCollection: {}}
	if err := auditProductionMutation(before, concurrent, nil, protected); err == nil {
		t.Fatal("precomputed acceptance collection was accepted without proxy evidence")
	}
}

func TestAuditProductionMutationChecksSamplesOnlyWhenBothCensusesLoaded(t *testing.T) {
	t.Parallel()

	identity := collectionIdentity{Database: "default", Collection: "operator"}
	before := productionInventory{
		Databases:   []string{"default"},
		Collections: collectionCensus{identity: "durable"},
		Samples:     collectionCensus{identity: "sample-a"},
	}
	changed := productionInventory{
		Databases:   []string{"default"},
		Collections: collectionCensus{identity: "durable"},
		Samples:     collectionCensus{identity: "sample-b"},
	}
	if err := auditProductionMutation(before, changed, nil, nil); err == nil {
		t.Fatal("loaded sample mutation was accepted")
	}
	cold := productionInventory{
		Databases:   []string{"default"},
		Collections: collectionCensus{identity: "durable"},
	}
	if err := auditProductionMutation(before, cold, nil, nil); err != nil {
		t.Fatalf("residency-only sample absence was rejected: %v", err)
	}
}

type recordingRunner struct {
	calls        [][]string
	environments []map[string]string
	failAt       int
	outputs      [][]byte
	deadlines    []time.Time
}

func configuredTestHarness(t *testing.T, paths runPaths, runner commandRunner) *harness {
	t.Helper()
	if err := os.MkdirAll(paths.Cases, 0o700); err != nil {
		t.Fatalf("create cases root: %v", err)
	}
	capturedAt := time.Date(2026, 8, 12, 1, 2, 3, 0, time.UTC)
	collections := collectionCensus{{Database: "default", Collection: "operator"}: "properties-loadstate-hash"}
	token, err := newInventoryToken(filepath.Base(paths.RunRoot), capturedAt, productionInventory{Databases: []string{"default"}, Collections: collections})
	if err != nil {
		t.Fatalf("create inventory token: %v", err)
	}
	return &harness{
		paths:          paths,
		composeProject: "lms-restart-abcdef01",
		runner:         runner,
		archiveSizes:   []int64{1},
		availableBytes: func(string) (int64, error) { return 1 << 30, nil },
		inventory:      token,
		now:            func() time.Time { return capturedAt },
		census: func(context.Context) (productionMilvusCensus, error) {
			return productionMilvusCensus{Databases: []string{"default"}, Collections: cloneCollectionCensus(collections)}, nil
		},
		readiness: func(context.Context) error { return nil },
	}
}

func (runner *recordingRunner) Run(
	ctx context.Context,
	environment map[string]string,
	name string,
	arguments ...string,
) ([]byte, error) {
	call := append([]string{name}, arguments...)
	runner.calls = append(runner.calls, call)
	runner.environments = append(runner.environments, environment)
	deadline, _ := ctx.Deadline()
	runner.deadlines = append(runner.deadlines, deadline)
	if len(runner.calls) == runner.failAt {
		return nil, errors.New("injected startup failure")
	}
	if len(runner.outputs) >= len(runner.calls) {
		return runner.outputs[len(runner.calls)-1], nil
	}
	return nil, nil
}
