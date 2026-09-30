//go:build live

package live

import (
	"context"
	"crypto/sha256"
	"database/sql"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/gksyntax/chunk"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/sandbox"
	"goodkind.io/lm-semantic-search/internal/store"
	"goodkind.io/lm-semantic-search/test/sandboxharness"
)

//go:embed library_codebase_namespace_rows.sql
var codebaseNamespaceRowsSQL string

const (
	// codebaseFunctionBytes sizes one generated function below the 2,500-byte
	// splitter chunk size and above half of it. The splitter then emits one
	// chunk per function.
	codebaseFunctionBytes = 1800
	// codebaseJobTimeout bounds one index or sync job with real embeddings.
	codebaseJobTimeout = 5 * time.Minute
	codebaseJobPoll    = 200 * time.Millisecond
)

// libraryCodebaseDaemon is one in-process daemon with the library codebase
// store over an isolated Milvus database.
type libraryCodebaseDaemon struct {
	harness *libraryHarness
	config  config.Config
	manager *daemon.Manager
	client  pb.SemanticSearchDaemonServiceClient
	stop    func()
}

func newLibraryCodebaseDaemon(t *testing.T) *libraryCodebaseDaemon {
	t.Helper()
	return newCodebaseLiveDaemon(t, config.CodebaseStoreLibrary)
}

func newCodebaseLiveDaemon(t *testing.T, codebaseStore config.CodebaseStoreKind) *libraryCodebaseDaemon {
	t.Helper()
	harness := newLibraryHarness(t)
	return newCodebaseLiveDaemonWithHarness(t, harness, codebaseStore)
}

func newCodebaseLiveDaemonWithHarness(t *testing.T, harness *libraryHarness, codebaseStore config.CodebaseStoreKind) *libraryCodebaseDaemon {
	t.Helper()
	sandboxRoot := t.TempDir()
	socketDir, err := os.MkdirTemp("/tmp", "lms-lib-live-")
	if err != nil {
		t.Fatalf("create socket directory: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(socketDir) })
	settings := [][2]string{
		{"CLAUDE_CONTEXT_PROFILE", config.ProfileStandard},
		{"CLAUDE_CONTEXT_CODEBASE_STORE", string(codebaseStore)},
		{"MILVUS_ADDRESS", harness.environment.MilvusAddress},
		{"MILVUS_DATABASE", harness.database},
		{"EMBEDDING_PROVIDER", string(model.EmbeddingProviderOpenAI)},
		{"EMBEDDING_MODEL", harness.environment.EmbeddingModel},
		{"OPENAI_BASE_URL", harness.environment.EmbeddingURL},
		{"OPENAI_API_KEY", harness.environment.APIKey},
		{"EMBEDDING_DIMENSION", strconv.Itoa(libraryLiveDimension)},
		{"CLAUDE_CONTEXTD_SOCKET_PATH", filepath.Join(socketDir, "daemon.sock")},
		{"CLAUDE_CONTEXT_BACKGROUND_SYNC", "false"},
		{"CLAUDE_CONTEXT_TRIGGER_WATCHER", "false"},
		{"CLAUDE_CONTEXT_FILE_WATCHER", "false"},
		{"CLAUDE_CONTEXT_DEBUG_LISTENER", "false"},
		{"CLAUDE_CONTEXT_PERF_COUNTERS_INTERVAL_MS", "0"},
		{"CLAUDE_CONTEXT_MAX_CONCURRENT_INDEX_JOBS", "1"},
		{"CLAUDE_CONTEXT_RESUME_ON_BOOT", "false"},
	}
	for _, setting := range settings {
		t.Setenv(setting[0], setting[1])
	}
	for _, variable := range sandbox.Env(sandboxRoot) {
		if _, alreadySet := os.LookupEnv(variable.Name); alreadySet {
			continue
		}
		t.Setenv(variable.Name, variable.Value)
	}
	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("resolve daemon config: %v", err)
	}
	if cfg.CodebaseStore != codebaseStore || cfg.MilvusDatabase != harness.database {
		t.Fatalf("daemon config store %q database %q, want %s and %s", cfg.CodebaseStore, cfg.MilvusDatabase, codebaseStore, harness.database)
	}
	for _, directory := range sandbox.Directories(cfg) {
		if err := store.EnsureDir(directory); err != nil {
			t.Fatalf("create %s: %v", directory, err)
		}
	}
	if err := store.WriteRegistry(cfg.RegistryPath, model.RegistryFile{}); err != nil {
		t.Fatalf("write registry: %v", err)
	}
	manager, err := daemon.NewManager(harness.context(), cfg)
	if err != nil {
		t.Fatalf("start daemon manager: %v", err)
	}
	stopServer := startInProcessServer(t, harness.context(), manager, cfg.SocketPath)
	conn, client, err := grpcutil.DialDaemon(context.Background(), cfg.SocketPath)
	if err != nil {
		t.Fatalf("dial daemon: %v", err)
	}
	var closeOnce sync.Once
	stop := func() {
		closeOnce.Do(func() {
			_ = conn.Close()
			stopServer()
			closeCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			if err := manager.Close(closeCtx); err != nil {
				t.Errorf("close daemon manager: %v", err)
			}
		})
	}
	t.Cleanup(stop)
	return &libraryCodebaseDaemon{harness: harness, config: cfg, manager: manager, client: client, stop: stop}
}

func (codebaseDaemon *libraryCodebaseDaemon) waitJob(t *testing.T, jobID string, label string) {
	t.Helper()
	deadline := time.Now().Add(codebaseJobTimeout)
	for time.Now().Before(deadline) {
		response, err := codebaseDaemon.client.GetJob(context.Background(), &pb.GetJobRequest{JobId: jobID})
		if err != nil {
			t.Fatalf("%s: get job %s: %v", label, jobID, err)
		}
		switch response.GetJob().GetState() {
		case "completed":
			return
		case "failed", "cancelled":
			t.Fatalf("%s: job %s ended %s: %s", label, jobID, response.GetJob().GetState(), response.GetJob().GetDisplayError())
		}
		time.Sleep(codebaseJobPoll)
	}
	t.Fatalf("%s: job %s did not finish within %s", label, jobID, codebaseJobTimeout)
}

func (codebaseDaemon *libraryCodebaseDaemon) index(t *testing.T, root string) {
	t.Helper()
	response, err := codebaseDaemon.client.StartIndex(context.Background(), &pb.StartIndexRequest{
		Path:     root,
		Splitter: &pb.SplitterConfig{Type: "ast"},
		Client:   &pb.ClientInfo{Name: "library-codebase-live"},
	})
	if err != nil {
		t.Fatalf("start index: %v", err)
	}
	codebaseDaemon.waitJob(t, response.GetJobId(), "index")
}

func (codebaseDaemon *libraryCodebaseDaemon) sync(t *testing.T, root string) {
	t.Helper()
	codebaseDaemon.waitJob(t, codebaseDaemon.startSync(t, root), "sync")
}

func (codebaseDaemon *libraryCodebaseDaemon) startSync(t *testing.T, root string) string {
	t.Helper()
	response, err := codebaseDaemon.client.SyncIndex(context.Background(), &pb.SyncIndexRequest{
		Path:   root,
		Client: &pb.ClientInfo{Name: "library-codebase-live"},
	})
	if err != nil {
		t.Fatalf("sync index: %v", err)
	}
	return response.GetJobId()
}

func (codebaseDaemon *libraryCodebaseDaemon) cancelPausedJob(t *testing.T, jobID string, release func()) string {
	t.Helper()
	if _, err := codebaseDaemon.client.CancelJob(context.Background(), &pb.CancelJobRequest{
		JobId:  jobID,
		Client: &pb.ClientInfo{Name: "library-codebase-live"},
	}); err != nil {
		t.Fatalf("cancel job %s: %v", jobID, err)
	}
	release()
	return codebaseDaemon.waitTerminalJob(t, jobID)
}

func (codebaseDaemon *libraryCodebaseDaemon) waitTerminalJob(t *testing.T, jobID string) string {
	t.Helper()
	deadline := time.Now().Add(codebaseJobTimeout)
	for time.Now().Before(deadline) {
		response, err := codebaseDaemon.client.GetJob(context.Background(), &pb.GetJobRequest{JobId: jobID})
		if err != nil {
			t.Fatalf("get job %s: %v", jobID, err)
		}
		switch state := response.GetJob().GetState(); state {
		case "completed", "failed", "cancelled":
			return state
		}
		time.Sleep(codebaseJobPoll)
	}
	t.Fatalf("job %s did not finish within %s", jobID, codebaseJobTimeout)
	return ""
}

// stagedRowCount counts staged, unpublished occurrences in the catalog.
func (codebaseDaemon *libraryCodebaseDaemon) stagedRowCount(t *testing.T) int {
	t.Helper()
	catalogPath := filepath.Join(codebaseDaemon.config.StateRoot, "library", "codebase", "catalog.sqlite")
	database, err := sql.Open("sqlite3", "file:"+catalogPath+"?mode=ro")
	if err != nil {
		t.Fatalf("open catalog %s: %v", catalogPath, err)
	}
	defer func() { _ = database.Close() }()
	var count int
	if err := database.QueryRowContext(codebaseDaemon.harness.context(), `SELECT COUNT(*) FROM staged_occurrences`).Scan(&count); err != nil {
		t.Fatalf("count staged occurrences: %v", err)
	}
	return count
}

// codebaseOccurrence is one published occurrence read from the catalog.
type codebaseOccurrence struct {
	rowKey          string
	vectorID        string
	source          string
	occurrenceHash  string
	generationOrder int64
}

// readCodebaseOwners reads every published occurrence of the only codebase
// namespace, grouped by owner, with its own SQLite connection.
func (codebaseDaemon *libraryCodebaseDaemon) readCodebaseOwners(t *testing.T) map[string][]codebaseOccurrence {
	t.Helper()
	catalogPath := filepath.Join(codebaseDaemon.config.StateRoot, "library", "codebase", "catalog.sqlite")
	database, err := sql.Open("sqlite3", "file:"+catalogPath+"?mode=ro")
	if err != nil {
		t.Fatalf("open catalog %s: %v", catalogPath, err)
	}
	defer func() { _ = database.Close() }()
	var namespaceCount int
	if err := database.QueryRowContext(codebaseDaemon.harness.context(), `SELECT COUNT(*) FROM namespaces`).Scan(&namespaceCount); err != nil {
		t.Fatalf("count namespaces: %v", err)
	}
	if namespaceCount != 1 {
		t.Fatalf("catalog has %d namespaces, want 1", namespaceCount)
	}
	rows, err := database.QueryContext(
		codebaseDaemon.harness.context(),
		`SELECT occurrences.owner_id, occurrences.row_key, occurrences.vector_id, source_blobs.content,
		occurrences.occurrence_hash, occurrences.generation_order
		FROM occurrences JOIN source_blobs ON source_blobs.blob_id = occurrences.source_blob_id
		ORDER BY occurrences.owner_id, occurrences.sort_key`,
	)
	if err != nil {
		t.Fatalf("read occurrences: %v", err)
	}
	defer func() { _ = rows.Close() }()
	owners := map[string][]codebaseOccurrence{}
	for rows.Next() {
		var owner string
		var occurrence codebaseOccurrence
		if err := rows.Scan(&owner, &occurrence.rowKey, &occurrence.vectorID, &occurrence.source, &occurrence.occurrenceHash, &occurrence.generationOrder); err != nil {
			t.Fatalf("scan occurrence: %v", err)
		}
		owners[owner] = append(owners[owner], occurrence)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("read occurrences: %v", err)
	}
	return owners
}

// goFunction returns one Go function with a unique marker, padded to
// codebaseFunctionBytes.
func goFunction(name string, marker string) string {
	header := fmt.Sprintf("// %s returns the %s marker.\nfunc %s() string {\n\treturn \"%s", name, marker, name, marker)
	footer := "\"\n}\n"
	padding := strings.Repeat(" "+marker, (codebaseFunctionBytes-len(header)-len(footer))/(len(marker)+1))
	return header + padding + footer
}

func goFile(functions ...string) string {
	return "package fixture\n\n" + strings.Join(functions, "\n")
}

func writeCodebaseFile(t *testing.T, root string, name string, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0o600); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}

// assertExcerpts requires every occurrence source excerpt of owner to appear
// in the file content.
func assertExcerpts(t *testing.T, owners map[string][]codebaseOccurrence, owner string, content string) {
	t.Helper()
	for _, occurrence := range owners[owner] {
		if !strings.Contains(content, occurrence.source) {
			t.Fatalf("%s occurrence %s excerpt is not in the current file", owner, occurrence.rowKey)
		}
	}
}

func TestLibraryCodebaseReplacesFileOwners(t *testing.T) {
	codebaseDaemon := newLibraryCodebaseDaemon(t)
	t.Logf("embedding window start %s", time.Now().UTC().Format(time.RFC3339))
	defer func() { t.Logf("embedding window end %s", time.Now().UTC().Format(time.RFC3339)) }()
	root := t.TempDir()

	grow := goFile(goFunction("GrowOne", "growone"))
	shrink := goFile(goFunction("ShrinkOne", "shrinkone"), goFunction("ShrinkTwo", "shrinktwo"), goFunction("ShrinkThree", "shrinkthree"))
	duplicate := goFile(goFunction("Shared", "shared"))
	stable := goFile(goFunction("Stable", "stable"))
	files := map[string]string{
		"grow.go":   grow,
		"shrink.go": shrink,
		"delete.go": goFile(goFunction("Deleted", "deleted")),
		"dup_a.go":  duplicate,
		"dup_b.go":  duplicate,
		"stable.go": stable,
	}
	for name, content := range files {
		writeCodebaseFile(t, root, name, content)
	}
	codebaseDaemon.index(t, root)
	before := codebaseDaemon.readCodebaseOwners(t)
	if got := len(before["shrink.go"]); got != 3 {
		t.Fatalf("shrink.go has %d occurrences after the first index, want 3", got)
	}
	for name, content := range files {
		if len(before[name]) == 0 {
			t.Fatalf("%s has no occurrence after the first index", name)
		}
		assertExcerpts(t, before, name, content)
	}
	if !slices.EqualFunc(before["dup_a.go"], before["dup_b.go"], func(left codebaseOccurrence, right codebaseOccurrence) bool {
		return left.vectorID == right.vectorID
	}) {
		t.Fatal("identical files in dup_a.go and dup_b.go reference different vectors")
	}

	grown := goFile(goFunction("GrowOne", "growone"), goFunction("GrowTwo", "growtwo"))
	shrunk := goFile(goFunction("ShrinkOne", "shrinkone"), goFunction("ShrinkTwo", "shrinktwo"))
	writeCodebaseFile(t, root, "grow.go", grown)
	writeCodebaseFile(t, root, "shrink.go", shrunk)
	if err := os.Remove(filepath.Join(root, "delete.go")); err != nil {
		t.Fatalf("remove delete.go: %v", err)
	}
	codebaseDaemon.sync(t, root)
	after := codebaseDaemon.readCodebaseOwners(t)

	if got := len(after["shrink.go"]); got != 2 {
		t.Fatalf("shrink.go has %d occurrences after shrinking, want 2", got)
	}
	assertExcerpts(t, after, "shrink.go", shrunk)
	if len(after["grow.go"]) <= len(before["grow.go"]) {
		t.Fatalf("grow.go has %d occurrences after growing, before %d", len(after["grow.go"]), len(before["grow.go"]))
	}
	assertExcerpts(t, after, "grow.go", grown)
	if got := len(after["delete.go"]); got != 0 {
		t.Fatalf("deleted delete.go still has %d occurrences", got)
	}
	for _, name := range []string{"stable.go", "dup_a.go", "dup_b.go"} {
		if !slices.Equal(before[name], after[name]) {
			t.Fatalf("unchanged %s changed its published occurrences", name)
		}
	}
	backend := codebaseDaemon.harness.backendVectorIDs("lms_library_codebase")
	for owner, occurrences := range after {
		for _, occurrence := range occurrences {
			if _, found := slices.BinarySearch(backend, occurrence.vectorID); !found {
				t.Fatalf("%s occurrence %s references vector %s, which the backend does not return", owner, occurrence.rowKey, occurrence.vectorID)
			}
		}
	}
	distinct := map[string]struct{}{}
	for _, occurrences := range after {
		for _, occurrence := range occurrences {
			distinct[occurrence.vectorID] = struct{}{}
		}
	}
	total := 0
	for _, occurrences := range after {
		total += len(occurrences)
	}
	if len(distinct) >= total {
		t.Fatalf("%d occurrences reference %d distinct vectors, want reuse across dup_a.go and dup_b.go", total, len(distinct))
	}

	response, err := codebaseDaemon.client.ClearIndex(context.Background(), &pb.ClearIndexRequest{
		Path:   root,
		Client: &pb.ClientInfo{Name: "library-codebase-live"},
	})
	if err != nil {
		t.Fatalf("clear index: %v", err)
	}
	if !response.GetCleared() {
		t.Fatalf("clear index returned cleared=false: %s", response.GetDisplayText())
	}
	cleared := codebaseDaemon.readCodebaseOwners(t)
	if len(cleared) != 0 {
		t.Fatalf("catalog still has %d owners after clearing the codebase", len(cleared))
	}
}

// interruptedFunctionCount gives one file several staged batches at the default
// batch size of 32 rows.
const interruptedFunctionCount = 96

func largeGoFile(markerPrefix string) string {
	functions := make([]string, 0, interruptedFunctionCount)
	for number := range interruptedFunctionCount {
		functions = append(functions, goFunction(fmt.Sprintf("Function%d", number), fmt.Sprintf("%s%d", markerPrefix, number)))
	}
	return goFile(functions...)
}

func TestLibraryCodebaseKeepsOldGenerationUntilCommit(t *testing.T) {
	codebaseDaemon, proxy := newProxiedCodebaseDaemon(t)
	t.Logf("embedding window start %s", time.Now().UTC().Format(time.RFC3339))
	defer func() { t.Logf("embedding window end %s", time.Now().UTC().Format(time.RFC3339)) }()
	root := t.TempDir()
	oldContent := largeGoFile("old")
	writeCodebaseFile(t, root, "large.go", oldContent)
	codebaseDaemon.index(t, root)
	before := codebaseDaemon.readCodebaseOwners(t)
	if got := len(before["large.go"]); got != interruptedFunctionCount {
		t.Fatalf("large.go has %d occurrences after the first index, want %d", got, interruptedFunctionCount)
	}

	newContent := largeGoFile("new")
	writeCodebaseFile(t, root, "large.go", newContent)
	observed, release := proxy.PausePublication(codebaseDaemon.harness.database, "lms_library_codebase", sandboxharness.BeforeStorePublication)
	t.Cleanup(release)
	jobID := codebaseDaemon.startSync(t, root)
	select {
	case boundary := <-observed:
		if boundary.ResponseReceived || boundary.Method != "Upsert" || boundary.Phase != sandboxharness.BeforeStorePublication {
			t.Fatalf("unexpected cancellation boundary: %+v", boundary)
		}
		t.Logf("public CancelJob at real %s before backend forwarding", boundary.Method)
	case <-codebaseDaemon.harness.context().Done():
		t.Fatal("the changed codebase did not request vector publication")
	}
	state := codebaseDaemon.cancelPausedJob(t, jobID, release)
	if state != "cancelled" {
		t.Fatalf("interrupted sync ended %s, want cancelled", state)
	}
	interrupted := codebaseDaemon.readCodebaseOwners(t)
	if !slices.Equal(before["large.go"], interrupted["large.go"]) {
		t.Fatal("an interrupted generation changed the published occurrences of large.go")
	}
	assertExcerpts(t, interrupted, "large.go", oldContent)
	assertPublicCodebaseSource(t, codebaseDaemon, root, "old0", oldContent)

	codebaseDaemon.sync(t, root)
	after := codebaseDaemon.readCodebaseOwners(t)
	if got := len(after["large.go"]); got != interruptedFunctionCount {
		t.Fatalf("large.go has %d occurrences after the resumed sync, want %d", got, interruptedFunctionCount)
	}
	assertExcerpts(t, after, "large.go", newContent)
	assertPublicCodebaseSource(t, codebaseDaemon, root, "new0", newContent)
	if remaining := codebaseDaemon.stagedRowCount(t); remaining != 0 {
		t.Fatalf("catalog keeps %d staged rows after the resumed sync committed", remaining)
	}
	t.Logf("cancelled before backend forwarding; the resumed sync committed %d occurrences", len(after["large.go"]))
}

func assertPublicCodebaseSource(t *testing.T, codebaseDaemon *libraryCodebaseDaemon, root string, query string, content string) {
	t.Helper()
	response, err := codebaseDaemon.client.SearchCode(codebaseDaemon.harness.context(), &pb.SearchCodeRequest{
		Path: root, Query: query, Limit: codebaseSearchLimit,
	})
	if err != nil || len(response.GetResults()) == 0 {
		t.Fatalf("public SearchCode returned %d results, error=%v", len(response.GetResults()), err)
	}
	for _, result := range response.GetResults() {
		if result.GetRelativePath() != "large.go" || !strings.Contains(content, result.GetContent()) {
			t.Fatalf("public SearchCode returned an excerpt outside the committed source: %q", result.GetContent())
		}
	}
}

func TestLibraryCodebaseReportsRealBackendFailure(t *testing.T) {
	codebaseDaemon, proxy := newProxiedCodebaseDaemon(t)
	root := t.TempDir()
	writeCodebaseFile(t, root, "failure.go", goFile(goFunction("OldVersion", "oldfailuremarker")))
	codebaseDaemon.index(t, root)
	writeCodebaseFile(t, root, "failure.go", goFile(goFunction("NewVersion", "newfailuremarker")))
	observed, release := proxy.PausePublication(codebaseDaemon.harness.database, "lms_library_codebase", sandboxharness.BeforeStorePublication)
	t.Cleanup(release)
	jobID := codebaseDaemon.startSync(t, root)
	select {
	case <-observed:
	case <-codebaseDaemon.harness.context().Done():
		t.Fatal("the changed codebase did not request vector publication")
	}
	if err := codebaseDaemon.harness.milvus.DropCollection(codebaseDaemon.harness.context(), milvusclient.NewDropCollectionOption("lms_library_codebase")); err != nil {
		t.Fatalf("drop the isolated vector collection: %v", err)
	}
	release()
	if state := codebaseDaemon.waitTerminalJob(t, jobID); state != "failed" {
		t.Fatalf("real backend failure ended %s, want failed", state)
	}
	response, err := codebaseDaemon.client.GetJob(codebaseDaemon.harness.context(), &pb.GetJobRequest{JobId: jobID})
	if err != nil || response.GetJob().GetDisplayError() == "" {
		t.Fatalf("public job omitted the failure message: %v, %s", err, response.GetJob().GetDisplayError())
	}
	t.Logf("public GetJob returned failed after the real backend collection error; sanitized message=%s", response.GetJob().GetDisplayError())
}

// codebaseSearchLimit is larger than every occurrence count of the search
// test codebase, so one page returns the complete ranking.
const codebaseSearchLimit = 100

// TestLibraryCodebaseSearchReturnsCompleteLibraryPages indexes a codebase into
// the library store and searches it through the SearchCode RPC. A page larger
// than the codebase returns every indexed file in descending score order, each
// hit is an excerpt of its file, the extension filter keeps only the matching
// files, and a search of a subdirectory keeps only its files. SearchCode drops
// a hit that overlaps more than half of an earlier hit of the same file
// (semantic.DeduplicateChunks). A file can return fewer hits than it has
// occurrences.
func TestLibraryCodebaseSearchReturnsCompleteLibraryPages(t *testing.T) {
	codebaseDaemon := newLibraryCodebaseDaemon(t)
	t.Logf("embedding window start %s", time.Now().UTC().Format(time.RFC3339))
	defer func() { t.Logf("embedding window end %s", time.Now().UTC().Format(time.RFC3339)) }()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "sub"), 0o700); err != nil {
		t.Fatalf("create sub directory: %v", err)
	}
	files := map[string]string{
		"alpha.go":     goFile(goFunction("Alpha", "alphamarker"), goFunction("AlphaTwo", "alphatwomarker")),
		"sub/bravo.go": goFile(goFunction("Bravo", "bravomarker")),
		"notes.md":     "# Notes\n\nThe alphamarker notes describe the fixture.\n",
	}
	for name, content := range files {
		writeCodebaseFile(t, root, name, content)
	}
	codebaseDaemon.index(t, root)
	owners := codebaseDaemon.readCodebaseOwners(t)
	counts := map[string]int{}
	for name := range files {
		if len(owners[name]) == 0 {
			t.Fatalf("%s has no published occurrence", name)
		}
		counts[name] = len(owners[name])
	}

	search := func(path string, extensions []string) []*pb.SearchResult {
		t.Helper()
		response, err := codebaseDaemon.client.SearchCode(context.Background(), &pb.SearchCodeRequest{
			Path:            path,
			Query:           "alphamarker",
			Limit:           codebaseSearchLimit,
			ExtensionFilter: extensions,
			Client:          &pb.ClientInfo{Name: "library-codebase-live"},
		})
		if err != nil {
			t.Fatalf("search %s %v: %v", path, extensions, err)
		}
		results := response.GetResults()
		for index, result := range results {
			content, known := files[result.GetRelativePath()]
			if !known || !strings.Contains(content, result.GetContent()) {
				t.Fatalf("search %s %v hit %d from %q is not an excerpt of an indexed file", path, extensions, index, result.GetRelativePath())
			}
			if index > 0 && result.GetScore() > results[index-1].GetScore() {
				t.Fatalf("search %s %v hit %d scores %v above hit %d at %v", path, extensions, index, result.GetScore(), index-1, results[index-1].GetScore())
			}
		}
		return results
	}
	requireFiles := func(label string, results []*pb.SearchResult, want ...string) {
		t.Helper()
		paths := map[string]int{}
		for _, result := range results {
			paths[result.GetRelativePath()]++
		}
		limit := 0
		for _, name := range want {
			limit += counts[name]
		}
		if len(paths) != len(want) || len(results) > limit {
			t.Fatalf("%s returned %d hits from %v, want hits from exactly %v and at most %d", label, len(results), paths, want, limit)
		}
		for _, name := range want {
			if paths[name] == 0 {
				t.Fatalf("%s returned no hit from %s: %v", label, name, paths)
			}
		}
	}

	requireFiles("search of the root", search(root, nil), "alpha.go", "notes.md", "sub/bravo.go")
	requireFiles("extension filter .go", search(root, []string{".go"}), "alpha.go", "sub/bravo.go")
	requireFiles("search of sub", search(filepath.Join(root, "sub"), nil), "sub/bravo.go")
}

func TestLibraryCodebaseNamespacesShareVectorsAndRetainExactResults(t *testing.T) {
	harness := newCrossNamespaceHarness(t)
	daemon := newCodebaseLiveDaemonWithHarness(t, harness, config.CodebaseStoreLibrary)
	firstRoot, secondRoot := t.TempDir(), t.TempDir()
	firstFiles := writeCrossNamespaceFiles(t, firstRoot, "firstmarker")
	secondFiles := writeCrossNamespaceFiles(t, secondRoot, "secondmarker")
	daemon.index(t, firstRoot)
	firstBefore := readCrossNamespaceOwners(t, daemon, firstRoot)
	firstVectors := harness.backendVectorIDs("lms_library_codebase")
	if len(firstVectors) != 3 {
		t.Fatalf("first root published %d canonical vectors, want 3", len(firstVectors))
	}
	firstHits := requireCrossNamespaceQueries(t, daemon, firstRoot, firstFiles)
	daemon.index(t, secondRoot)
	firstAfter := readCrossNamespaceOwners(t, daemon, firstRoot)
	secondBefore := readCrossNamespaceOwners(t, daemon, secondRoot)
	requireCrossNamespaceOwnersEqual(t, firstBefore, firstAfter)
	for _, owners := range []map[string][]codebaseOccurrence{firstAfter, secondBefore} {
		if len(owners) != 3 {
			t.Fatalf("namespace published %d owners, want 3", len(owners))
		}
		for _, owner := range []string{"shared.go", "notes.md", "sub/unique.go"} {
			if len(owners[owner]) != 1 {
				t.Fatalf("namespace owner %s published %d rows, want 1", owner, len(owners[owner]))
			}
		}
	}
	for _, owner := range []string{"shared.go", "notes.md"} {
		if len(firstAfter[owner]) != 1 || len(secondBefore[owner]) != 1 || firstAfter[owner][0].vectorID != secondBefore[owner][0].vectorID {
			t.Fatalf("shared owner %s did not retain one canonical vector across namespaces", owner)
		}
	}
	secondVectors := harness.backendVectorIDs("lms_library_codebase")
	added := 0
	for _, id := range secondVectors {
		if _, found := slices.BinarySearch(firstVectors, id); !found {
			added++
		}
	}
	if len(secondVectors) != 4 || added != 1 || secondBefore["sub/unique.go"][0].vectorID == firstAfter["sub/unique.go"][0].vectorID {
		t.Fatalf("second root vector pool has %d IDs and %d new IDs, want 4 and 1", len(secondVectors), added)
	}
	secondHits := requireCrossNamespaceQueries(t, daemon, secondRoot, secondFiles)
	requireCrossNamespaceQueryOrder(t, firstHits, requireCrossNamespaceQueries(t, daemon, firstRoot, firstFiles))
	cleared, err := daemon.client.ClearIndex(t.Context(), &pb.ClearIndexRequest{Path: firstRoot})
	if err != nil || !cleared.GetCleared() {
		t.Fatalf("clear first root: cleared=%t error=%v", cleared.GetCleared(), err)
	}
	if owners := readCrossNamespaceOwners(t, daemon, firstRoot); len(owners) != 0 {
		t.Fatalf("cleared first namespace retained %d owners", len(owners))
	}
	requireCrossNamespaceOwnersEqual(t, secondBefore, readCrossNamespaceOwners(t, daemon, secondRoot))
	requireCrossNamespaceQueryOrder(t, secondHits, requireCrossNamespaceQueries(t, daemon, secondRoot, secondFiles))
	if actual := harness.backendVectorIDs("lms_library_codebase"); !slices.Equal(secondVectors, actual) {
		t.Fatal("clearing one namespace changed the canonical pool IDs")
	}
	t.Logf("two namespaces each published 3 owners; second root added 1 canonical input, shared 2 vectors; retained 4 vectors and exact remaining public results after ClearIndex")
}

func writeCrossNamespaceFiles(t *testing.T, root, marker string) map[string]string {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(root, "sub"), 0o700); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{
		"shared.go":     goFile(goFunction("Shared", "sharedmarker")),
		"sub/unique.go": goFile(goFunction("Unique", marker)),
		"notes.md":      "# Shared marker\n\nThe sharedmarker notes describe both source roots.\n",
	}
	for path, content := range files {
		writeCodebaseFile(t, root, path, content)
	}
	return files
}

func requireCrossNamespaceQueries(t *testing.T, daemon *libraryCodebaseDaemon, root string, files map[string]string) map[string][]string {
	t.Helper()
	expected := crossNamespaceSourceIdentities(t, root, files)
	results := make(map[string][]string)
	for _, query := range []struct {
		label      string
		path       string
		extensions []string
		owners     []string
	}{
		{label: "all", path: root, extensions: nil, owners: []string{"shared.go", "notes.md", "sub/unique.go"}},
		{label: "go", path: root, extensions: []string{".go"}, owners: []string{"shared.go", "sub/unique.go"}},
		{label: "sub", path: filepath.Join(root, "sub"), extensions: nil, owners: []string{"sub/unique.go"}},
	} {
		response, err := daemon.client.SearchCode(t.Context(), &pb.SearchCodeRequest{Path: query.path, Query: "sharedmarker", Limit: codebaseSearchLimit, ExtensionFilter: query.extensions})
		if err != nil {
			t.Fatalf("%s public search: %v", query.label, err)
		}
		actual := baselineHitIdentities(t, response.GetResults(), files, query.owners)
		wanted := make([]string, 0, len(query.owners))
		for _, owner := range query.owners {
			wanted = append(wanted, expected[owner])
		}
		actualSet, wantedSet := slices.Clone(actual), slices.Clone(wanted)
		slices.Sort(actualSet)
		slices.Sort(wantedSet)
		if !slices.Equal(actualSet, wantedSet) {
			t.Fatalf("%s public identities differ: actual=%v expected=%v", query.label, actualSet, wantedSet)
		}
		results[query.label] = actual
	}
	return results
}

func crossNamespaceSourceIdentities(t *testing.T, root string, files map[string]string) map[string]string {
	t.Helper()
	dispatcher := chunk.NewDispatcher()
	identities := make(map[string]string, len(files))
	for path, source := range files {
		projected, err := dispatcher.SplitFileWithType(t.Context(), filepath.Join(root, path), []byte(source), "ast")
		if err != nil || len(projected.Chunks) != 1 {
			t.Fatalf("source %s has %d chunks, want one nonoverlapping chunk: %v", path, len(projected.Chunks), err)
		}
		part := projected.Chunks[0]
		identities[path] = fmt.Sprintf("%s:%d:%d:%x", path, part.StartLine, part.EndLine, sha256.Sum256([]byte(part.Content)))
	}
	return identities
}

func requireCrossNamespaceQueryOrder(t *testing.T, before, after map[string][]string) {
	t.Helper()
	for label, expected := range before {
		if !slices.Equal(expected, after[label]) {
			t.Fatalf("%s ordered public identities changed: before=%v after=%v", label, expected, after[label])
		}
	}
}

func requireCrossNamespaceOwnersEqual(t *testing.T, before, after map[string][]codebaseOccurrence) {
	t.Helper()
	if len(before) != len(after) {
		t.Fatalf("namespace owner count changed from %d to %d", len(before), len(after))
	}
	for owner, expected := range before {
		if !slices.Equal(expected, after[owner]) {
			t.Fatalf("namespace owner %s occurrence state changed", owner)
		}
	}
}

func readCrossNamespaceOwners(t *testing.T, daemon *libraryCodebaseDaemon, root string) map[string][]codebaseOccurrence {
	t.Helper()
	canonical, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256([]byte(canonical))
	namespace := fmt.Sprintf("code_%x", digest[:8])
	catalog, err := sql.Open("sqlite3", "file:"+filepath.Join(daemon.config.StateRoot, "library", "codebase", "catalog.sqlite")+"?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := catalog.Close(); err != nil {
			t.Error(err)
		}
	})
	rows, err := catalog.QueryContext(t.Context(), codebaseNamespaceRowsSQL, namespace)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := rows.Close(); err != nil {
			t.Error(err)
		}
	}()
	owners := make(map[string][]codebaseOccurrence)
	for rows.Next() {
		var owner string
		var occurrence codebaseOccurrence
		if err := rows.Scan(&owner, &occurrence.rowKey, &occurrence.vectorID, &occurrence.source, &occurrence.occurrenceHash, &occurrence.generationOrder); err != nil {
			t.Fatal(err)
		}
		owners[owner] = append(owners[owner], occurrence)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	return owners
}

func newCrossNamespaceHarness(t *testing.T) *libraryHarness {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), libraryLiveTimeout)
	t.Cleanup(cancel)
	environment := resolveLibraryEnvironment(t)
	if environment.MilvusAddress != "localhost:39530" {
		t.Fatalf("cross-namespace fixture requires localhost:39530, got %s", environment.MilvusAddress)
	}
	database := libraryLiveDatabasePrefix + randomHex(t, 16)
	admin, err := milvusclient.New(ctx, &milvusclient.ClientConfig{Address: environment.MilvusAddress})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(t.Context()), 30*time.Second)
		defer cancel()
		if err := admin.Close(cleanupCtx); err != nil {
			t.Error(err)
		}
	})
	existing, err := admin.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil || slices.Contains(existing, database) {
		t.Fatalf("verify fresh database %s absence: %v", database, err)
	}
	registerCrossNamespaceDatabase(t, database, environment.MilvusAddress)
	t.Cleanup(func() { dropCrossNamespaceDatabase(t, admin, database, environment.MilvusAddress) })
	if err := admin.CreateDatabase(ctx, milvusclient.NewCreateDatabaseOption(database)); err != nil {
		t.Fatal(err)
	}
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{Address: environment.MilvusAddress, DBName: database})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(t.Context()), 30*time.Second)
		defer cancel()
		if err := client.Close(cleanupCtx); err != nil {
			t.Error(err)
		}
	})
	return &libraryHarness{t: t, context: func() context.Context { return ctx }, environment: environment, database: database, admin: admin, milvus: client, root: t.TempDir(), embedder: newLiveEmbedder(t, ctx, environment)}
}

func registerCrossNamespaceDatabase(t *testing.T, database, address string) {
	t.Helper()
	path := os.Getenv("LMS_L4_DATABASE_REGISTRY")
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || filepath.Base(path) != "database-registration.jsonl" || filepath.Dir(filepath.Dir(path)) != "/private/tmp" || !strings.HasPrefix(filepath.Dir(path), "/private/tmp/lms-cross-namespace-live-") {
		t.Fatalf("invalid private cross-namespace registry path %s", path)
	}
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Mode().Perm() != 0o600 {
		t.Fatalf("registry must be an existing private regular file: %v", err)
	}
	directory, err := os.Lstat(filepath.Dir(path))
	if err != nil || !directory.IsDir() || directory.Mode().Perm() != 0o700 {
		t.Fatalf("registry root must be a private directory: %v", err)
	}
	file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := file.Close(); err != nil {
			t.Error(err)
		}
	}()
	opened, err := file.Stat()
	if err != nil || !os.SameFile(info, opened) {
		t.Fatalf("registry file changed before registration: %v", err)
	}
	entry := struct {
		Database        string `json:"database"`
		Address         string `json:"address"`
		AbsenceVerified bool   `json:"absence_verified"`
	}{Database: database, Address: address, AbsenceVerified: true}
	if err := json.NewEncoder(file).Encode(entry); err != nil {
		t.Fatal(err)
	}
	if err := file.Sync(); err != nil {
		t.Fatal(err)
	}
	t.Logf("registered absent isolated database %s at %s before creation", database, address)
}

func dropCrossNamespaceDatabase(t *testing.T, admin *milvusclient.Client, database, address string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.WithoutCancel(t.Context()), 30*time.Second)
	defer cancel()
	existing, err := admin.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil {
		t.Error(err)
		return
	}
	if !slices.Contains(existing, database) {
		return
	}
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{Address: address, DBName: database})
	if err != nil {
		t.Error(err)
		return
	}
	defer func() {
		if err := client.Close(ctx); err != nil {
			t.Error(err)
		}
	}()
	collections, err := client.ListCollections(ctx, milvusclient.NewListCollectionOption())
	if err != nil {
		t.Error(err)
		return
	}
	for _, collection := range collections {
		if err := client.DropCollection(ctx, milvusclient.NewDropCollectionOption(collection)); err != nil {
			t.Error(err)
			return
		}
	}
	if err := admin.DropDatabase(ctx, milvusclient.NewDropDatabaseOption(database)); err != nil {
		t.Error(err)
		return
	}
	existing, err = admin.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil || slices.Contains(existing, database) {
		t.Errorf("exact database %s remains after cleanup: %v", database, err)
		return
	}
	t.Logf("verified isolated database %s absent after cleanup", database)
}
