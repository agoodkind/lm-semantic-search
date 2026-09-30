//go:build live

package live

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/sandbox"
	"goodkind.io/lm-semantic-search/internal/store"
)

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

// cancelAfterFirstStagedBatch waits until the catalog has staged rows, which
// the library saves only after it writes and strongly verifies their vectors,
// and then cancels the job. It returns the staged row count at cancellation
// and the terminal job state.
func (codebaseDaemon *libraryCodebaseDaemon) cancelAfterFirstStagedBatch(t *testing.T, jobID string) (int, string) {
	t.Helper()
	deadline := time.Now().Add(codebaseJobTimeout)
	staged := 0
	for staged == 0 {
		if time.Now().After(deadline) {
			t.Fatalf("job %s staged no rows within %s", jobID, codebaseJobTimeout)
		}
		time.Sleep(codebaseJobPoll)
		staged = codebaseDaemon.stagedRowCount(t)
	}
	if _, err := codebaseDaemon.client.CancelJob(context.Background(), &pb.CancelJobRequest{
		JobId:  jobID,
		Client: &pb.ClientInfo{Name: "library-codebase-live"},
	}); err != nil {
		t.Fatalf("cancel job %s: %v", jobID, err)
	}
	for time.Now().Before(deadline) {
		response, err := codebaseDaemon.client.GetJob(context.Background(), &pb.GetJobRequest{JobId: jobID})
		if err != nil {
			t.Fatalf("get job %s: %v", jobID, err)
		}
		switch state := response.GetJob().GetState(); state {
		case "completed", "failed", "cancelled":
			return staged, state
		}
		time.Sleep(codebaseJobPoll)
	}
	t.Fatalf("job %s did not end after cancellation", jobID)
	return staged, ""
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
	codebaseDaemon := newLibraryCodebaseDaemon(t)
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
	staged, state := codebaseDaemon.cancelAfterFirstStagedBatch(t, codebaseDaemon.startSync(t, root))
	if state != "cancelled" {
		t.Fatalf("interrupted sync ended %s, want cancelled", state)
	}
	if staged >= interruptedFunctionCount {
		t.Fatalf("the sync staged %d rows before cancellation, want fewer than %d", staged, interruptedFunctionCount)
	}
	interrupted := codebaseDaemon.readCodebaseOwners(t)
	if !slices.Equal(before["large.go"], interrupted["large.go"]) {
		t.Fatal("an interrupted generation changed the published occurrences of large.go")
	}
	assertExcerpts(t, interrupted, "large.go", oldContent)

	codebaseDaemon.sync(t, root)
	after := codebaseDaemon.readCodebaseOwners(t)
	if got := len(after["large.go"]); got != interruptedFunctionCount {
		t.Fatalf("large.go has %d occurrences after the resumed sync, want %d", got, interruptedFunctionCount)
	}
	assertExcerpts(t, after, "large.go", newContent)
	if remaining := codebaseDaemon.stagedRowCount(t); remaining != 0 {
		t.Fatalf("catalog keeps %d staged rows after the resumed sync committed", remaining)
	}
	t.Logf("cancelled after %d staged rows; the resumed sync committed %d occurrences", staged, len(after["large.go"]))
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
