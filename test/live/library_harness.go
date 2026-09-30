//go:build live

package live

import (
	"context"
	cryptorand "crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/embedding"
	"goodkind.io/lm-semantic-search/library/milvus"
)

const (
	// libraryLiveDatabasePrefix starts the name of every database a library
	// live test creates. Cleanup drops only the database the same test created.
	libraryLiveDatabasePrefix = "lms_lib_live_"
	// libraryLiveDimension is the vector dimension of the production embedding
	// model the library live tests call.
	libraryLiveDimension = 4096
	// libraryLiveModelTokenLimit is the input token limit of the production
	// embedding model.
	libraryLiveModelTokenLimit = 4096
	// libraryLiveRevision is the model revision the test descriptors record.
	libraryLiveRevision = "live-test"
	// libraryLiveChildEnv makes the test binary run one library operation as a
	// separate process instead of the tests.
	libraryLiveChildEnv = "LMS_LIBRARY_LIVE_CHILD"
	// libraryLiveTimeout bounds one live test.
	libraryLiveTimeout = 10 * time.Minute
	// libraryEmbeddingRequestTimeout bounds one embedding request.
	libraryEmbeddingRequestTimeout = 2 * time.Minute
)

// libraryProtectedDatabases are the databases a library live test never reads,
// writes, or drops.
var libraryProtectedDatabases = []string{"default", "lms_clean_sample"}

// libraryEnvironment is the resolved real Milvus endpoint and embedding
// endpoint for library live tests.
type libraryEnvironment struct {
	MilvusAddress  string `json:"milvus_address"`
	EmbeddingURL   string `json:"embedding_url"`
	EmbeddingModel string `json:"embedding_model"`
	APIKey         string `json:"-"`
}

// libraryHarness owns one isolated Milvus database and one temporary
// directory for catalogs.
type libraryHarness struct {
	t               *testing.T
	context         func() context.Context
	environment     libraryEnvironment
	database        string
	admin           *milvusclient.Client
	milvus          *milvusclient.Client
	root            string
	embedder        library.Embedder
	createRequested bool
}

// newLibraryHarness creates a new isolated Milvus database and a real
// embedding adapter. A missing endpoint fails the test.
func newLibraryHarness(t *testing.T) *libraryHarness {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), libraryLiveTimeout)
	t.Cleanup(cancel)
	environment := resolveLibraryEnvironment(t)
	database := libraryLiveDatabasePrefix + randomHex(t, 16)
	if path := os.Getenv("LMS_LIBRARY_LIVE_DATABASE_INTENT"); path != "" {
		var err error
		var registration *os.File
		database, registration, err = readLibraryDatabaseIntent(t.Name(), path, environment.MilvusAddress)
		if err != nil {
			t.Fatalf("reject library database intent: %v", err)
		}
		// Cleanup callbacks run in reverse order, after database cleanup.
		t.Cleanup(func() {
			if err := registration.Close(); err != nil {
				t.Errorf("close database registration lock: %v", err)
			}
		})
	}

	admin, err := milvusclient.New(ctx, &milvusclient.ClientConfig{Address: environment.MilvusAddress})
	if err != nil {
		t.Fatalf("connect to Milvus at %s: %v", environment.MilvusAddress, err)
	}
	harness := &libraryHarness{
		t: t, context: func() context.Context { return ctx }, environment: environment,
		database: database, admin: admin, root: t.TempDir(),
	}
	t.Cleanup(harness.dropDatabase)
	existing, err := admin.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil {
		t.Fatalf("list Milvus databases: %v", err)
	}
	if slices.Contains(existing, database) {
		t.Fatalf("Milvus database %s already exists", database)
	}
	harness.createRequested = true
	if err := admin.CreateDatabase(ctx, milvusclient.NewCreateDatabaseOption(database)); err != nil {
		t.Fatalf("create Milvus database %s: %v", database, err)
	}
	t.Logf("created Milvus database %s", database)
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{Address: environment.MilvusAddress, DBName: database})
	if err != nil {
		t.Fatalf("connect to Milvus database %s: %v", database, err)
	}
	harness.milvus = client
	harness.embedder = newLiveEmbedder(t, ctx, environment)
	return harness
}

type libraryDatabaseIntent struct {
	SchemaVersion   int       `json:"schema_version"`
	Database        string    `json:"database"`
	MilvusAddress   string    `json:"milvus_address"`
	RunRoot         string    `json:"run_root"`
	RunID           string    `json:"run_id"`
	RegisteredAt    time.Time `json:"registered_at"`
	AbsenceVerified bool      `json:"absence_verified"`
}

func readLibraryDatabaseIntent(testName, path, address string) (database string, registration *os.File, resultErr error) {
	defer func() {
		if resultErr != nil {
			if registration != nil {
				resultErr = errors.Join(resultErr, registration.Close())
				registration = nil
			}
			slog.Warn("library database intent rejected", "err", resultErr)
		}
	}()
	if testName != "TestLibrarySearchCompletePagesMatchTheExhaustiveOracle" {
		return "", nil, errors.New("database intent requires the exact existing exhaustive oracle test")
	}
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || filepath.Base(path) != "database-intent.json" {
		return "", nil, errors.New("database intent requires an absolute canonical registration path")
	}
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", nil, fmt.Errorf("resolve database intent: %w", err)
	}
	if resolved != path {
		return "", nil, errors.New("database intent path contains a symlink")
	}
	registrationRoot, err := os.OpenRoot("/private/tmp")
	if err != nil {
		return "", nil, fmt.Errorf("open isolated registration root: %w", err)
	}
	defer func() { resultErr = errors.Join(resultErr, registrationRoot.Close()) }()
	file, err := registrationRoot.Open(strings.TrimPrefix(path, "/private/tmp/"))
	if err != nil {
		return "", nil, fmt.Errorf("open database intent: %w", err)
	}
	defer func() {
		if resultErr != nil {
			resultErr = errors.Join(resultErr, file.Close())
		}
	}()
	info, err := file.Stat()
	if err != nil {
		return "", nil, fmt.Errorf("inspect database intent: %w", err)
	}
	owner, ok := info.Sys().(*syscall.Stat_t)
	if !info.Mode().IsRegular() || info.Mode().Perm() != 0o600 || !ok || strconv.FormatUint(uint64(owner.Uid), 10) != strconv.Itoa(os.Geteuid()) || info.Size() > 4096 {
		return "", nil, errors.New("database intent must be a private owned regular file of at most 4096 bytes")
	}
	if err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		return "", nil, fmt.Errorf("claim exclusive database intent: %w", err)
	}
	pathInfo, err := registrationRoot.Stat(strings.TrimPrefix(path, "/private/tmp/"))
	if err != nil {
		return "", nil, fmt.Errorf("inspect locked database intent path: %w", err)
	}
	if !os.SameFile(info, pathInfo) {
		return "", nil, errors.New("database intent path differs from the locked file")
	}
	database, err = decodeLibraryDatabaseIntent(file, path, address)
	if err != nil {
		return "", nil, err
	}
	return database, file, nil
}

func decodeLibraryDatabaseIntent(file *os.File, path, address string) (database string, resultErr error) {
	defer func() {
		if resultErr != nil {
			slog.Warn("library database registration rejected", "err", resultErr)
		}
	}()
	var intent libraryDatabaseIntent
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&intent); err != nil {
		return "", fmt.Errorf("decode database intent: %w", err)
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return "", errors.New("database intent contains trailing data")
	}
	root := filepath.Join("/private/tmp", "lms-oracle-rerun-"+intent.RunID)
	if !validLibraryRunID(intent.RunID) || intent.RunRoot != root || filepath.Dir(path) != root || intent.Database != libraryLiveDatabasePrefix+intent.RunID || len(intent.Database) > 128 {
		return "", errors.New("database intent does not bind its exact isolated root and database")
	}
	if intent.SchemaVersion != 1 || !intent.AbsenceVerified || intent.RegisteredAt.IsZero() || intent.RegisteredAt.After(clock.Now()) || address != "localhost:39630" || intent.MilvusAddress != address {
		return "", errors.New("database intent requires prior absence, timestamp and exact isolated native endpoint")
	}
	return intent.Database, nil
}

func validLibraryRunID(runID string) bool {
	if runID == "" {
		return false
	}
	for _, character := range runID {
		if character != '_' && (character < 'a' || character > 'z') && (character < '0' || character > '9') {
			return false
		}
	}
	return true
}

// resolveLibraryEnvironment reads the Milvus and embedding endpoints from the
// daemon configuration, which loads ~/.context/.env.
func resolveLibraryEnvironment(t *testing.T) libraryEnvironment {
	t.Helper()
	daemonConfig, err := config.Default()
	if err != nil {
		t.Fatalf("load configuration: %v", err)
	}
	environment := libraryEnvironment{
		MilvusAddress:  daemonConfig.MilvusAddress,
		EmbeddingURL:   daemonConfig.OpenAIBaseURL,
		EmbeddingModel: daemonConfig.EmbeddingModel,
		APIKey:         daemonConfig.OpenAIAPIKey,
	}
	if environment.MilvusAddress == "" || environment.EmbeddingURL == "" || environment.EmbeddingModel == "" {
		t.Fatalf("library live tests need a Milvus address, an embedding base URL, and an embedding model; got %+v", environment)
	}
	return environment
}

func newLiveEmbedder(t *testing.T, ctx context.Context, environment libraryEnvironment) library.Embedder {
	t.Helper()
	embedder, err := embedding.NewOpenAI(ctx, embedding.OpenAIConfig{
		BaseURL:        environment.EmbeddingURL,
		APIKey:         environment.APIKey,
		Model:          environment.EmbeddingModel,
		Dimension:      libraryLiveDimension,
		RequestTimeout: libraryEmbeddingRequestTimeout,
		MaxAttempts:    0,
		BackoffBase:    0,
	})
	if err != nil {
		t.Fatalf("create embedding adapter: %v", err)
	}
	return embedder
}

// dropDatabase drops every collection in the test database and then the
// database. It refuses any database without the library live prefix.
func (harness *libraryHarness) dropDatabase() {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	defer func() {
		var closeErr error
		if harness.milvus != nil {
			closeErr = harness.milvus.Close(ctx)
		}
		if err := errors.Join(closeErr, harness.admin.Close(ctx)); err != nil {
			harness.t.Errorf("close Milvus clients: %v", err)
		}
	}()
	if !strings.HasPrefix(harness.database, libraryLiveDatabasePrefix) || slices.Contains(libraryProtectedDatabases, harness.database) {
		harness.t.Errorf("refusing to drop database %s", harness.database)
		return
	}
	if !harness.createRequested {
		return
	}
	databases, err := harness.admin.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil {
		harness.t.Errorf("inspect database %s for cleanup: %v", harness.database, err)
		return
	}
	if !slices.Contains(databases, harness.database) {
		return
	}
	client := harness.milvus
	if client == nil {
		client, err = milvusclient.New(ctx, &milvusclient.ClientConfig{Address: harness.environment.MilvusAddress, DBName: harness.database})
		if err != nil {
			harness.t.Errorf("open partial-startup cleanup client: %v", err)
			return
		}
		harness.milvus = client
	}
	collections, err := client.ListCollections(ctx, milvusclient.NewListCollectionOption())
	if err != nil {
		harness.t.Errorf("list collections in %s: %v", harness.database, err)
		return
	}
	for _, collection := range collections {
		if err := client.DropCollection(ctx, milvusclient.NewDropCollectionOption(collection)); err != nil {
			harness.t.Errorf("drop collection %s.%s: %v", harness.database, collection, err)
		}
	}
	if err := harness.admin.DropDatabase(ctx, milvusclient.NewDropDatabaseOption(harness.database)); err != nil {
		harness.t.Errorf("drop database %s: %v", harness.database, err)
		return
	}
	harness.t.Logf("dropped Milvus database %s", harness.database)
	databases, err = harness.admin.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil || slices.Contains(databases, harness.database) {
		harness.t.Errorf("verify database %s absence after cleanup: present=%t err=%v", harness.database, slices.Contains(databases, harness.database), err)
	}
}

// descriptor returns a store descriptor for a catalog under the harness root.
func (harness *libraryHarness) descriptor(name string) library.StoreDescriptor {
	directory := filepath.Join(harness.root, name)
	return library.StoreDescriptor{
		CatalogPath:       filepath.Join(directory, "catalog.sqlite"),
		LockPath:          filepath.Join(directory, "catalog.lock"),
		PoolID:            name,
		EmbeddingModel:    harness.environment.EmbeddingModel,
		EmbeddingRevision: libraryLiveRevision,
		Dimension:         libraryLiveDimension,
		Normalization:     "none",
	}
}

func (harness *libraryHarness) vectorStore(collection string) *milvus.Store {
	harness.t.Helper()
	store, err := milvus.New(harness.milvus, milvus.Config{Database: harness.database, Collection: collection})
	if err != nil {
		harness.t.Fatalf("create Milvus adapter: %v", err)
	}
	return store
}

// open opens a library over descriptor and vectors and closes it at cleanup.
func (harness *libraryHarness) open(descriptor library.StoreDescriptor, vectors library.VectorStore) *library.Library {
	harness.t.Helper()
	opened, err := library.Open(harness.context(), library.Config{Store: descriptor, Vectors: vectors, Embedder: harness.embedder})
	if err != nil {
		harness.t.Fatalf("open library %s: %v", descriptor.CatalogPath, err)
	}
	harness.t.Cleanup(func() {
		if err := opened.Close(); err != nil {
			harness.t.Errorf("close library: %v", err)
		}
	})
	return opened
}

// backendVectorIDs reads every vector ID in collection with a strong read.
func (harness *libraryHarness) backendVectorIDs(collection string) []string {
	harness.t.Helper()
	result, err := harness.milvus.Query(
		harness.context(),
		milvusclient.NewQueryOption(collection).
			WithFilter(`vector_id != ""`).
			WithOutputFields("vector_id").
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		harness.t.Fatalf("read vector IDs from %s: %v", collection, err)
	}
	ids := make([]string, 0, result.ResultCount)
	for row := range result.ResultCount {
		id, err := result.GetColumn("vector_id").GetAsString(row)
		if err != nil {
			harness.t.Fatalf("decode vector ID: %v", err)
		}
		ids = append(ids, id)
	}
	slices.Sort(ids)
	return ids
}

// catalogRows is the content of a catalog that the test reads directly with
// its own SQLite connection.
type catalogRows struct {
	occurrences      map[string]string
	unverifiedLinked int
	outboxEntries    int
	vectorStates     map[string]string
}

func (harness *libraryHarness) readCatalog(descriptor library.StoreDescriptor) catalogRows {
	harness.t.Helper()
	database, err := sql.Open("sqlite3", "file:"+descriptor.CatalogPath+"?mode=ro")
	if err != nil {
		harness.t.Fatalf("open catalog %s: %v", descriptor.CatalogPath, err)
	}
	defer func() {
		if err := database.Close(); err != nil {
			harness.t.Errorf("close catalog: %v", err)
		}
	}()
	rows := catalogRows{occurrences: map[string]string{}, unverifiedLinked: 0, outboxEntries: 0, vectorStates: map[string]string{}}
	harness.scanPairs(database, `SELECT namespace || '/' || owner_id || '/' || row_key, vector_id FROM occurrences`, rows.occurrences)
	harness.scanPairs(database, `SELECT vector_id, state FROM vectors`, rows.vectorStates)
	if err := database.QueryRowContext(
		harness.context(),
		`SELECT COUNT(*) FROM occurrences JOIN vectors ON vectors.vector_id = occurrences.vector_id WHERE vectors.state != 'verified'`,
	).Scan(&rows.unverifiedLinked); err != nil {
		harness.t.Fatalf("count unverified occurrence vectors: %v", err)
	}
	if err := database.QueryRowContext(harness.context(), `SELECT COUNT(*) FROM vector_outbox`).Scan(&rows.outboxEntries); err != nil {
		harness.t.Fatalf("count outbox entries: %v", err)
	}
	return rows
}

// scalarQueries read one row's typed scalar columns from a scalar table.
var scalarQueries = map[string]string{
	"occurrence_scalars": `SELECT column_name, type, string_value, int64_value, bool_value, is_null FROM occurrence_scalars WHERE row_key = ?`,
	"effective_scalars":  `SELECT column_name, type, string_value, int64_value, bool_value, is_null FROM effective_scalars WHERE row_key = ?`,
}

// readScalars returns the scalar columns of rowKey in table as
// "string=<value>", "int64=<value>", "bool=<0 or 1>", or "null".
func (harness *libraryHarness) readScalars(descriptor library.StoreDescriptor, table string, rowKey string) map[string]string {
	harness.t.Helper()
	query, known := scalarQueries[table]
	if !known {
		harness.t.Fatalf("no scalar query for table %s", table)
	}
	database, err := sql.Open("sqlite3", "file:"+descriptor.CatalogPath+"?mode=ro")
	if err != nil {
		harness.t.Fatalf("open catalog %s: %v", descriptor.CatalogPath, err)
	}
	defer func() { _ = database.Close() }()
	rows, err := database.QueryContext(harness.context(), query, rowKey)
	if err != nil {
		harness.t.Fatalf("read %s: %v", table, err)
	}
	defer func() { _ = rows.Close() }()
	values := map[string]string{}
	for rows.Next() {
		var name string
		var scalarType library.ScalarType
		var stringValue sql.NullString
		var int64Value sql.NullInt64
		var boolValue sql.NullBool
		var isNull bool
		if err := rows.Scan(&name, &scalarType, &stringValue, &int64Value, &boolValue, &isNull); err != nil {
			harness.t.Fatalf("scan %s: %v", table, err)
		}
		switch {
		case isNull:
			values[name] = "null"
		case scalarType == library.String:
			values[name] = "string=" + stringValue.String
		case scalarType == library.Int64:
			values[name] = fmt.Sprintf("int64=%d", int64Value.Int64)
		case boolValue.Bool:
			values[name] = "bool=1"
		default:
			values[name] = "bool=0"
		}
	}
	if err := rows.Err(); err != nil {
		harness.t.Fatalf("read %s: %v", table, err)
	}
	return values
}

func (harness *libraryHarness) scanPairs(database *sql.DB, query string, into map[string]string) {
	harness.t.Helper()
	result, err := database.QueryContext(harness.context(), query)
	if err != nil {
		harness.t.Fatalf("query catalog: %v", err)
	}
	defer func() {
		if err := result.Close(); err != nil {
			harness.t.Errorf("close catalog rows: %v", err)
		}
	}()
	for result.Next() {
		var key, value string
		if err := result.Scan(&key, &value); err != nil {
			harness.t.Fatalf("scan catalog row: %v", err)
		}
		into[key] = value
	}
	if err := result.Err(); err != nil {
		harness.t.Fatalf("read catalog rows: %v", err)
	}
}

func randomHex(t *testing.T, byteCount int) string {
	t.Helper()
	random := make([]byte, byteCount)
	if _, err := cryptorand.Read(random); err != nil {
		t.Fatalf("read random bytes: %v", err)
	}
	return hex.EncodeToString(random)
}

// libraryChildRequest is one library operation that a child process runs.
type libraryChildRequest struct {
	Action      string                  `json:"action"`
	Environment libraryEnvironment      `json:"environment"`
	Database    string                  `json:"database"`
	Collection  string                  `json:"collection"`
	Descriptor  library.StoreDescriptor `json:"descriptor"`
	Namespace   library.NamespaceSpec   `json:"namespace"`
	Batches     []library.Batch         `json:"batches"`
	CrashPoint  string                  `json:"crash_point"`
	StartAt     time.Time               `json:"start_at"`
}

// Child actions and crash points.
const (
	childActionOpen  = "open"
	childActionApply = "apply"

	crashBeforePut    = "before_put"
	crashAfterPut     = "after_put"
	crashAfterVerify  = "after_verify"
	crashAfterPublish = "after_publish"
)

// childExitStoreMismatch is the exit status of a child that received
// ErrStoreMismatch.
const childExitStoreMismatch = 3

// startLibraryChild starts one child process for request. The child reads
// the embedding credential from the same configuration as the parent.
func startLibraryChild(t *testing.T, request libraryChildRequest) *exec.Cmd {
	t.Helper()
	request.Environment.APIKey = ""
	encoded, err := MarshalLibraryChild(request)
	if err != nil {
		t.Fatalf("encode child request: %v", err)
	}
	slog.Debug("start library acceptance child", "action", request.Action)
	executable, err := os.Executable()
	if err != nil {
		t.Fatalf("resolve test executable: %v", err)
	}
	command := exec.CommandContext(t.Context(), executable, "-test.run=^$")
	command.Env = append(os.Environ(), libraryLiveChildEnv+"="+string(encoded))
	command.Stdout = os.Stderr
	command.Stderr = os.Stderr
	if err := command.Start(); err != nil {
		t.Fatalf("start child process: %v", err)
	}
	return command
}

// waitLibraryChild waits for command and returns its exit status. A child
// killed by a signal returns -1 with the signal.
func waitLibraryChild(t *testing.T, command *exec.Cmd) (int, syscall.Signal) {
	t.Helper()
	err := command.Wait()
	if err == nil {
		return 0, 0
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("wait for child process: %v", err)
	}
	status, ok := exitErr.Sys().(syscall.WaitStatus)
	if ok && status.Signaled() {
		return -1, status.Signal()
	}
	return exitErr.ExitCode(), 0
}

// runLibraryChild runs one child request and returns the process exit status.
func runLibraryChild(encoded string) int {
	request, err := UnmarshalLibraryChild([]byte(encoded))
	if err != nil {
		fmt.Fprintf(os.Stderr, "decode child request: %v\n", err)
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), libraryLiveTimeout)
	defer cancel()
	daemonConfig, err := config.Default()
	if err != nil {
		fmt.Fprintf(os.Stderr, "load child configuration: %v\n", err)
		return 2
	}
	request.Environment.APIKey = daemonConfig.OpenAIAPIKey
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{Address: request.Environment.MilvusAddress, DBName: request.Database})
	if err != nil {
		fmt.Fprintf(os.Stderr, "connect child to Milvus: %v\n", err)
		return 2
	}
	defer func() {
		_ = client.Close(context.Background())
	}()
	store, err := milvus.New(client, milvus.Config{Database: request.Database, Collection: request.Collection})
	if err != nil {
		fmt.Fprintf(os.Stderr, "create child Milvus adapter: %v\n", err)
		return 2
	}
	embedder, err := embedding.NewOpenAI(ctx, embedding.OpenAIConfig{
		BaseURL:        request.Environment.EmbeddingURL,
		APIKey:         request.Environment.APIKey,
		Model:          request.Environment.EmbeddingModel,
		Dimension:      libraryLiveDimension,
		RequestTimeout: libraryEmbeddingRequestTimeout,
		MaxAttempts:    0,
		BackoffBase:    0,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "create child embedder: %v\n", err)
		return 2
	}
	if wait := clock.Until(request.StartAt); wait > 0 {
		timer := time.NewTimer(wait)
		defer timer.Stop()
		select {
		case <-timer.C:
		case <-ctx.Done():
			return 2
		}
	}
	return runChildAction(ctx, request, &crashingVectorStore{VectorStore: store, crashPoint: request.CrashPoint, putDone: false}, embedder)
}

func runChildAction(ctx context.Context, request libraryChildRequest, vectors library.VectorStore, embedder library.Embedder) int {
	opened, err := library.Open(ctx, library.Config{Store: request.Descriptor, Vectors: vectors, Embedder: embedder})
	if errors.Is(err, library.ErrStoreMismatch) {
		fmt.Fprintf(os.Stderr, "child open: %v\n", err)
		return childExitStoreMismatch
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "child open: %v\n", err)
		return 1
	}
	defer func() {
		_ = opened.Close()
	}()
	if request.Action == childActionOpen {
		return 0
	}
	if err := opened.RegisterNamespace(ctx, request.Namespace); err != nil {
		fmt.Fprintf(os.Stderr, "child register namespace: %v\n", err)
		return 1
	}
	for _, batch := range request.Batches {
		if _, err := opened.Apply(ctx, batch); err != nil {
			fmt.Fprintf(os.Stderr, "child apply owner %s: %v\n", batch.OwnerID, err)
			return 1
		}
	}
	if request.CrashPoint == crashAfterPublish {
		killSelf()
	}
	return 0
}

// crashingVectorStore kills its own process at one write boundary. It passes
// every call before that boundary to the real adapter.
type crashingVectorStore struct {
	library.VectorStore
	crashPoint string
	putDone    bool
}

func (store *crashingVectorStore) PutCanonical(ctx context.Context, record library.VectorRecord) error {
	if store.crashPoint == crashBeforePut {
		killSelf()
	}
	err := store.VectorStore.PutCanonical(ctx, record)
	store.putDone = err == nil
	if store.crashPoint == crashAfterPut && store.putDone {
		killSelf()
	}
	if err != nil {
		slog.Warn("live test dependency failed", "err", err)
		return fmt.Errorf("put canonical vector before child interruption: %w", err)
	}
	return nil
}

func (store *crashingVectorStore) VerifyStrong(ctx context.Context, identities []library.VectorIdentity) error {
	err := store.VectorStore.VerifyStrong(ctx, identities)
	if store.crashPoint == crashAfterVerify && store.putDone && err == nil {
		killSelf()
	}
	if err != nil {
		slog.Warn("live test dependency failed", "err", err)
		return fmt.Errorf("verify vector before child interruption: %w", err)
	}
	return nil
}

// killSelf ends the process with SIGKILL, which runs no deferred function and
// no cleanup.
func killSelf() {
	_ = syscall.Kill(os.Getpid(), syscall.SIGKILL)
	select {}
}
