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
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
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
	t           *testing.T
	ctx         context.Context
	environment libraryEnvironment
	database    string
	admin       *milvusclient.Client
	milvus      *milvusclient.Client
	root        string
	embedder    library.Embedder
}

// newLibraryHarness creates a new isolated Milvus database and a real
// embedding adapter. A missing endpoint fails the test.
func newLibraryHarness(t *testing.T) *libraryHarness {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), libraryLiveTimeout)
	t.Cleanup(cancel)
	environment := resolveLibraryEnvironment(t)
	database := libraryLiveDatabasePrefix + randomHex(t, 16)

	admin, err := milvusclient.New(ctx, &milvusclient.ClientConfig{Address: environment.MilvusAddress})
	if err != nil {
		t.Fatalf("connect to Milvus at %s: %v", environment.MilvusAddress, err)
	}
	existing, err := admin.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil {
		t.Fatalf("list Milvus databases: %v", err)
	}
	if slices.Contains(existing, database) {
		t.Fatalf("Milvus database %s already exists", database)
	}
	if err := admin.CreateDatabase(ctx, milvusclient.NewCreateDatabaseOption(database)); err != nil {
		t.Fatalf("create Milvus database %s: %v", database, err)
	}
	t.Logf("created Milvus database %s", database)
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{Address: environment.MilvusAddress, DBName: database})
	if err != nil {
		t.Fatalf("connect to Milvus database %s: %v", database, err)
	}
	harness := &libraryHarness{
		t:           t,
		ctx:         ctx,
		environment: environment,
		database:    database,
		admin:       admin,
		milvus:      client,
		root:        t.TempDir(),
		embedder:    newLiveEmbedder(t, ctx, environment),
	}
	t.Cleanup(harness.dropDatabase)
	return harness
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
	if !strings.HasPrefix(harness.database, libraryLiveDatabasePrefix) || slices.Contains(libraryProtectedDatabases, harness.database) {
		harness.t.Errorf("refusing to drop database %s", harness.database)
		return
	}
	collections, err := harness.milvus.ListCollections(ctx, milvusclient.NewListCollectionOption())
	if err != nil {
		harness.t.Errorf("list collections in %s: %v", harness.database, err)
	}
	for _, collection := range collections {
		if err := harness.milvus.DropCollection(ctx, milvusclient.NewDropCollectionOption(collection)); err != nil {
			harness.t.Errorf("drop collection %s.%s: %v", harness.database, collection, err)
		}
	}
	if err := harness.admin.DropDatabase(ctx, milvusclient.NewDropDatabaseOption(harness.database)); err != nil {
		harness.t.Errorf("drop database %s: %v", harness.database, err)
		return
	}
	harness.t.Logf("dropped Milvus database %s", harness.database)
	if err := errors.Join(harness.milvus.Close(ctx), harness.admin.Close(ctx)); err != nil {
		harness.t.Errorf("close Milvus clients: %v", err)
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
	opened, err := library.Open(harness.ctx, library.Config{Store: descriptor, Vectors: vectors, Embedder: harness.embedder})
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
		harness.ctx,
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
		harness.ctx,
		`SELECT COUNT(*) FROM occurrences JOIN vectors ON vectors.vector_id = occurrences.vector_id WHERE vectors.state != 'verified'`,
	).Scan(&rows.unverifiedLinked); err != nil {
		harness.t.Fatalf("count unverified occurrence vectors: %v", err)
	}
	if err := database.QueryRowContext(harness.ctx, `SELECT COUNT(*) FROM vector_outbox`).Scan(&rows.outboxEntries); err != nil {
		harness.t.Fatalf("count outbox entries: %v", err)
	}
	return rows
}

func (harness *libraryHarness) scanPairs(database *sql.DB, query string, into map[string]string) {
	harness.t.Helper()
	result, err := database.QueryContext(harness.ctx, query)
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
	encoded, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("encode child request: %v", err)
	}
	command := exec.Command(os.Args[0], "-test.run=^$")
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
	var request libraryChildRequest
	if err := json.Unmarshal([]byte(encoded), &request); err != nil {
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
	if wait := time.Until(request.StartAt); wait > 0 {
		time.Sleep(wait)
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
	return err
}

func (store *crashingVectorStore) VerifyStrong(ctx context.Context, identities []library.VectorIdentity) error {
	err := store.VectorStore.VerifyStrong(ctx, identities)
	if store.crashPoint == crashAfterVerify && store.putDone && err == nil {
		killSelf()
	}
	return err
}

// killSelf ends the process with SIGKILL, which runs no deferred function and
// no cleanup.
func killSelf() {
	_ = syscall.Kill(os.Getpid(), syscall.SIGKILL)
	select {}
}
