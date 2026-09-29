package library

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"sync"

	"goodkind.io/lm-semantic-search/library/internal/storebinding"

	// The driver registers the "sqlite3" database/sql driver. Clyde links the
	// same SQLite driver, so one SQLite build serves both callers.
	_ "github.com/mattn/go-sqlite3"
)

// sqliteBusyTimeoutMilliseconds bounds how long a catalog connection waits for
// another connection's write transaction before it fails.
const sqliteBusyTimeoutMilliseconds = 30000

// catalogUUIDBytes is the random byte length of a catalog UUID.
const catalogUUIDBytes = 16

// Library is an open shared search store. It owns its SQLite catalog handle
// and its writer lock file. The caller owns the vector backend and the
// embedder, and Close leaves both open.
type Library struct {
	config Config
	// catalog runs write transactions with BEGIN IMMEDIATE.
	catalog *sql.DB
	// reader runs read transactions with a deferred BEGIN, which takes no
	// write lock.
	reader      *sql.DB
	lock        *writerLock
	catalogUUID string
	closeOnce   sync.Once
	closeErr    error
}

// Open validates config, opens or creates the SQLite catalog in WAL mode, and
// binds the catalog UUID to the vector pool before any write. A catalog saved
// under another descriptor, schema version, or analyzer, and a pool bound to
// another catalog, return an error that wraps [ErrStoreMismatch]. Open then
// replays every vector write that an interrupted process left in the outbox.
func Open(ctx context.Context, config Config) (*Library, error) {
	resolved, err := config.resolved()
	if err != nil {
		return nil, err
	}
	if resolved.Vectors == nil || resolved.Embedder == nil {
		return nil, invalidRequest("open library: Config.Vectors and Config.Embedder are required")
	}
	descriptor, err := canonicalDescriptor(ctx, resolved.Store)
	if err != nil {
		return nil, err
	}
	resolved.Store = descriptor

	lock, err := openWriterLock(ctx, descriptor.LockPath)
	if err != nil {
		return nil, err
	}
	// The first connection switches a new catalog to WAL mode. SQLite returns
	// SQLITE_BUSY without waiting when another process writes the file during
	// that switch. The writer lock serializes the switch.
	release, err := lock.acquire(ctx)
	if err != nil {
		slog.ErrorContext(ctx, "open library failed", "catalog", descriptor.CatalogPath, "err", err)
		return nil, errors.Join(err, lock.close())
	}
	library, err := openLocked(ctx, resolved, lock)
	release()
	if err != nil {
		slog.ErrorContext(ctx, "open library failed", "catalog", descriptor.CatalogPath, "err", err)
		return nil, errors.Join(err, lock.close())
	}
	return library, nil
}

// openLocked opens both catalog handles and initializes the library. The
// caller owns the writer lock and closes it when openLocked fails.
func openLocked(ctx context.Context, resolved Config, lock *writerLock) (*Library, error) {
	catalog, err := openCatalog(ctx, resolved.Store.CatalogPath, catalogTxLockImmediate)
	if err != nil {
		return nil, err
	}
	reader, err := openCatalog(ctx, resolved.Store.CatalogPath, catalogTxLockDeferred)
	if err != nil {
		slog.ErrorContext(ctx, "open catalog reader failed", "err", err)
		return nil, errors.Join(err, closeCatalog(catalog))
	}
	library := &Library{
		config:      resolved,
		catalog:     catalog,
		reader:      reader,
		lock:        lock,
		catalogUUID: "",
		closeOnce:   sync.Once{},
		closeErr:    nil,
	}
	if err := library.initialize(ctx); err != nil {
		slog.ErrorContext(ctx, "initialize library failed", "err", err)
		return nil, errors.Join(err, closeCatalog(catalog), closeCatalog(reader))
	}
	return library, nil
}

// initialize creates or checks the catalog identity, binds the vector pool,
// and replays the outbox. The caller owns the writer lock.
func (library *Library) initialize(ctx context.Context) error {
	catalogUUID, err := library.initializeIdentity(ctx)
	if err != nil {
		return err
	}
	library.catalogUUID = catalogUUID

	hostname, err := os.Hostname()
	if err != nil {
		slog.WarnContext(ctx, "read writer host name failed", "err", err)
		hostname = "unknown"
	}
	descriptor := library.config.Store
	binding, err := storebinding.Encode(storebinding.Binding{
		CatalogUUID:       catalogUUID,
		CatalogPath:       descriptor.CatalogPath,
		WriterHost:        hostname,
		PoolID:            descriptor.PoolID,
		EmbeddingModel:    descriptor.EmbeddingModel,
		EmbeddingRevision: descriptor.EmbeddingRevision,
		Dimension:         descriptor.Dimension,
		Normalization:     descriptor.Normalization,
	})
	if err != nil {
		slog.ErrorContext(ctx, "encode catalog binding failed", "err", err)
		return fmt.Errorf("encode catalog binding: %w", err)
	}
	if err := library.config.Vectors.BindCatalog(ctx, binding); err != nil {
		slog.ErrorContext(ctx, "bind vector pool to catalog failed", "pool", library.config.Vectors.PoolIdentity(), "err", err)
		return fmt.Errorf("bind vector pool %s to catalog %s: %w", library.config.Vectors.PoolIdentity(), catalogUUID, err)
	}
	return library.replayOutbox(ctx)
}

// savedDescriptor is the catalog's stored identity. The analyzer identity is
// saved beside it.
type savedDescriptor struct {
	CatalogPath       string `json:"catalog_path"`
	PoolID            string `json:"pool_id"`
	EmbeddingModel    string `json:"embedding_model"`
	EmbeddingRevision string `json:"embedding_revision"`
	Dimension         int    `json:"dimension"`
	Normalization     string `json:"normalization"`
}

// initializeIdentity creates the schema and saves the catalog UUID,
// descriptor, and analyzer on first open. A later open compares them and
// returns an error that wraps [ErrStoreMismatch] on any difference.
func (library *Library) initializeIdentity(ctx context.Context) (string, error) {
	descriptor := library.config.Store
	wantDescriptor, err := json.Marshal(savedDescriptor{
		CatalogPath:       descriptor.CatalogPath,
		PoolID:            descriptor.PoolID,
		EmbeddingModel:    descriptor.EmbeddingModel,
		EmbeddingRevision: descriptor.EmbeddingRevision,
		Dimension:         descriptor.Dimension,
		Normalization:     descriptor.Normalization,
	})
	if err != nil {
		slog.ErrorContext(ctx, "encode store descriptor failed", "err", err)
		return "", fmt.Errorf("encode store descriptor: %w", err)
	}

	var catalogUUID string
	err = library.write(ctx, func(tx *sql.Tx) error {
		if err := createCatalogSchema(ctx, tx); err != nil {
			return err
		}
		savedUUID, found, err := readIdentityValue(ctx, tx, identityKeyCatalogUUID)
		if err != nil {
			return err
		}
		if !found {
			catalogUUID, err = newCatalogUUID()
			if err != nil {
				return err
			}
			return saveNewIdentity(ctx, tx, catalogUUID, string(wantDescriptor), library.config.AnalyzerIdentity)
		}
		catalogUUID = savedUUID
		return compareSavedIdentity(ctx, tx, string(wantDescriptor), library.config.AnalyzerIdentity)
	})
	return catalogUUID, err
}

func saveNewIdentity(ctx context.Context, tx *sql.Tx, catalogUUID string, descriptor string, analyzer string) error {
	values := [][2]string{
		{identityKeyCatalogUUID, catalogUUID},
		{identityKeyDescriptor, descriptor},
		{identityKeyAnalyzer, analyzer},
		{identityKeyVisibilityRevision, "0"},
		{identityKeyProjectionRevision, "0"},
	}
	for _, value := range values {
		if err := writeIdentityValue(ctx, tx, value[0], value[1]); err != nil {
			return err
		}
	}
	return nil
}

func compareSavedIdentity(ctx context.Context, tx *sql.Tx, descriptor string, analyzer string) error {
	checks := [][2]string{
		{identityKeyDescriptor, descriptor},
		{identityKeyAnalyzer, analyzer},
	}
	for _, check := range checks {
		saved, found, err := readIdentityValue(ctx, tx, check[0])
		if err != nil {
			return err
		}
		if !found || saved != check[1] {
			err := fmt.Errorf("%w: catalog %s is %q, the configuration requests %q", ErrStoreMismatch, check[0], saved, check[1])
			slog.ErrorContext(ctx, "catalog identity mismatch", "key", check[0], "err", err)
			return err
		}
	}
	return nil
}

func newCatalogUUID() (string, error) {
	random := make([]byte, catalogUUIDBytes)
	if _, err := rand.Read(random); err != nil {
		slog.Error("generate catalog UUID failed", "err", err)
		return "", fmt.Errorf("generate catalog UUID: %w", err)
	}
	return hex.EncodeToString(random), nil
}

// canonicalDescriptor resolves symbolic links in the catalog and lock
// directories. Every writer then uses the same canonical paths. Open creates a
// missing directory.
func canonicalDescriptor(ctx context.Context, descriptor StoreDescriptor) (StoreDescriptor, error) {
	catalogPath, err := canonicalFilePath(ctx, descriptor.CatalogPath)
	if err != nil {
		return StoreDescriptor{}, err
	}
	lockPath, err := canonicalFilePath(ctx, descriptor.LockPath)
	if err != nil {
		return StoreDescriptor{}, err
	}
	canonical := descriptor
	canonical.CatalogPath = catalogPath
	canonical.LockPath = lockPath
	return canonical, nil
}

func canonicalFilePath(ctx context.Context, path string) (string, error) {
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		slog.ErrorContext(ctx, "create catalog directory failed", "path", directory, "err", err)
		return "", fmt.Errorf("create catalog directory %s: %w", directory, err)
	}
	resolved, err := filepath.EvalSymlinks(directory)
	if err != nil {
		slog.ErrorContext(ctx, "resolve catalog directory failed", "path", directory, "err", err)
		return "", fmt.Errorf("resolve catalog directory %s: %w", directory, err)
	}
	return filepath.Join(resolved, filepath.Base(path)), nil
}

// Transaction lock modes of the SQLite driver.
const (
	catalogTxLockImmediate = "immediate"
	catalogTxLockDeferred  = "deferred"
)

// openCatalog opens the SQLite catalog in WAL mode with foreign keys, a busy
// timeout, and the given transaction lock mode on every connection.
func openCatalog(ctx context.Context, path string, txLock string) (*sql.DB, error) {
	query := url.Values{}
	query.Set("_busy_timeout", strconv.Itoa(sqliteBusyTimeoutMilliseconds))
	query.Set("_journal_mode", "WAL")
	query.Set("_synchronous", "FULL")
	query.Set("_foreign_keys", "on")
	query.Set("_txlock", txLock)
	dataSource := "file:" + path + "?" + query.Encode()
	catalog, err := sql.Open("sqlite3", dataSource)
	if err != nil {
		slog.ErrorContext(ctx, "open catalog failed", "path", path, "err", err)
		return nil, fmt.Errorf("open catalog %s: %w", path, err)
	}
	if err := catalog.PingContext(ctx); err != nil {
		closeErr := catalog.Close()
		slog.ErrorContext(ctx, "open catalog failed", "path", path, "err", err)
		return nil, errors.Join(fmt.Errorf("open catalog %s: %w", path, err), closeErr)
	}
	return catalog, nil
}

// Close releases the library's SQLite handle and writer lock file. It leaves
// the caller's vector backend and embedder open. A second call returns the
// first result.
func (library *Library) Close() error {
	library.closeOnce.Do(func() {
		library.closeErr = errors.Join(
			closeCatalog(library.catalog),
			closeCatalog(library.reader),
			library.lock.close(),
		)
	})
	return library.closeErr
}

func closeCatalog(catalog *sql.DB) error {
	if err := catalog.Close(); err != nil {
		slog.Error("close catalog failed", "err", err)
		return fmt.Errorf("close catalog: %w", err)
	}
	return nil
}

// write runs work in one SQLite write transaction and commits it when work
// returns nil.
func (library *Library) write(ctx context.Context, work func(*sql.Tx) error) error {
	tx, err := library.catalog.BeginTx(ctx, nil)
	if err != nil {
		slog.ErrorContext(ctx, "begin catalog transaction failed", "err", err)
		return fmt.Errorf("begin catalog transaction: %w", err)
	}
	if err := work(tx); err != nil {
		if rollbackErr := tx.Rollback(); rollbackErr != nil {
			slog.ErrorContext(ctx, "roll back catalog transaction failed", "err", rollbackErr)
			return errors.Join(err, fmt.Errorf("roll back catalog transaction: %w", rollbackErr))
		}
		return err
	}
	if err := tx.Commit(); err != nil {
		slog.ErrorContext(ctx, "commit catalog transaction failed", "err", err)
		return fmt.Errorf("commit catalog transaction: %w", err)
	}
	return nil
}
