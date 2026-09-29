// Package embedded implements [library.VectorStore] in one local directory.
// The directory stores one vector pool: a catalog binding file and one file for
// each canonical vector. ScoreExact reads every requested vector and computes
// its exact cosine similarity. The package builds no approximate index.
package embedded

import (
	"bytes"
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/internal/storebinding"
	"goodkind.io/lm-semantic-search/library/internal/vectorcodec"
)

const (
	// directoryMode is the mode of the pool root and its vector directories.
	directoryMode = 0o700
	// bindingFileName is the catalog binding file in the pool root.
	bindingFileName = "binding.json"
	// vectorsDirectoryName is the pool root subdirectory of the vector files.
	vectorsDirectoryName = "vectors"
	// vectorFileSuffix ends the name of every vector file.
	vectorFileSuffix = ".vec"
	// vectorFileHeader starts every vector file. The identity digest line, the
	// checksum line, and the [vectorcodec.Encode] bytes follow it.
	vectorFileHeader = "lms-embedded-vector-v1\n"
	// poolIdentityPrefix starts the value of [Store.PoolIdentity].
	poolIdentityPrefix = "embedded:"
	// missingFanOutByte replaces an absent ID byte in a fan-out directory name.
	missingFanOutByte = 0
)

// Config selects the pool directory of a [Store].
type Config struct {
	// Root is the absolute, clean directory of one vector pool. [New] creates it
	// with mode 0700 when it is absent.
	Root string
}

// Store is the embedded [library.VectorStore]. [Store.BindCatalog] must
// succeed before any other vector method, and the winning binding sets the
// vector dimension the store enforces.
type Store struct {
	root string

	mutex sync.RWMutex
	// dimension is the dimension of the bound catalog. It is zero until
	// BindCatalog succeeds.
	dimension int
}

// storedVector is the decoded content of one vector file.
type storedVector struct {
	identityDigest string
	checksum       string
	values         []float32
}

// New returns a store for config.Root. It creates the root directory when it
// is absent. A Root that is not absolute and clean returns an error that wraps
// [library.ErrInvalidRequest].
func New(config Config) (*Store, error) {
	if !filepath.IsAbs(config.Root) || filepath.Clean(config.Root) != config.Root {
		err := fmt.Errorf(
			"embedded vector store: root %q is not an absolute clean path: %w",
			config.Root,
			library.ErrInvalidRequest,
		)
		slog.Warn("create embedded vector store failed", "root", config.Root, "err", err)
		return nil, err
	}
	if err := os.MkdirAll(config.Root, directoryMode); err != nil {
		slog.Error("create embedded vector store root failed", "root", config.Root, "err", err)
		return nil, fmt.Errorf("embedded vector store: create root %s: %w", config.Root, err)
	}
	return &Store{root: config.Root, mutex: sync.RWMutex{}, dimension: 0}, nil
}

// PoolIdentity returns "embedded:" followed by the pool root.
func (store *Store) PoolIdentity() string {
	return poolIdentityPrefix + store.root
}

// BindCatalog binds the pool to the catalog in binding, the text of
// [storebinding.Encode]. The first binding in the pool root wins. The store
// writes the complete binding to a temporary file and hard-links that file to
// binding.json. The hard link fails when binding.json exists, and a reader
// finds either no binding.json or the complete binding. A later binding with
// the same catalog UUID and dimension succeeds. A binding with another catalog
// UUID or dimension returns an error that wraps [library.ErrStoreMismatch].
func (store *Store) BindCatalog(ctx context.Context, binding string) error {
	requested, err := storebinding.Decode(binding)
	if err != nil {
		return wrapCause(ctx, library.ErrInvalidRequest, "decode requested catalog binding", err)
	}
	if err := ctx.Err(); err != nil {
		return contextError(ctx, "bind catalog", err)
	}
	created, err := store.createBinding(ctx, binding)
	if err != nil {
		return err
	}
	if !created {
		stored, err := store.readBinding(ctx)
		if err != nil {
			return err
		}
		if stored.CatalogUUID != requested.CatalogUUID || stored.Dimension != requested.Dimension {
			return newError(ctx, library.ErrStoreMismatch, fmt.Sprintf(
				"pool %s is bound to catalog %s with dimension %d, not catalog %s with dimension %d",
				store.root,
				stored.CatalogUUID,
				stored.Dimension,
				requested.CatalogUUID,
				requested.Dimension,
			))
		}
	}
	store.mutex.Lock()
	store.dimension = requested.Dimension
	store.mutex.Unlock()
	return nil
}

// PutCanonical writes record as one vector file. Invalid values, a checksum
// that differs from [vectorcodec.Checksum] of the values, or an invalid ID
// return an error that wraps [library.ErrInvalidRequest]. A file that already
// exists for the ID with the same identity digest and checksum makes the call
// succeed without a write. A file with another digest or checksum stays
// unchanged, and the call returns an error that wraps [library.ErrVectorCorrupt].
func (store *Store) PutCanonical(ctx context.Context, record library.VectorRecord) error {
	dimension, err := store.boundDimension(ctx)
	if err != nil {
		return err
	}
	if err := validateRecord(ctx, record, dimension); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return contextError(ctx, "put canonical vector", err)
	}
	path := store.vectorPath(record.ID)
	content, found, err := readVectorFile(ctx, path)
	if err != nil {
		return err
	}
	if found {
		return compareExisting(ctx, record, content, dimension)
	}
	return store.writeVector(ctx, record, dimension)
}

// VerifyStrong reads the vector file of every identity. A missing file
// returns an error that wraps [library.ErrVectorMissing]. A stored digest or
// checksum that differs from the identity, or stored values that do not match
// the stored checksum, return an error that wraps [library.ErrVectorCorrupt].
func (store *Store) VerifyStrong(ctx context.Context, identities []library.VectorIdentity) error {
	dimension, err := store.boundDimension(ctx)
	if err != nil {
		return err
	}
	for _, identity := range identities {
		if err := ctx.Err(); err != nil {
			return contextError(ctx, "verify canonical vectors", err)
		}
		if err := validateID(ctx, identity.ID); err != nil {
			return err
		}
		stored, err := store.loadVector(ctx, identity.ID, dimension)
		if err != nil {
			return err
		}
		if stored.identityDigest != identity.IdentityDigest || stored.checksum != identity.Checksum {
			return newError(ctx, library.ErrVectorCorrupt, fmt.Sprintf(
				"vector %q stores digest %q and checksum %q, want digest %q and checksum %q",
				identity.ID,
				stored.identityDigest,
				stored.checksum,
				identity.IdentityDigest,
				identity.Checksum,
			))
		}
	}
	return nil
}

// ScoreExact returns the exact cosine similarity of query and every vector in
// ids, in request order. The score is the float64 dot product divided by both
// float64 norms. An invalid query, an invalid ID, or a repeated ID returns an
// error that wraps [library.ErrInvalidRequest]. A missing vector wraps
// [library.ErrVectorMissing]. A corrupt vector file or a nonfinite score wraps
// [library.ErrVectorCorrupt]. The call returns no partial result with an error.
func (store *Store) ScoreExact(
	ctx context.Context,
	query []float32,
	ids []string,
) ([]library.VectorScore, error) {
	dimension, err := store.boundDimension(ctx)
	if err != nil {
		return nil, err
	}
	if err := vectorcodec.Validate(query, dimension); err != nil {
		return nil, wrapCause(ctx, library.ErrInvalidRequest, "validate query vector", err)
	}
	if err := validateScoreIDs(ctx, ids); err != nil {
		return nil, err
	}
	queryNorm := norm(query)
	scores := make([]library.VectorScore, 0, len(ids))
	for _, id := range ids {
		if err := ctx.Err(); err != nil {
			return nil, contextError(ctx, "score canonical vectors", err)
		}
		stored, err := store.loadVector(ctx, id, dimension)
		if err != nil {
			return nil, err
		}
		score := dot(query, stored.values) / (queryNorm * norm(stored.values))
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return nil, newError(ctx, library.ErrVectorCorrupt, fmt.Sprintf(
				"vector %q has nonfinite cosine score %v",
				id,
				score,
			))
		}
		scores = append(scores, library.VectorScore{ID: id, Score: score})
	}
	return scores, nil
}

// boundDimension returns the dimension of the bound catalog. An unbound store
// returns an error that wraps [library.ErrInvalidRequest].
func (store *Store) boundDimension(ctx context.Context) (int, error) {
	store.mutex.RLock()
	dimension := store.dimension
	store.mutex.RUnlock()
	if dimension == 0 {
		return 0, newError(ctx, library.ErrInvalidRequest, fmt.Sprintf(
			"pool %s is not bound to a catalog",
			store.root,
		))
	}
	return dimension, nil
}

// createBinding hard-links a complete temporary copy of binding to
// binding.json. It returns false when binding.json already exists.
func (store *Store) createBinding(ctx context.Context, binding string) (bool, error) {
	temporaryPath, err := writeTemporaryFile(ctx, store.root, ".binding-*.tmp", []byte(binding))
	if err != nil {
		return false, err
	}
	defer removeTemporaryFile(ctx, temporaryPath)
	bindingPath := filepath.Join(store.root, bindingFileName)
	if err := os.Link(temporaryPath, bindingPath); err != nil {
		if errors.Is(err, os.ErrExist) {
			return false, nil
		}
		return false, wrapIO(ctx, "link catalog binding "+bindingPath, err)
	}
	if err := syncDirectory(ctx, store.root); err != nil {
		return false, err
	}
	return true, nil
}

// readBinding decodes binding.json. A file that does not decode returns an
// error that wraps [library.ErrStoreMismatch].
func (store *Store) readBinding(ctx context.Context) (storebinding.Binding, error) {
	bindingPath := filepath.Join(store.root, bindingFileName)
	content, err := os.ReadFile(filepath.Clean(bindingPath))
	if err != nil {
		return storebinding.Binding{}, wrapIO(ctx, "read catalog binding "+bindingPath, err)
	}
	stored, err := storebinding.Decode(string(content))
	if err != nil {
		return storebinding.Binding{}, wrapCause(
			ctx,
			library.ErrStoreMismatch,
			"decode stored catalog binding "+bindingPath,
			err,
		)
	}
	return stored, nil
}

// vectorPath returns the file path of id. The two fan-out directories are the
// hex values of the first and second bytes of id.
func (store *Store) vectorPath(id string) string {
	fanOut := []byte{missingFanOutByte, missingFanOutByte}
	copy(fanOut, id)
	return filepath.Join(
		store.root,
		vectorsDirectoryName,
		hex.EncodeToString(fanOut[:1]),
		hex.EncodeToString(fanOut[1:]),
		id+vectorFileSuffix,
	)
}

// loadVector reads and checks the vector file of id. A missing file returns an
// error that wraps [library.ErrVectorMissing].
func (store *Store) loadVector(ctx context.Context, id string, dimension int) (storedVector, error) {
	content, found, err := readVectorFile(ctx, store.vectorPath(id))
	if err != nil {
		return storedVector{}, err
	}
	if !found {
		return storedVector{}, newError(ctx, library.ErrVectorMissing, fmt.Sprintf(
			"vector %q is missing from pool %s",
			id,
			store.root,
		))
	}
	return parseStoredVector(ctx, id, content, dimension)
}

// writeVector writes record to a temporary file and hard-links it to the
// vector path. When another writer linked the vector path first, writeVector
// compares that file with record and leaves it unchanged.
func (store *Store) writeVector(ctx context.Context, record library.VectorRecord, dimension int) error {
	path := store.vectorPath(record.ID)
	directory := filepath.Dir(path)
	for _, component := range []string{
		filepath.Dir(filepath.Dir(directory)),
		filepath.Dir(directory),
		directory,
	} {
		if err := ensureDirectory(ctx, component); err != nil {
			return err
		}
	}
	temporaryPath, err := writeTemporaryFile(ctx, directory, ".vector-*.tmp", encodeVector(record))
	if err != nil {
		return err
	}
	defer removeTemporaryFile(ctx, temporaryPath)
	if err := os.Link(temporaryPath, path); err != nil {
		if !errors.Is(err, os.ErrExist) {
			return wrapIO(ctx, "link vector file "+path, err)
		}
		content, found, err := readVectorFile(ctx, path)
		if err != nil {
			return err
		}
		if !found {
			return wrapIO(ctx, "read concurrently written vector file "+path, os.ErrNotExist)
		}
		return compareExisting(ctx, record, content, dimension)
	}
	return syncDirectory(ctx, directory)
}

// compareExisting checks the stored vector file content of record.ID and
// compares it with record.
func compareExisting(ctx context.Context, record library.VectorRecord, content []byte, dimension int) error {
	stored, err := parseStoredVector(ctx, record.ID, content, dimension)
	if err != nil {
		return err
	}
	if stored.identityDigest != record.IdentityDigest || stored.checksum != record.Checksum {
		return newError(ctx, library.ErrVectorCorrupt, fmt.Sprintf(
			"vector %q already stores digest %q and checksum %q, not digest %q and checksum %q",
			record.ID,
			stored.identityDigest,
			stored.checksum,
			record.IdentityDigest,
			record.Checksum,
		))
	}
	return nil
}

// validateRecord checks the ID, identity digest, values, and checksum of
// record.
func validateRecord(ctx context.Context, record library.VectorRecord, dimension int) error {
	if err := validateID(ctx, record.ID); err != nil {
		return err
	}
	if strings.Contains(record.IdentityDigest, "\n") {
		return newError(ctx, library.ErrInvalidRequest, fmt.Sprintf(
			"vector %q identity digest contains a newline",
			record.ID,
		))
	}
	if err := vectorcodec.Validate(record.Values, dimension); err != nil {
		return wrapCause(ctx, library.ErrInvalidRequest, fmt.Sprintf("validate vector %q", record.ID), err)
	}
	if record.Checksum != vectorcodec.Checksum(record.Values) {
		return newError(ctx, library.ErrInvalidRequest, fmt.Sprintf(
			"vector %q checksum %q does not match its values",
			record.ID,
			record.Checksum,
		))
	}
	return nil
}

// validateID rejects an empty ID and an ID that contains a path separator or
// "..".
func validateID(ctx context.Context, id string) error {
	if id == "" || strings.ContainsAny(id, `/\`) || strings.Contains(id, "..") {
		return newError(ctx, library.ErrInvalidRequest, fmt.Sprintf("vector ID %q is not a valid file name", id))
	}
	return nil
}

// validateScoreIDs validates every ID and rejects a repeated ID.
func validateScoreIDs(ctx context.Context, ids []string) error {
	seen := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		if err := validateID(ctx, id); err != nil {
			return err
		}
		if _, repeated := seen[id]; repeated {
			return newError(ctx, library.ErrInvalidRequest, fmt.Sprintf("vector ID %q is repeated", id))
		}
		seen[id] = struct{}{}
	}
	return nil
}

// encodeVector returns the vector file content of record.
func encodeVector(record library.VectorRecord) []byte {
	var content bytes.Buffer
	content.WriteString(vectorFileHeader)
	content.WriteString(record.IdentityDigest)
	content.WriteByte('\n')
	content.WriteString(record.Checksum)
	content.WriteByte('\n')
	content.Write(vectorcodec.Encode(record.Values))
	return content.Bytes()
}

// decodeVector parses the vector file content of id. A failure wraps
// [library.ErrVectorCorrupt].
func decodeVector(ctx context.Context, id string, content []byte) (storedVector, error) {
	body, found := bytes.CutPrefix(content, []byte(vectorFileHeader))
	if !found {
		return storedVector{}, newError(ctx, library.ErrVectorCorrupt, fmt.Sprintf(
			"vector %q file has no format header",
			id,
		))
	}
	digest, body, found := bytes.Cut(body, []byte{'\n'})
	if !found {
		return storedVector{}, newError(ctx, library.ErrVectorCorrupt, fmt.Sprintf(
			"vector %q file has no identity digest line",
			id,
		))
	}
	checksum, encoded, found := bytes.Cut(body, []byte{'\n'})
	if !found {
		return storedVector{}, newError(ctx, library.ErrVectorCorrupt, fmt.Sprintf(
			"vector %q file has no checksum line",
			id,
		))
	}
	values, err := vectorcodec.Decode(encoded)
	if err != nil {
		return storedVector{}, wrapCause(ctx, library.ErrVectorCorrupt, fmt.Sprintf("decode vector %q values", id), err)
	}
	return storedVector{identityDigest: string(digest), checksum: string(checksum), values: values}, nil
}

// parseStoredVector decodes content and checks that the values have dimension
// and match the stored checksum. A failure wraps [library.ErrVectorCorrupt].
func parseStoredVector(ctx context.Context, id string, content []byte, dimension int) (storedVector, error) {
	stored, err := decodeVector(ctx, id, content)
	if err != nil {
		return storedVector{}, err
	}
	if len(stored.values) != dimension {
		return storedVector{}, newError(ctx, library.ErrVectorCorrupt, fmt.Sprintf(
			"vector %q stores %d values, want %d",
			id,
			len(stored.values),
			dimension,
		))
	}
	if actual := vectorcodec.Checksum(stored.values); actual != stored.checksum {
		return storedVector{}, newError(ctx, library.ErrVectorCorrupt, fmt.Sprintf(
			"vector %q values have checksum %q, file stores %q",
			id,
			actual,
			stored.checksum,
		))
	}
	return stored, nil
}

// readVectorFile returns the content of path and false when path is absent.
func readVectorFile(ctx context.Context, path string) ([]byte, bool, error) {
	content, err := os.ReadFile(filepath.Clean(path))
	if errors.Is(err, os.ErrNotExist) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, wrapIO(ctx, "read vector file "+path, err)
	}
	return content, true, nil
}

// writeTemporaryFile writes content to a new file in directory, syncs it, and
// returns its path.
func writeTemporaryFile(ctx context.Context, directory string, pattern string, content []byte) (string, error) {
	file, err := os.CreateTemp(directory, pattern)
	if err != nil {
		slog.ErrorContext(ctx, "create embedded vector store temporary file failed", "directory", directory, "err", err)
		return "", fmt.Errorf("embedded vector store: create temporary file in %s: %w", directory, err)
	}
	path := file.Name()
	_, writeErr := file.Write(content)
	var syncErr error
	if writeErr == nil {
		syncErr = file.Sync()
	}
	closeErr := file.Close()
	if err := errors.Join(writeErr, syncErr, closeErr); err != nil {
		slog.ErrorContext(ctx, "write embedded vector store temporary file failed", "path", path, "err", err)
		removeTemporaryFile(ctx, path)
		return "", fmt.Errorf("embedded vector store: write temporary file %s: %w", path, err)
	}
	return path, nil
}

// removeTemporaryFile removes a temporary file after the store links or
// abandons it. It logs a removal failure and returns nothing.
func removeTemporaryFile(ctx context.Context, path string) {
	if err := os.Remove(path); err != nil {
		slog.WarnContext(ctx, "remove embedded vector store temporary file failed", "path", path, "err", err)
	}
}

// ensureDirectory creates path when it is absent and syncs its parent after a
// creation.
func ensureDirectory(ctx context.Context, path string) error {
	err := os.Mkdir(path, directoryMode)
	if errors.Is(err, os.ErrExist) {
		return nil
	}
	if err != nil {
		return wrapIO(ctx, "create directory "+path, err)
	}
	return syncDirectory(ctx, filepath.Dir(path))
}

// syncDirectory flushes the entries of directory to stable storage.
func syncDirectory(ctx context.Context, directory string) error {
	handle, err := os.Open(filepath.Clean(directory))
	if err != nil {
		return wrapIO(ctx, "open directory "+directory, err)
	}
	syncErr := handle.Sync()
	closeErr := handle.Close()
	if err := errors.Join(syncErr, closeErr); err != nil {
		return wrapIO(ctx, "sync directory "+directory, err)
	}
	return nil
}

// dot returns the float64 dot product of two vectors of equal length.
func dot(left []float32, right []float32) float64 {
	var sum float64
	for index, value := range left {
		sum += float64(value) * float64(right[index])
	}
	return sum
}

// norm returns the float64 Euclidean norm of values.
func norm(values []float32) float64 {
	return math.Sqrt(dot(values, values))
}

// newError logs and returns an error that wraps sentinel.
func newError(ctx context.Context, sentinel error, detail string) error {
	err := fmt.Errorf("embedded vector store: %s: %w", detail, sentinel)
	slog.WarnContext(ctx, "embedded vector store request failed", "err", err)
	return err
}

// wrapCause logs and returns an error that wraps sentinel and cause.
func wrapCause(ctx context.Context, sentinel error, detail string, cause error) error {
	err := fmt.Errorf("embedded vector store: %s: %w: %w", detail, sentinel, cause)
	slog.WarnContext(ctx, "embedded vector store request failed", "err", err)
	return err
}

// wrapIO logs and returns an error that wraps a filesystem failure.
func wrapIO(ctx context.Context, detail string, cause error) error {
	err := fmt.Errorf("embedded vector store: %s: %w", detail, cause)
	slog.ErrorContext(ctx, "embedded vector store filesystem operation failed", "err", err)
	return err
}

// contextError logs and returns an error that wraps a context failure.
func contextError(ctx context.Context, operation string, cause error) error {
	err := fmt.Errorf("embedded vector store: %s: %w", operation, cause)
	slog.WarnContext(ctx, "embedded vector store operation stopped", "operation", operation, "err", err)
	return err
}
