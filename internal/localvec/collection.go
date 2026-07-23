package localvec

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/usearch"
)

const (
	indexFileName    = "index.usearch"
	metadataFileName = "metadata.jsonl"
)

type collection struct {
	name       string
	path       string
	mutex      sync.RWMutex
	rows       []row
	index      *usearch.Index
	dimensions int
	loaded     bool
	exists     bool
}

func newCollection(name string, path string) *collection {
	return &collection{
		name:       name,
		path:       path,
		mutex:      sync.RWMutex{},
		rows:       nil,
		index:      nil,
		dimensions: 0,
		loaded:     false,
		exists:     false,
	}
}

func (stored *collection) snapshot() ([]row, bool, error) {
	stored.mutex.Lock()
	defer stored.mutex.Unlock()
	if err := stored.loadLocked(); err != nil {
		return nil, false, err
	}
	return cloneRows(stored.rows), stored.exists, nil
}

func (stored *collection) rowCount() (int32, bool, error) {
	stored.mutex.Lock()
	defer stored.mutex.Unlock()
	if err := stored.loadLocked(); err != nil {
		return 0, false, err
	}
	return safeInt32(len(stored.rows)), stored.exists, nil
}

func (stored *collection) vectorCount() (int, bool, error) {
	stored.mutex.Lock()
	defer stored.mutex.Unlock()
	if err := stored.loadLocked(); err != nil {
		return 0, false, err
	}
	if !stored.exists {
		return 0, false, nil
	}
	count, err := stored.index.Size()
	if err != nil {
		slog.Error(
			"read local vector index size failed",
			"collection",
			stored.name,
			"err",
			err,
		)
		return 0, true, fmt.Errorf(
			"read usearch index size for local vector collection %s: %w",
			stored.name,
			err,
		)
	}
	return count, true, nil
}

func (stored *collection) nearest(
	vector []float32,
	count int,
) ([]row, []float32, bool, error) {
	stored.mutex.Lock()
	defer stored.mutex.Unlock()
	if err := stored.loadLocked(); err != nil {
		return nil, nil, false, err
	}
	if !stored.exists {
		return nil, nil, false, nil
	}
	keys, distances, err := stored.index.Search(vector, count)
	if err != nil {
		slog.Error(
			"search local vector index failed",
			"collection",
			stored.name,
			"count",
			count,
			"err",
			err,
		)
		return nil, nil, true, fmt.Errorf(
			"search usearch index for local vector collection %s: %w",
			stored.name,
			err,
		)
	}
	rowsByLabel := make(map[uint64]row, len(stored.rows))
	for _, candidate := range stored.rows {
		rowsByLabel[candidate.Label] = candidate
	}
	rows := make([]row, 0, len(keys))
	for _, key := range keys {
		candidate, found := rowsByLabel[key]
		if !found {
			return nil, nil, true, fmt.Errorf(
				"usearch returned unknown label %d for %s",
				key,
				stored.name,
			)
		}
		rows = append(rows, candidate)
	}
	return cloneRows(rows), append([]float32(nil), distances...), true, nil
}

func (stored *collection) mutate(
	removal semantic.Removal,
	added []row,
	requireExisting bool,
) error {
	stored.mutex.Lock()
	defer stored.mutex.Unlock()
	if err := stored.loadLocked(); err != nil {
		return err
	}
	if requireExisting && !stored.exists {
		return semantic.ErrCollectionMissing
	}
	if !stored.exists && len(added) == 0 {
		return nil
	}

	kept, _ := removeRows(stored.rows, removal)
	addedIDs := make(map[string]struct{}, len(added))
	for _, candidate := range added {
		addedIDs[candidate.ID] = struct{}{}
	}
	withoutReplaced := make([]row, 0, len(kept))
	for _, candidate := range kept {
		if _, found := addedIDs[candidate.ID]; found {
			continue
		}
		withoutReplaced = append(withoutReplaced, candidate)
	}
	rewritten := cloneRows(withoutReplaced)
	rewritten = append(rewritten, cloneRows(added)...)
	return stored.persistLocked(rewritten)
}

func (stored *collection) rewrite(
	requireExisting bool,
	transform func([]row) ([]row, error),
) error {
	stored.mutex.Lock()
	defer stored.mutex.Unlock()
	if err := stored.loadLocked(); err != nil {
		return err
	}
	if requireExisting && !stored.exists {
		return semantic.ErrCollectionMissing
	}
	if !stored.exists {
		return nil
	}
	rewritten, err := transform(cloneRows(stored.rows))
	if err != nil {
		return err
	}
	return stored.persistLocked(rewritten)
}

func (stored *collection) drop() error {
	stored.mutex.Lock()
	defer stored.mutex.Unlock()
	if stored.index != nil {
		stored.index.Close()
		stored.index = nil
	}
	if err := os.RemoveAll(stored.path); err != nil {
		slog.Error(
			"remove local vector collection failed",
			"collection",
			stored.name,
			"path",
			stored.path,
			"err",
			err,
		)
		return fmt.Errorf("remove local vector collection %s: %w", stored.path, err)
	}
	stored.rows = nil
	stored.dimensions = 0
	stored.loaded = true
	stored.exists = false
	return nil
}

func (stored *collection) loadLocked() error {
	if stored.loaded {
		return nil
	}
	info, err := os.Stat(stored.path)
	if errors.Is(err, os.ErrNotExist) {
		stored.loaded = true
		stored.exists = false
		return nil
	}
	if err != nil {
		slog.Error(
			"inspect local vector collection failed",
			"path",
			stored.path,
			"err",
			err,
		)
		return fmt.Errorf("inspect local vector collection %s: %w", stored.path, err)
	}
	if !info.IsDir() {
		return fmt.Errorf("local vector collection %s is not a directory", stored.path)
	}

	metadataPath := filepath.Join(stored.path, metadataFileName)
	rows, metadataExists, healed, err := readRows(metadataPath)
	if err != nil {
		return err
	}
	if !metadataExists {
		return fmt.Errorf("local vector collection %s has no metadata sidecar", stored.name)
	}
	if err := assignLabels(rows); err != nil {
		return err
	}
	if healed {
		if err := rewriteRows(metadataPath, rows); err != nil {
			return err
		}
	}
	dimensions, err := vectorDimensions(rows)
	if err != nil {
		return err
	}
	if dimensions == 0 {
		dimensions = 1
	}
	indexPath := filepath.Join(stored.path, indexFileName)
	vectorIndex, indexDimensions, err := readVectorIndex(
		stored.name,
		indexPath,
		rows,
		dimensions,
	)
	if err != nil {
		slog.Error(
			"load local vector index failed",
			"collection",
			stored.name,
			"path",
			indexPath,
			"err",
			err,
		)
		return err
	}
	stored.rows = rows
	stored.index = vectorIndex
	stored.dimensions = indexDimensions
	stored.exists = true
	stored.loaded = true
	return nil
}

func readVectorIndex(
	collectionName string,
	indexPath string,
	rows []row,
	dimensions int,
) (*usearch.Index, int, error) {
	vectorIndex, err := usearch.New(dimensions)
	if err != nil {
		slog.Error(
			"create local usearch index failed",
			"collection",
			collectionName,
			"dimensions",
			dimensions,
			"err",
			err,
		)
		return nil, 0, fmt.Errorf(
			"create usearch index for local vector collection %s: %w",
			collectionName,
			err,
		)
	}
	if err := vectorIndex.Load(indexPath); err != nil {
		vectorIndex.Close()
		slog.Error(
			"load local usearch index failed",
			"collection",
			collectionName,
			"path",
			indexPath,
			"err",
			err,
		)
		return nil, 0, fmt.Errorf("load local usearch index %s: %w", indexPath, err)
	}
	indexDimensions, err := vectorIndex.Dimensions()
	if err != nil {
		vectorIndex.Close()
		slog.Error(
			"read local usearch index dimensions failed",
			"collection",
			collectionName,
			"path",
			indexPath,
			"err",
			err,
		)
		return nil, 0, fmt.Errorf("read local usearch index dimensions: %w", err)
	}
	if len(rows) > 0 && dimensions != indexDimensions {
		vectorIndex.Close()
		return nil, 0, fmt.Errorf(
			"local vector metadata has %d dimensions, index has %d",
			dimensions,
			indexDimensions,
		)
	}
	size, err := vectorIndex.Size()
	if err != nil {
		vectorIndex.Close()
		slog.Error(
			"read local usearch index size failed",
			"collection",
			collectionName,
			"path",
			indexPath,
			"err",
			err,
		)
		return nil, 0, fmt.Errorf("read local usearch index size: %w", err)
	}
	if err := validateVectorIndexRows(collectionName, rows, size, vectorIndex); err != nil {
		vectorIndex.Close()
		return nil, 0, err
	}
	return vectorIndex, indexDimensions, nil
}

func validateVectorIndexRows(
	collectionName string,
	rows []row,
	size int,
	vectorIndex *usearch.Index,
) error {
	if size != len(rows) {
		return fmt.Errorf(
			"local vector collection %s has %d metadata rows and %d indexed vectors",
			collectionName,
			len(rows),
			size,
		)
	}
	for _, candidate := range rows {
		contains, containsErr := vectorIndex.Contains(candidate.Label)
		if containsErr != nil {
			slog.Error(
				"check local usearch index label failed",
				"collection",
				collectionName,
				"label",
				candidate.Label,
				"err",
				containsErr,
			)
			return fmt.Errorf(
				"check local usearch index label %d: %w",
				candidate.Label,
				containsErr,
			)
		}
		if !contains {
			return fmt.Errorf(
				"local vector index %s is missing label %d",
				collectionName,
				candidate.Label,
			)
		}
	}
	return nil
}

func (stored *collection) persistLocked(rows []row) error {
	rewritten := cloneRows(rows)
	if err := assignLabels(rewritten); err != nil {
		return err
	}
	sort.Slice(rewritten, func(left int, right int) bool {
		return rewritten[left].Label < rewritten[right].Label
	})
	dimensions, err := vectorDimensions(rewritten)
	if err != nil {
		return err
	}
	if dimensions == 0 {
		dimensions = stored.dimensions
	}
	if dimensions == 0 {
		return errors.New("cannot persist an empty local vector collection without dimensions")
	}
	vectorIndex, err := buildVectorIndex(rewritten, dimensions)
	if err != nil {
		slog.Error(
			"build local vector index failed",
			"collection",
			stored.name,
			"err",
			err,
		)
		return err
	}
	tempPath, err := writeCollectionDirectory(stored.path, rewritten, vectorIndex)
	if err != nil {
		vectorIndex.Close()
		slog.Error(
			"write local vector collection failed",
			"collection",
			stored.name,
			"path",
			stored.path,
			"err",
			err,
		)
		return err
	}
	if err := replaceCollectionDirectory(tempPath, stored.path); err != nil {
		vectorIndex.Close()
		_ = os.RemoveAll(tempPath)
		slog.Error(
			"replace local vector collection failed",
			"collection",
			stored.name,
			"path",
			stored.path,
			"err",
			err,
		)
		return err
	}
	if stored.index != nil {
		stored.index.Close()
	}
	stored.rows = rewritten
	stored.index = vectorIndex
	stored.dimensions = dimensions
	stored.exists = true
	stored.loaded = true
	return nil
}

func buildVectorIndex(rows []row, dimensions int) (*usearch.Index, error) {
	vectorIndex, err := usearch.New(dimensions)
	if err != nil {
		slog.Error("create usearch index failed", "dimensions", dimensions, "err", err)
		return nil, fmt.Errorf("create usearch index for local vectors: %w", err)
	}
	if err := vectorIndex.Reserve(len(rows)); err != nil {
		vectorIndex.Close()
		slog.Error("reserve usearch index failed", "row_count", len(rows), "err", err)
		return nil, fmt.Errorf("reserve usearch index for local vectors: %w", err)
	}
	for _, candidate := range rows {
		if err := vectorIndex.Add(candidate.Label, candidate.Vector); err != nil {
			vectorIndex.Close()
			slog.Error(
				"add local vector to usearch failed",
				"row_id",
				candidate.ID,
				"label",
				candidate.Label,
				"err",
				err,
			)
			return nil, fmt.Errorf(
				"add local vector row %s to usearch index: %w",
				candidate.ID,
				err,
			)
		}
	}
	return vectorIndex, nil
}

func writeCollectionDirectory(
	destination string,
	rows []row,
	vectorIndex *usearch.Index,
) (string, error) {
	parent := filepath.Dir(destination)
	tempPath, err := os.MkdirTemp(parent, "."+filepath.Base(destination)+".write-*")
	if err != nil {
		slog.Error(
			"create local vector staging directory failed",
			"path",
			destination,
			"err",
			err,
		)
		return "", fmt.Errorf("create local vector staging directory: %w", err)
	}
	removeTemp := true
	defer func() {
		if removeTemp {
			_ = os.RemoveAll(tempPath)
		}
	}()
	indexPath := filepath.Join(tempPath, indexFileName)
	if err := vectorIndex.Save(indexPath); err != nil {
		slog.Error("save local usearch index failed", "path", indexPath, "err", err)
		return "", fmt.Errorf("save local usearch index %s: %w", indexPath, err)
	}
	if err := os.Chmod(indexPath, 0o600); err != nil {
		slog.Error("set local vector index permissions failed", "path", indexPath, "err", err)
		return "", fmt.Errorf("set local vector index permissions: %w", err)
	}
	if err := rewriteRows(filepath.Join(tempPath, metadataFileName), rows); err != nil {
		return "", err
	}
	removeTemp = false
	return tempPath, nil
}

func replaceCollectionDirectory(source string, destination string) error {
	if err := recoverCollectionDirectory(destination); err != nil {
		return err
	}
	_, err := os.Stat(destination)
	if errors.Is(err, os.ErrNotExist) {
		if renameErr := os.Rename(source, destination); renameErr != nil {
			slog.Error(
				"install local vector collection failed",
				"source",
				source,
				"destination",
				destination,
				"err",
				renameErr,
			)
			return fmt.Errorf("install local vector collection %s: %w", destination, renameErr)
		}
		return nil
	}
	if err != nil {
		slog.Error(
			"inspect local vector collection before replacement failed",
			"path",
			destination,
			"err",
			err,
		)
		return fmt.Errorf("inspect local vector collection %s: %w", destination, err)
	}

	backupPath := destination + backupCollectionSuffix
	if err := os.Rename(destination, backupPath); err != nil {
		slog.Error(
			"backup local vector collection failed",
			"source",
			destination,
			"destination",
			backupPath,
			"err",
			err,
		)
		return fmt.Errorf("backup local vector collection %s: %w", destination, err)
	}
	if err := os.Rename(source, destination); err != nil {
		if rollbackErr := os.Rename(backupPath, destination); rollbackErr != nil {
			combinedErr := fmt.Errorf(
				"install local vector collection: %w; rollback failed: %w",
				err,
				rollbackErr,
			)
			slog.Error(
				"install and rollback local vector collection failed",
				"source",
				source,
				"destination",
				destination,
				"backup",
				backupPath,
				"install_err",
				err,
				"rollback_err",
				rollbackErr,
				"err",
				combinedErr,
			)
			return combinedErr
		}
		slog.Error(
			"install local vector collection failed",
			"source",
			source,
			"destination",
			destination,
			"err",
			err,
		)
		return fmt.Errorf("install local vector collection %s: %w", destination, err)
	}
	if err := os.RemoveAll(backupPath); err != nil {
		slog.Warn(
			"remove local vector backup directory failed",
			"path",
			backupPath,
			"err",
			err,
		)
	}
	return nil
}

func recoverCollectionDirectory(collectionPath string) error {
	collectionInfo, collectionErr := os.Stat(collectionPath)
	if collectionErr != nil && !errors.Is(collectionErr, os.ErrNotExist) {
		slog.Error(
			"inspect local vector collection for recovery failed",
			"path",
			collectionPath,
			"err",
			collectionErr,
		)
		return fmt.Errorf(
			"inspect local vector collection %s for recovery: %w",
			collectionPath,
			collectionErr,
		)
	}

	backupPath := collectionPath + backupCollectionSuffix
	backupInfo, backupErr := os.Stat(backupPath)
	if errors.Is(backupErr, os.ErrNotExist) {
		return nil
	}
	if backupErr != nil {
		slog.Error(
			"inspect local vector collection backup failed",
			"path",
			backupPath,
			"err",
			backupErr,
		)
		return fmt.Errorf("inspect local vector collection backup %s: %w", backupPath, backupErr)
	}
	if !backupInfo.IsDir() {
		return fmt.Errorf("local vector collection backup %s is not a directory", backupPath)
	}

	if errors.Is(collectionErr, os.ErrNotExist) {
		if err := os.Rename(backupPath, collectionPath); err != nil {
			slog.Error(
				"restore local vector collection backup failed",
				"source",
				backupPath,
				"destination",
				collectionPath,
				"err",
				err,
			)
			return fmt.Errorf(
				"restore local vector collection backup %s: %w",
				backupPath,
				err,
			)
		}
		return nil
	}
	if !collectionInfo.IsDir() {
		return fmt.Errorf("local vector collection %s is not a directory", collectionPath)
	}
	if err := os.RemoveAll(backupPath); err != nil {
		slog.Error(
			"remove stale local vector collection backup failed",
			"path",
			backupPath,
			"err",
			err,
		)
		return fmt.Errorf("remove stale local vector collection backup %s: %w", backupPath, err)
	}
	return nil
}

func removeRows(rows []row, removal semantic.Removal) ([]row, bool) {
	if removal.Empty() {
		return cloneRows(rows), false
	}
	paths := make(map[string]struct{}, len(removal.Paths))
	for _, relativePath := range removal.Paths {
		paths[relativePath] = struct{}{}
	}
	kept := make([]row, 0, len(rows))
	removed := false
	for _, stored := range rows {
		if _, found := paths[stored.RelativePath]; found {
			removed = true
			continue
		}
		if matchesAnyPrefix(stored.RelativePath, removal.Prefixes) {
			removed = true
			continue
		}
		kept = append(kept, stored)
	}
	return kept, removed
}

func matchesAnyPrefix(value string, prefixes []string) bool {
	for _, prefix := range prefixes {
		if prefix != "" && strings.HasPrefix(value, prefix) {
			return true
		}
	}
	return false
}

func safeInt32(value int) int32 {
	if value > math.MaxInt32 {
		return math.MaxInt32
	}
	if value < math.MinInt32 {
		return math.MinInt32
	}
	return int32(value)
}
