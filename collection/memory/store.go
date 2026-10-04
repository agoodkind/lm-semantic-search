// Package memory implements [collection.Store] on the Go heap. A Store writes
// no file and opens no connection, and its rows end with the process. Search
// scores every row the filter matches by cosine similarity and runs no lexical
// leg.
package memory

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"math"
	"slices"
	"strings"
	"sync"

	"goodkind.io/lm-semantic-search/collection"
)

// Options configures a [Store].
type Options struct {
	// EmbeddingModel is the model name [Store.QueryRows] reports for every row.
	EmbeddingModel string
}

// Store implements [collection.Store] in process memory. Its methods are safe
// for concurrent use.
type Store struct {
	options     Options
	mutex       sync.RWMutex
	collections map[string]*storedCollection
}

var _ collection.Store = (*Store)(nil)

// New returns an empty Store.
func New(options Options) *Store {
	return &Store{
		options:     options,
		mutex:       sync.RWMutex{},
		collections: make(map[string]*storedCollection),
	}
}

type storedCollection struct {
	dimension int
	scalars   []collection.ScalarColumn
	rows      map[string]*storedRow
}

// storedRow is immutable after Upsert stores it. A backfill replaces the map
// entry with a changed copy and does not write to the stored value.
type storedRow struct {
	row         collection.Row
	contentHash string
	norm        float64
}

func requireName(collectionName string) (string, error) {
	trimmed := strings.TrimSpace(collectionName)
	if trimmed == "" {
		return "", errors.New("collection name is required")
	}
	return trimmed, nil
}

func (stored *storedCollection) column(name string) (collection.ScalarColumn, bool) {
	for _, declared := range stored.scalars {
		if declared.Name == name {
			return declared, true
		}
	}
	return collection.ScalarColumn{Name: "", Type: "", Nullable: false, MaxLength: 0}, false
}

func supportedType(scalarType collection.ScalarType) bool {
	switch scalarType {
	case collection.ScalarTypeString, collection.ScalarTypeBool, collection.ScalarTypeInt64:
		return true
	default:
		return false
	}
}

// EnsureCollection creates the collection when it is absent. A collection that
// exists gains any declared scalar column it lacks. A column added to a
// collection with rows must be nullable, because those rows store no value for
// it.
func (store *Store) EnsureCollection(ctx context.Context, request collection.EnsureRequest) error {
	name, err := requireName(request.Collection)
	if err != nil {
		return err
	}
	for _, declared := range request.Declaration.Scalars {
		if !supportedType(declared.Type) {
			err := fmt.Errorf("ensure collection %s: column %s has unsupported type %q", name, declared.Name, declared.Type)
			slog.ErrorContext(ctx, "ensure memory collection failed", "collection", name, "err", err)
			return err
		}
	}
	store.mutex.Lock()
	defer store.mutex.Unlock()
	stored, exists := store.collections[name]
	if !exists {
		if request.Dimension <= 0 {
			err := fmt.Errorf("create collection %s: dimension %d is not positive", name, request.Dimension)
			slog.ErrorContext(ctx, "ensure memory collection failed", "collection", name, "err", err)
			return err
		}
		stored = &storedCollection{
			dimension: request.Dimension,
			scalars:   nil,
			rows:      make(map[string]*storedRow),
		}
	}
	scalars := slices.Clone(stored.scalars)
	for _, declared := range request.Declaration.Scalars {
		position := slices.IndexFunc(scalars, func(existing collection.ScalarColumn) bool {
			return existing.Name == declared.Name
		})
		if position >= 0 {
			if scalars[position].Type != declared.Type {
				err := fmt.Errorf("ensure collection %s: column %s is declared %s and stored as %s", name, declared.Name, declared.Type, scalars[position].Type)
				slog.ErrorContext(ctx, "ensure memory collection failed", "collection", name, "err", err)
				return err
			}
			continue
		}
		if !declared.Nullable && len(stored.rows) > 0 {
			err := fmt.Errorf("ensure collection %s: column %s is not nullable and the collection stores rows", name, declared.Name)
			slog.ErrorContext(ctx, "ensure memory collection failed", "collection", name, "err", err)
			return err
		}
		scalars = append(scalars, declared)
	}
	stored.scalars = scalars
	store.collections[name] = stored
	return nil
}

// Drop removes a collection and its rows. It reports whether the collection
// existed.
func (store *Store) Drop(collectionName string) bool {
	name := strings.TrimSpace(collectionName)
	store.mutex.Lock()
	defer store.mutex.Unlock()
	_, exists := store.collections[name]
	delete(store.collections, name)
	return exists
}

// Upsert writes rows to a collection. A row replaces the stored row with the
// same ID. Upsert validates every row before it stores any row. A rejected
// call changes no stored row.
func (store *Store) Upsert(ctx context.Context, collectionName string, declaration collection.Declaration, rows []collection.Row) error {
	if len(rows) == 0 {
		return nil
	}
	name, err := requireName(collectionName)
	if err != nil {
		return err
	}
	store.mutex.Lock()
	defer store.mutex.Unlock()
	stored, exists := store.collections[name]
	if !exists {
		return collection.ErrCollectionMissing
	}
	prepared := make([]*storedRow, 0, len(rows))
	for _, row := range rows {
		next, prepareErr := stored.prepareRow(declaration.Scalars, row)
		if prepareErr != nil {
			slog.ErrorContext(ctx, "upsert memory collection row failed", "collection", name, "relative_path", row.RelativePath, "err", prepareErr)
			return fmt.Errorf("upsert into %s: %w", name, prepareErr)
		}
		prepared = append(prepared, next)
	}
	for _, next := range prepared {
		stored.rows[next.row.ID] = next
	}
	return nil
}

func (stored *storedCollection) prepareRow(declared []collection.ScalarColumn, row collection.Row) (*storedRow, error) {
	if row.ID == "" {
		return nil, fmt.Errorf("row %s has no ID", row.RelativePath)
	}
	if len(row.Vector) != stored.dimension {
		return nil, fmt.Errorf("row %s has a vector of width %d, want %d", row.RelativePath, len(row.Vector), stored.dimension)
	}
	norm, err := vectorNorm(row.Vector)
	if err != nil {
		return nil, fmt.Errorf("row %s: %w", row.RelativePath, err)
	}
	scalars := make(map[string]collection.ScalarValue, len(declared))
	for _, declaration := range declared {
		schemaColumn, inSchema := stored.column(declaration.Name)
		if !inSchema {
			return nil, fmt.Errorf("the collection has no column %s", declaration.Name)
		}
		if schemaColumn.Type != declaration.Type {
			return nil, fmt.Errorf("column %s is declared %s and stored as %s", declaration.Name, declaration.Type, schemaColumn.Type)
		}
		value, present := row.Scalars[declaration.Name]
		valid := present && !value.Null
		if !valid {
			if !declaration.Nullable {
				return nil, fmt.Errorf("row %s has no value for column %s, which is not nullable", row.RelativePath, declaration.Name)
			}
			continue
		}
		if value.Type != declaration.Type {
			return nil, fmt.Errorf("row %s has a %s value for %s column %s", row.RelativePath, value.Type, declaration.Type, declaration.Name)
		}
		value.String = strings.ToValidUTF8(value.String, replacementCharacter)
		scalars[declaration.Name] = value
	}
	content := strings.ToValidUTF8(row.Content, replacementCharacter)
	contentSum := sha256.Sum256([]byte(content))
	return &storedRow{
		row: collection.Row{
			ID:                row.ID,
			Content:           content,
			RelativePath:      strings.ToValidUTF8(row.RelativePath, replacementCharacter),
			StartLine:         row.StartLine,
			EndLine:           row.EndLine,
			FileExtension:     strings.ToValidUTF8(row.FileExtension, replacementCharacter),
			Metadata:          strings.ToValidUTF8(row.Metadata, replacementCharacter),
			SplitPart:         row.SplitPart,
			SplitPartRecorded: row.SplitPartRecorded,
			Vector:            slices.Clone(row.Vector),
			Scalars:           scalars,
		},
		contentHash: hex.EncodeToString(contentSum[:]),
		norm:        norm,
	}, nil
}

const replacementCharacter = "�"

// vectorNorm returns the Euclidean length of vector. It rejects a NaN or
// infinite component, because such a component makes the score order
// undefined.
func vectorNorm(vector []float32) (float64, error) {
	var squared float64
	for _, component := range vector {
		value := float64(component)
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return 0, errors.New("vector has a component that is not finite")
		}
		squared += value * value
	}
	return math.Sqrt(squared), nil
}

// withScalars returns a copy of the row with changed scalar values. A null
// value removes the stored value.
func (row *storedRow) withScalars(changed map[string]collection.ScalarValue) *storedRow {
	scalars := maps.Clone(row.row.Scalars)
	for name, value := range changed {
		if value.Null {
			delete(scalars, name)
			continue
		}
		scalars[name] = value
	}
	next := *row
	next.row.Scalars = scalars
	return &next
}
