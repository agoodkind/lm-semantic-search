//go:build live

package live

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"goodkind.io/lm-semantic-search/library"
)

type childStoreDescriptor struct {
	CatalogPath       string `json:"CatalogPath"`
	LockPath          string `json:"LockPath"`
	PoolID            string `json:"PoolID"`
	EmbeddingModel    string `json:"EmbeddingModel"`
	EmbeddingRevision string `json:"EmbeddingRevision"`
	Dimension         int    `json:"Dimension"`
	Normalization     string `json:"Normalization"`
}

type childScalarColumn struct {
	Name      string             `json:"Name"`
	Type      library.ScalarType `json:"Type"`
	Nullable  bool               `json:"Nullable"`
	Mutable   bool               `json:"Mutable"`
	MaxLength int                `json:"MaxLength"`
}

type childNamespace struct {
	ID      string                  `json:"ID"`
	Policy  library.NamespacePolicy `json:"Policy"`
	Scalars []childScalarColumn     `json:"Scalars"`
}

type childScalarValue struct {
	Type   library.ScalarType `json:"Type"`
	Null   bool               `json:"Null"`
	String string             `json:"String"`
	Bool   bool               `json:"Bool"`
	Int64  int64              `json:"Int64"`
}

type childOccurrence struct {
	RowKey         string                      `json:"RowKey"`
	SortKey        string                      `json:"SortKey"`
	SourceText     string                      `json:"SourceText"`
	SearchText     string                      `json:"SearchText"`
	EmbeddingInput string                      `json:"EmbeddingInput"`
	Scalars        map[string]childScalarValue `json:"Scalars"`
}

type childBatch struct {
	Namespace        string            `json:"Namespace"`
	OwnerID          string            `json:"OwnerID"`
	GenerationOrder  uint64            `json:"GenerationOrder"`
	IdempotencyToken string            `json:"IdempotencyToken"`
	Mode             library.BatchMode `json:"Mode"`
	Rows             []childOccurrence `json:"Rows"`
}

type childWireRequest struct {
	Action      string               `json:"action"`
	Environment libraryEnvironment   `json:"environment"`
	Database    string               `json:"database"`
	Collection  string               `json:"collection"`
	Descriptor  childStoreDescriptor `json:"descriptor"`
	Namespace   childNamespace       `json:"namespace"`
	Batches     []childBatch         `json:"batches"`
	CrashPoint  string               `json:"crash_point"`
	StartAt     time.Time            `json:"start_at"`
}

// MarshalLibraryChild encodes the child request without embedding credentials.
func MarshalLibraryChild(request libraryChildRequest) ([]byte, error) {
	wire := childWireRequest{
		Action: request.Action, Environment: request.Environment, Database: request.Database,
		Collection: request.Collection, Descriptor: childStoreDescriptor(request.Descriptor),
		Namespace: childNamespace{ID: request.Namespace.ID, Policy: request.Namespace.Policy, Scalars: nil},
		Batches:   nil, CrashPoint: request.CrashPoint, StartAt: request.StartAt,
	}
	for _, column := range request.Namespace.Scalars {
		wire.Namespace.Scalars = append(wire.Namespace.Scalars, childScalarColumn(column))
	}
	for _, batch := range request.Batches {
		converted := childBatch{
			Namespace: batch.Namespace, OwnerID: batch.OwnerID,
			GenerationOrder: batch.GenerationOrder, IdempotencyToken: batch.IdempotencyToken,
			Mode: batch.Mode, Rows: nil,
		}
		for _, row := range batch.Rows {
			scalars := make(map[string]childScalarValue, len(row.Scalars))
			for name, value := range row.Scalars {
				scalars[name] = childScalarValue(value)
			}
			converted.Rows = append(converted.Rows, childOccurrence{
				RowKey:  row.RowKey,
				SortKey: row.SortKey, SourceText: row.SourceText, SearchText: row.SearchText,
				EmbeddingInput: row.EmbeddingInput, Scalars: scalars,
			})
		}
		wire.Batches = append(wire.Batches, converted)
	}
	encoded, err := json.Marshal(wire)
	if err != nil {
		slog.Error("Encode library child transport", "err", err)
		return nil, fmt.Errorf("encode library child transport: %w", err)
	}
	return encoded, nil
}

// UnmarshalLibraryChild decodes the typed transport into a library request.
func UnmarshalLibraryChild(encoded []byte) (libraryChildRequest, error) {
	var wire childWireRequest
	if err := json.Unmarshal(encoded, &wire); err != nil {
		slog.Error("Decode library child transport", "err", err)
		return libraryChildRequest{}, fmt.Errorf("decode library child transport: %w", err)
	}
	request := libraryChildRequest{
		Action: wire.Action, Environment: wire.Environment,
		Database: wire.Database, Collection: wire.Collection,
		Descriptor: library.StoreDescriptor(wire.Descriptor),
		Namespace:  library.NamespaceSpec{ID: wire.Namespace.ID, Policy: wire.Namespace.Policy, Scalars: nil},
		Batches:    nil, CrashPoint: wire.CrashPoint, StartAt: wire.StartAt,
	}
	for _, column := range wire.Namespace.Scalars {
		request.Namespace.Scalars = append(request.Namespace.Scalars, library.ScalarColumn(column))
	}
	for _, batch := range wire.Batches {
		converted := library.Batch{
			Namespace: batch.Namespace, OwnerID: batch.OwnerID,
			GenerationOrder: batch.GenerationOrder, IdempotencyToken: batch.IdempotencyToken,
			Mode: batch.Mode, Rows: nil,
		}
		for _, row := range batch.Rows {
			scalars := make(map[string]library.ScalarValue, len(row.Scalars))
			for name, value := range row.Scalars {
				scalars[name] = library.ScalarValue(value)
			}
			converted.Rows = append(converted.Rows, library.Occurrence{
				RowKey:  row.RowKey,
				SortKey: row.SortKey, SourceText: row.SourceText, SearchText: row.SearchText,
				EmbeddingInput: row.EmbeddingInput, Scalars: scalars,
			})
		}
		request.Batches = append(request.Batches, converted)
	}
	return request, nil
}
