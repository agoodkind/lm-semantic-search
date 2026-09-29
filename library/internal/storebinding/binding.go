// Package storebinding encodes the catalog binding that the library passes to
// [library.VectorStore.BindCatalog]. A vector adapter stores the encoded
// binding with its pool and compares the catalog UUID of the stored binding
// with the UUID of every later binding.
package storebinding

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
)

// Binding identifies the catalog that owns a vector pool and the writer that
// first bound it.
type Binding struct {
	CatalogUUID       string `json:"catalog_uuid"`
	CatalogPath       string `json:"catalog_path"`
	WriterHost        string `json:"writer_host"`
	PoolID            string `json:"pool_id"`
	EmbeddingModel    string `json:"embedding_model"`
	EmbeddingRevision string `json:"embedding_revision"`
	Dimension         int    `json:"dimension"`
	Normalization     string `json:"normalization"`
}

// Encode returns the JSON text of binding.
func Encode(binding Binding) (string, error) {
	encoded, err := json.Marshal(binding)
	if err != nil {
		slog.Error("encode catalog binding failed", "err", err)
		return "", fmt.Errorf("encode catalog binding: %w", err)
	}
	return string(encoded), nil
}

// Decode parses text produced by [Encode]. Text without a catalog UUID or
// without a positive dimension returns an error.
func Decode(text string) (Binding, error) {
	var binding Binding
	if err := json.Unmarshal([]byte(text), &binding); err != nil {
		slog.Error("decode catalog binding failed", "err", err)
		return Binding{}, fmt.Errorf("decode catalog binding: %w", err)
	}
	if binding.CatalogUUID == "" {
		err := errors.New("catalog binding has no catalog_uuid")
		slog.Error("decode catalog binding failed", "err", err)
		return Binding{}, err
	}
	if binding.Dimension <= 0 {
		err := fmt.Errorf("catalog binding dimension %d is not positive", binding.Dimension)
		slog.Error("decode catalog binding failed", "err", err)
		return Binding{}, err
	}
	return binding, nil
}
