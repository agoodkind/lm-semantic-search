// Package local implements [collection.Store] on files in one directory. Each
// collection is a row file and a vector index.
package local

import (
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/localvec"
)

// Options configures [Open].
type Options struct {
	Root string
	// EmbeddingModel is the model name written to each row and returned by
	// QueryRows.
	EmbeddingModel string
}

// Store stores the collections under one root directory.
type Store struct {
	*localvec.GenericStore
}

var _ collection.Store = (*Store)(nil)

// Open creates options.Root when the directory is absent.
func Open(options Options) (*Store, error) {
	if strings.TrimSpace(options.Root) == "" {
		return nil, errors.New("local collection store requires a root directory")
	}
	generic, err := localvec.OpenGeneric(options.Root, options.EmbeddingModel)
	if err != nil {
		slog.Error("open local collection store failed", "root", options.Root, "err", err)
		return nil, fmt.Errorf("open local collection store at %s: %w", options.Root, err)
	}
	return &Store{GenericStore: generic}, nil
}
