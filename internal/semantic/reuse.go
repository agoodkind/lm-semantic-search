package semantic

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// reuseVectorBatchSize bounds one QueryIterator page when streaming a reuse
// map out of a collection, so a large source never materializes a single
// oversized Milvus response.
const reuseVectorBatchSize = 1000

// contentVectorKey is the reuse-map key for one chunk: the hex SHA-256 of its
// content. The dense embedding is a pure function of content (the embedder
// receives content only, with no path or codebase salt), so identical content
// anywhere produces the same vector. Keying on content lets a fresh build
// reuse an already-embedded vector instead of calling the embedder again.
func contentVectorKey(content string) string {
	sum := sha256.Sum256([]byte(content))
	return hex.EncodeToString(sum[:])
}

// LoadReuseVectors reads every chunk's content and dense vector from each
// collection in collectionNames and returns a map from contentVectorKey to the
// stored dense vector. The merge-down path passes the collections of the
// already-indexed child codebases that a new parent index contains, so the
// parent reuses their embeddings for the shared subtree rather than re-embedding
// it. A missing collection is skipped. The caller must restrict collectionNames
// to indexes built with the current embedding model, so a reused vector is
// valid for the parent's model.
func (service *Service) LoadReuseVectors(ctx context.Context, collectionNames []string) (map[string][]float32, error) {
	reuse := make(map[string][]float32)
	if !service.Available() || len(collectionNames) == 0 {
		return reuse, nil
	}
	for _, collectionName := range collectionNames {
		if err := service.loadReuseVectorsFromCollection(ctx, collectionName, reuse); err != nil {
			return nil, err
		}
	}
	slog.InfoContext(ctx, "semantic.reuse_vectors_loaded", "collections", len(collectionNames), "chunks", len(reuse))
	return reuse, nil
}

// LoadReuseVectorsForPrefix reads one collection's chunks whose relativePath
// begins with relativePathPrefix and returns the contentVectorKey -> vector
// map for them. A conversation ingest passes the live conversation collection
// and the changed conversation's conv/<id>/ prefix, loaded before that
// conversation's prefix delete runs, so the reindex reuses the unchanged
// chunks' stored vectors instead of re-embedding the whole conversation. A
// missing collection or an empty prefix returns an empty map.
func (service *Service) LoadReuseVectorsForPrefix(ctx context.Context, collectionName string, relativePathPrefix string) (map[string][]float32, error) {
	reuse := make(map[string][]float32)
	if !service.Available() || collectionName == "" || relativePathPrefix == "" {
		return reuse, nil
	}
	if err := service.loadReuseVectorsFiltered(ctx, collectionName, relativePathPrefixExpression(relativePathPrefix), reuse); err != nil {
		return nil, err
	}
	slog.DebugContext(
		ctx, "semantic.reuse_vectors_loaded_for_prefix",
		"collection", collectionName,
		"prefix", relativePathPrefix,
		"chunks", len(reuse),
	)
	return reuse, nil
}

// loadReuseVectorsFromCollection streams one whole source collection into reuse.
func (service *Service) loadReuseVectorsFromCollection(ctx context.Context, collectionName string, reuse map[string][]float32) error {
	return service.loadReuseVectorsFiltered(ctx, collectionName, relativePathFieldName+` != ""`, reuse)
}

// loadReuseVectorsFiltered streams the rows of one collection matching
// filterExpression into reuse, keyed by contentVectorKey. A missing collection
// loads nothing.
func (service *Service) loadReuseVectorsFiltered(ctx context.Context, collectionName string, filterExpression string, reuse map[string][]float32) error {
	hasCollection, err := service.milvus.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "check collection for reuse load failed", "collection", collectionName, "err", err)
		return fmt.Errorf("check Milvus collection %s: %w", collectionName, err)
	}
	if !hasCollection {
		return nil
	}
	if err := service.loadCollectionForRead(ctx, collectionName); err != nil {
		return err
	}

	iterator, err := service.milvus.QueryIterator(ctx, milvusclient.NewQueryIteratorOption(collectionName).
		WithBatchSize(reuseVectorBatchSize).
		WithFilter(filterExpression).
		WithOutputFields(contentFieldName, denseVectorFieldName))
	if err != nil {
		slog.ErrorContext(ctx, "open reuse query iterator failed", "collection", collectionName, "err", err)
		return fmt.Errorf("open query iterator for %s: %w", collectionName, err)
	}

	for {
		resultSet, nextErr := iterator.Next(ctx)
		if errors.Is(nextErr, io.EOF) {
			return nil
		}
		if nextErr != nil {
			slog.ErrorContext(ctx, "reuse query iterator next failed", "collection", collectionName, "err", nextErr)
			return fmt.Errorf("iterate %s: %w", collectionName, nextErr)
		}
		contentColumn := resultSet.GetColumn(contentFieldName)
		vectorColumn := resultSet.GetColumn(denseVectorFieldName)
		if contentColumn == nil || vectorColumn == nil {
			return ErrSearchResultIncomplete
		}
		for rowIndex := range resultSet.ResultCount {
			contentValue, contentErr := contentColumn.GetAsString(rowIndex)
			if contentErr != nil {
				return fmt.Errorf("read content column at %d: %w", rowIndex, contentErr)
			}
			vector, vectorErr := vectorAt(vectorColumn, rowIndex)
			if vectorErr != nil {
				return fmt.Errorf("read vector column at %d: %w", rowIndex, vectorErr)
			}
			reuse[contentVectorKey(contentValue)] = vector
		}
	}
}

// loadCollectionForRead ensures a collection is loaded before a query iterator
// reads it. A collection built in this process is already loaded, but one left
// unloaded by a Milvus restart would otherwise fail the read, so this loads it
// idempotently.
func (service *Service) loadCollectionForRead(ctx context.Context, collectionName string) error {
	loadTask, err := service.milvus.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collectionName))
	if err != nil {
		slog.ErrorContext(ctx, "load collection for reuse read failed", "collection", collectionName, "err", err)
		return fmt.Errorf("load Milvus collection %s: %w", collectionName, err)
	}
	if err := loadTask.Await(ctx); err != nil {
		slog.ErrorContext(ctx, "await collection load for reuse read failed", "collection", collectionName, "err", err)
		return fmt.Errorf("await Milvus collection load %s: %w", collectionName, err)
	}
	return nil
}
