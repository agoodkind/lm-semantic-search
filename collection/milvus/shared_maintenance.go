package milvus

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/index"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
)

const (
	mmapEnabledKey   = "mmap.enabled"
	mmapEnabledValue = "true"
	// milvusIndexNotFoundCode is the Milvus error code for a missing index.
	milvusIndexNotFoundCode = 700
)

// mmapTargets lists the fields and indexes of a collection that do not have
// mmap enabled. Every field except the primary key and the BM25 sparse vector
// supports mmap, and so does every index.
func (store *SharedStore) mmapTargets(ctx context.Context, collectionName string) ([]string, []string, error) {
	client := store.inner.client
	described, err := client.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(collectionName))
	if err != nil {
		return nil, nil, WrapError(ctx, err, "describe Milvus collection "+collectionName)
	}
	if described.Schema == nil {
		return nil, nil, fmt.Errorf("describe Milvus collection %s: missing schema", collectionName)
	}
	fields := make([]string, 0, len(described.Schema.Fields))
	for _, field := range described.Schema.Fields {
		if field.PrimaryKey || field.Name == SparseVectorField {
			continue
		}
		if field.DataType.IsVectorType() && field.Name != DenseVectorField {
			continue
		}
		if field.TypeParams[mmapEnabledKey] != mmapEnabledValue {
			fields = append(fields, field.Name)
		}
	}
	indexNames, err := client.ListIndexes(ctx, milvusclient.NewListIndexOption(collectionName))
	if err != nil {
		return nil, nil, WrapError(ctx, err, "list Milvus indexes of "+collectionName)
	}
	slices.Sort(indexNames)
	indexes := make([]string, 0, len(indexNames))
	for _, indexName := range indexNames {
		description, describeErr := client.DescribeIndex(ctx, milvusclient.NewDescribeIndexOption(collectionName, indexName))
		if describeErr != nil {
			return nil, nil, WrapError(ctx, describeErr, "describe Milvus index "+indexName+" of "+collectionName)
		}
		if description.Params()[mmapEnabledKey] != mmapEnabledValue {
			indexes = append(indexes, indexName)
		}
	}
	return fields, indexes, nil
}

// enableMmap enables mmap on every field and index of a released collection
// that lacks it.
func (store *SharedStore) enableMmap(ctx context.Context, collectionName string) error {
	fields, indexes, err := store.mmapTargets(ctx, collectionName)
	if err != nil {
		return err
	}
	client := store.inner.client
	for _, field := range fields {
		option := milvusclient.NewAlterCollectionFieldPropertiesOption(collectionName, field).WithProperty(mmapEnabledKey, mmapEnabledValue)
		if err := client.AlterCollectionFieldProperty(ctx, option); err != nil {
			return WrapError(ctx, err, "enable mmap on field "+field+" of "+collectionName)
		}
	}
	for _, indexName := range indexes {
		option := milvusclient.NewAlterIndexPropertiesOption(collectionName, indexName).WithProperty(mmapEnabledKey, mmapEnabledValue)
		if err := client.AlterIndexProperties(ctx, option); err != nil {
			return WrapError(ctx, err, "enable mmap on index "+indexName+" of "+collectionName)
		}
	}
	return nil
}

// EnsureMmap enables mmap on every field and index of the occurrence and vector
// collections that lacks it. A loaded collection that needs a change is
// released, changed, and loaded again; searches of that collection fail until
// the load completes. It returns the names of the changed collections.
func (store *SharedStore) EnsureMmap(ctx context.Context, collectionName string) ([]string, error) {
	occurrences := strings.TrimSpace(collectionName)
	client := store.inner.client
	changed := make([]string, 0, 2)
	for _, name := range []string{occurrences, VectorCollectionName(occurrences)} {
		fields, indexes, err := store.mmapTargets(ctx, name)
		if err != nil {
			return changed, err
		}
		if len(fields) == 0 && len(indexes) == 0 {
			continue
		}
		state, err := client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(name))
		if err != nil {
			return changed, WrapError(ctx, err, "read load state of "+name)
		}
		loaded := state.State == entity.LoadStateLoaded
		if loaded {
			if err := client.ReleaseCollection(ctx, milvusclient.NewReleaseCollectionOption(name)); err != nil {
				return changed, WrapError(ctx, err, "release Milvus collection "+name)
			}
		}
		alterErr := store.enableMmap(ctx, name)
		if loaded {
			if err := store.loadAndWait(ctx, name); err != nil {
				return changed, err
			}
		}
		if alterErr != nil {
			return changed, alterErr
		}
		changed = append(changed, name)
	}
	slog.InfoContext(ctx, "collection.shared_mmap_enabled", "collection", occurrences, "changed", changed)
	return changed, nil
}

func (store *SharedStore) loadAndWait(ctx context.Context, collectionName string) error {
	task, err := store.inner.client.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collectionName))
	if err != nil {
		return WrapError(ctx, err, "load Milvus collection "+collectionName)
	}
	if err := task.Await(ctx); err != nil {
		return WrapError(ctx, err, "await load of Milvus collection "+collectionName)
	}
	return nil
}

// denseIndexDefinition is the index type, metric type, and build parameters of
// a dense index. The build parameters exclude mmap.enabled and index_type and
// hold canonical JSON.
type denseIndexDefinition struct {
	indexType  string
	metricType string
	params     string
}

func (store *SharedStore) denseIndexDefinition(ctx context.Context, collectionName string) (denseIndexDefinition, error) {
	description, err := store.inner.client.DescribeIndex(ctx, milvusclient.NewDescribeIndexOption(collectionName, DenseVectorField))
	if err != nil {
		return denseIndexDefinition{}, WrapError(ctx, err, "describe dense index of "+collectionName)
	}
	values := description.Params()
	build := map[string]json.RawMessage{}
	if raw := values["params"]; raw != "" {
		if err := json.Unmarshal([]byte(raw), &build); err != nil {
			slog.ErrorContext(ctx, "decode dense index parameters failed", "collection", collectionName, "err", err)
			return denseIndexDefinition{}, fmt.Errorf("decode dense index parameters of %s: %w", collectionName, err)
		}
	}
	delete(build, mmapEnabledKey)
	delete(build, "index_type")
	encoded, err := json.Marshal(build)
	if err != nil {
		slog.ErrorContext(ctx, "encode dense index parameters failed", "collection", collectionName, "err", err)
		return denseIndexDefinition{}, fmt.Errorf("encode dense index parameters of %s: %w", collectionName, err)
	}
	return denseIndexDefinition{indexType: values["index_type"], metricType: values["metric_type"], params: string(encoded)}, nil
}

// milvusErrorCode is the exported method of the Milvus client RPC error.
type milvusErrorCode interface {
	Code() int32
}

// hasDenseIndex reports whether a collection has an index on the dense vector
// field. Milvus answers ListIndexes on a collection without indexes with the
// index-not-found error code.
func (store *SharedStore) hasDenseIndex(ctx context.Context, collectionName string) (bool, error) {
	indexNames, err := store.inner.client.ListIndexes(ctx, milvusclient.NewListIndexOption(collectionName).WithFieldName(DenseVectorField))
	var coded milvusErrorCode
	if errors.As(err, &coded) && coded.Code() == milvusIndexNotFoundCode {
		return false, nil
	}
	if err != nil {
		return false, WrapError(ctx, err, "list dense indexes of "+collectionName)
	}
	return len(indexNames) > 0, nil
}

// MatchDenseIndex gives the vector collection of the shared target the dense
// index type, metric type, and build parameters of the source collection's
// dense index. A missing or different target index is released, dropped when
// present, built again with the source definition and mmap, and loaded again
// when it was loaded. It reports whether it built the index.
func (store *SharedStore) MatchDenseIndex(ctx context.Context, source string, target string) (bool, error) {
	vectors := VectorCollectionName(strings.TrimSpace(target))
	want, err := store.denseIndexDefinition(ctx, strings.TrimSpace(source))
	if err != nil {
		return false, err
	}
	present, err := store.hasDenseIndex(ctx, vectors)
	if err != nil {
		return false, err
	}
	have := denseIndexDefinition{indexType: "", metricType: "", params: ""}
	if present {
		have, err = store.denseIndexDefinition(ctx, vectors)
		if err != nil {
			return false, err
		}
		if have == want {
			return false, nil
		}
	}
	client := store.inner.client
	state, err := client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(vectors))
	if err != nil {
		return false, WrapError(ctx, err, "read load state of "+vectors)
	}
	loaded := state.State == entity.LoadStateLoaded
	if loaded {
		if err := client.ReleaseCollection(ctx, milvusclient.NewReleaseCollectionOption(vectors)); err != nil {
			return false, WrapError(ctx, err, "release Milvus collection "+vectors)
		}
	}
	if present {
		if err := client.DropIndex(ctx, milvusclient.NewDropIndexOption(vectors, DenseVectorField)); err != nil {
			return false, WrapError(ctx, err, "drop dense index of "+vectors)
		}
	}
	definition := index.NewGenericIndex(DenseVectorField, map[string]string{"index_type": want.indexType, "metric_type": want.metricType, "params": want.params})
	task, err := client.CreateIndex(ctx, milvusclient.NewCreateIndexOption(vectors, DenseVectorField, definition).WithIndexName(DenseVectorField))
	if err != nil {
		return false, WrapError(ctx, err, "create dense index of "+vectors)
	}
	if err := task.Await(ctx); err != nil {
		return false, WrapError(ctx, err, "await dense index of "+vectors)
	}
	if err := store.enableMmap(ctx, vectors); err != nil {
		return false, err
	}
	if loaded {
		if err := store.loadAndWait(ctx, vectors); err != nil {
			return false, err
		}
	}
	slog.InfoContext(ctx, "collection.shared_dense_index_replaced", "collection", vectors, "from", have.indexType, "to", want.indexType, "params", want.params)
	return true, nil
}

// RemoveUnreferencedVectors deletes every vector row that no occurrence row
// references and returns the number of deleted vector rows. A write that
// stopped after its vector upsert and before its occurrence upsert leaves such
// rows.
func (store *SharedStore) RemoveUnreferencedVectors(ctx context.Context, collectionName string) (int64, error) {
	store.write.Lock()
	defer store.write.Unlock()
	occurrences := strings.TrimSpace(collectionName)
	vectors := VectorCollectionName(occurrences)
	var removed int64
	err := store.iterateBatch(ctx, vectors, VectorKeyField+` != ""`, []string{VectorKeyField}, vectorKeyBatchSize, func(resultSet milvusclient.ResultSet) error {
		keys := make([]string, 0, resultSet.ResultCount)
		for rowIndex := range resultSet.ResultCount {
			key, keyErr := requiredString(resultSet, VectorKeyField, rowIndex)
			if keyErr != nil {
				return keyErr
			}
			keys = append(keys, key)
		}
		pageRemoved, removeErr := store.removeUnreferencedKeys(ctx, occurrences, vectors, keys)
		removed += pageRemoved
		return removeErr
	})
	return removed, err
}
