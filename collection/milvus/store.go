package milvus

import (
	"strconv"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/index"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

// Options configures a [Store].
type Options struct {
	// Hybrid adds a BM25 sparse leg to searches and a sparse vector column to
	// the collections the store creates.
	Hybrid bool
	// EmbeddingModel is written to the embeddingModel column of every row the
	// store inserts. An empty value writes null.
	EmbeddingModel string
	// DenseSearchParams sets index search parameters, such as nprobe, on every
	// dense vector search. Milvus uses the index default for an omitted key.
	DenseSearchParams map[string]string
	// DenseIndex is the dense vector index a created collection uses. A nil
	// value uses AUTOINDEX with the cosine metric.
	DenseIndex index.Index
}

func (store *Store) denseIndex() index.Index {
	if store.options.DenseIndex != nil {
		return store.options.DenseIndex
	}
	return index.NewAutoIndex(entity.COSINE)
}

func (store *Store) denseAnnRequest(depth int, vector []float32) *milvusclient.AnnRequest {
	request := milvusclient.NewAnnRequest(DenseVectorField, depth, entity.FloatVector(vector))
	if len(store.options.DenseSearchParams) > 0 {
		request = request.WithAnnParam(denseAnnParam(store.options.DenseSearchParams))
	}
	return request
}

// denseAnnParam returns the dense search parameters as one ANN parameter.
// Milvus reads index search parameters such as nprobe only from the params
// JSON of a search request, which the client fills from the ANN parameter. A
// value that parses as an integer is sent as a JSON number.
func denseAnnParam(params map[string]string) index.AnnParam {
	annParam := index.NewCustomAnnParam()
	for key, value := range params {
		if number, err := strconv.Atoi(value); err == nil {
			annParam.WithExtraParam(key, number)
			continue
		}
		annParam.WithExtraParam(key, value)
	}
	return annParam
}

// Store implements [collection.Store] on a Milvus client. It owns no
// connection lifecycle. The caller creates and closes the client.
type Store struct {
	client  *milvusclient.Client
	options Options
}

var _ collection.Store = (*Store)(nil)

// New returns a Store over client.
func New(client *milvusclient.Client, options Options) *Store {
	return &Store{client: client, options: options}
}
