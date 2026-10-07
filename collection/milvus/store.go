package milvus

import (
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
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
}

func (store *Store) denseAnnRequest(depth int, vector []float32) *milvusclient.AnnRequest {
	request := milvusclient.NewAnnRequest(DenseVectorField, depth, entity.FloatVector(vector))
	for key, value := range store.options.DenseSearchParams {
		request = request.WithSearchParam(key, value)
	}
	return request
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
