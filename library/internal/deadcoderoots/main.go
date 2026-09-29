// Command deadcoderoots is the reachability root of the exported library API
// for the deadcode lint gate. No LMS program calls most of the library yet,
// and Clyde and other consumers import it from other modules. The main
// function calls every exported function and method of the library packages,
// and deadcode then counts that API and everything it calls as reachable. A new
// exported function or method needs a call here. deadcode still reports an
// unexported function that none of these calls invokes directly or
// transitively.
//
// The program is never run. Its calls use zero values and nil receivers.
package main

import (
	"context"
	"log/slog"

	"github.com/milvus-io/milvus/client/v2/milvusclient"

	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/embedded"
	"goodkind.io/lm-semantic-search/library/embedding"
	"goodkind.io/lm-semantic-search/library/embedding/onnx"
	"goodkind.io/lm-semantic-search/library/milvus"
)

func main() {
	slog.Debug("deadcode roots process entry")
	ctx := context.Background()
	callLibrary(ctx)
	callEmbedded(ctx)
	callEmbedding(ctx)
	callMilvus(ctx)
}

func callLibrary(ctx context.Context) {
	var (
		config      library.Config
		descriptor  library.StoreDescriptor
		spec        library.NamespaceSpec
		occurrence  library.Occurrence
		opened      *library.Library
		key         library.GenerationKey
		seal        library.GenerationSeal
		stageBatch  library.StageBatch
		batch       library.Batch
		ids         []library.OccurrenceID
		request     library.SearchRequest
		projection  library.ScalarProjection
		prepare     library.PrepareRequest
		occurrences []library.Occurrence
	)
	_, _ = library.Open(ctx, config)
	_, _ = library.PrepareText(ctx, prepare)
	_, _ = library.SealRows(occurrences)
	_ = config.Validate()
	_ = config.ValidateSearchRequest(spec, request)
	_ = descriptor.Validate()
	_ = spec.Validate()
	_ = spec.ValidateOccurrence(occurrence)
	_ = opened.AbortGeneration(ctx, key)
	_, _ = opened.Apply(ctx, batch)
	_ = opened.Close()
	_, _ = opened.CommitGeneration(ctx, key, seal)
	_ = opened.Delete(ctx, ids)
	_, _ = opened.GetOwnerState(ctx, "", "")
	_, _ = opened.ListOwnerOccurrences(ctx, "", "")
	_ = opened.RegisterNamespace(ctx, spec)
	_, _ = opened.ReprojectScalars(ctx, projection)
	_, _ = opened.Search(ctx, request)
	_ = opened.Stage(ctx, stageBatch)
}

func callEmbedded(ctx context.Context) {
	var (
		config     embedded.Config
		store      *embedded.Store
		record     library.VectorRecord
		identities []library.VectorIdentity
	)
	_, _ = embedded.New(config)
	_ = store.BindCatalog(ctx, "")
	_ = store.PoolIdentity()
	_ = store.PutCanonical(ctx, record)
	_, _ = store.ScoreExact(ctx, nil, nil)
	_ = store.VerifyStrong(ctx, identities)
}

func callEmbedding(ctx context.Context) {
	var (
		openAIConfig embedding.OpenAIConfig
		onnxConfig   onnx.Config
		tokenizer    *onnx.Tokenizer
	)
	_, _ = embedding.NewOpenAI(ctx, openAIConfig)
	_, _ = onnx.New(ctx, onnxConfig)
	_, _ = onnx.NewTokenizer(ctx, onnxConfig)
	_, _ = tokenizer.CountTokens(ctx, "")
	_ = tokenizer.MaxInputBytes()
	_ = tokenizer.MaxTokens()
}

func callMilvus(ctx context.Context) {
	var (
		client     *milvusclient.Client
		config     milvus.Config
		store      *milvus.Store
		record     library.VectorRecord
		identities []library.VectorIdentity
	)
	_, _ = milvus.New(client, config)
	_ = store.BindCatalog(ctx, "")
	_ = store.PoolIdentity()
	_ = store.PutCanonical(ctx, record)
	_, _ = store.ScoreExact(ctx, nil, nil)
	_ = store.VerifyStrong(ctx, identities)
}
