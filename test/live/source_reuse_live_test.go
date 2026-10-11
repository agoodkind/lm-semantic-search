//go:build live

package live

import (
	"context"
	"path/filepath"
	"slices"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/store"
)

func stageAndPromoteContents(
	t *testing.T,
	service *semantic.Service,
	codebasePath string,
	contents []string,
) {
	t.Helper()

	chunks := make([]model.StoredChunk, 0, len(contents))
	for _, content := range contents {
		chunks = append(chunks, model.StoredChunk{
			Content:      content,
			RelativePath: filepath.Join("source", randomID()+".txt"),
		})
	}
	if err := service.StageReindex(
		context.Background(),
		codebasePath,
		chunks,
		semantic.Removal{},
		nil,
		map[string]model.ReuseVector{},
		semantic.CodeColumns(),
	); err != nil {
		t.Fatalf("Staging the contents at %s failed with error %v.", codebasePath, err)
	}
	if err := service.PromoteStaging(context.Background(), codebasePath); err != nil {
		t.Fatalf("Promotion of the staged collection at %s failed with error %v.", codebasePath, err)
	}
}

func TestExplicitSourceReuseReadsOnlyListedCollections(t *testing.T) {
	harness := newHarness(t)
	embedServer := newFakeEmbeddingServerWithDimension(t, nil, fakeEmbeddingDimension)
	lookupConfig := harness.childConfig()
	lookupConfig.OpenAIBaseURL = embedServer.URL
	lookupConfig.EmbeddingDimension = fakeEmbeddingDimension
	lookupConfig.RegistryPath = filepath.Join(harness.stateRoot, "explicit-source-registry-"+randomID()+".json")
	service, err := semantic.NewService(harness.milvusContext, lookupConfig)
	if err != nil {
		t.Fatalf("Opening the semantic service failed with error %v.", err)
	}
	t.Cleanup(func() { _ = service.Close(context.Background()) })

	targetPath := filepath.Join(harness.stateRoot, "explicit-target-"+randomID())
	sourcePath := filepath.Join(harness.stateRoot, "explicit-source-"+randomID())
	unrelatedPath := filepath.Join(harness.stateRoot, "explicit-unrelated-"+randomID())
	targetCollection := service.CollectionName(targetPath)
	sourceCollection := service.CollectionName(sourcePath)
	unrelatedCollection := service.CollectionName(unrelatedPath)
	catalogCollection := semantic.ReuseCatalogCollectionName(lookupConfig)
	for _, collectionName := range []string{targetCollection, sourceCollection, unrelatedCollection} {
		harness.trackCollectionFamily(collectionName)
	}
	harness.trackTemporaryCollection(catalogCollection)

	sourceContent := "explicit source content " + randomID()
	unrelatedOnlyContent := "explicit unrelated content " + randomID()
	stageAndPromoteContents(t, service, sourcePath, []string{sourceContent})
	stageAndPromoteContents(t, service, unrelatedPath, []string{sourceContent, unrelatedOnlyContent})
	registry := model.RegistryFile{Codebases: []model.Codebase{
		{ID: "explicit-source", CanonicalPath: sourcePath, CollectionName: sourceCollection},
		{ID: "explicit-unrelated", CanonicalPath: unrelatedPath, CollectionName: unrelatedCollection},
	}}
	if err := store.WriteRegistry(lookupConfig.RegistryPath, registry); err != nil {
		t.Fatalf("Writing the registry at %s failed with error %v.", lookupConfig.RegistryPath, err)
	}

	requestedChunks := []model.StoredChunk{{Content: sourceContent}, {Content: unrelatedOnlyContent}}
	harness.callRecorder.reset()
	reuse, err := service.LoadReuseVectorsForContentsFrom(
		context.Background(),
		[]string{targetCollection, sourceCollection},
		requestedChunks,
	)
	if err != nil {
		t.Fatalf("The explicit source lookup returned error %v.", err)
	}
	if _, found := reuse[semantic.ContentVectorKey(sourceContent)]; !found {
		t.Fatalf("The explicit source lookup returned no vector for content %q stored in the source collection.", sourceContent)
	}
	if _, found := reuse[semantic.ContentVectorKey(unrelatedOnlyContent)]; found {
		t.Fatalf("The explicit source lookup returned content %q stored only in the unrelated collection.", unrelatedOnlyContent)
	}
	for _, call := range harness.callRecorder.snapshot() {
		if slices.Contains(call.collectionNames, unrelatedCollection) ||
			slices.Contains(call.collectionNames, catalogCollection) {
			t.Fatalf("The lookup called Milvus method %s for a collection outside the listed collections %v using caller %s.", call.method, call.collectionNames, call.caller)
		}
	}

	storeWideReuse, err := service.LoadReuseVectorsForContents(
		context.Background(),
		targetCollection,
		requestedChunks,
	)
	if err != nil {
		t.Fatalf("The lookup across the whole store returned error %v.", err)
	}
	if _, found := storeWideReuse[semantic.ContentVectorKey(unrelatedOnlyContent)]; !found {
		t.Fatalf("The lookup across the whole store returned no vector for content %q.", unrelatedOnlyContent)
	}
}
