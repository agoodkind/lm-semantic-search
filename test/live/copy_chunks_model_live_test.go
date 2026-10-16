//go:build live

package live

import (
	"context"
	"path/filepath"
	"slices"
	"testing"

	"github.com/milvus-io/milvus/client/v2/entity"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestCopyChunksWritesEachSourceRowModelOnTheCopiedRow(t *testing.T) {
	harness := newHarness(t)
	embedServer := newFakeEmbeddingServerWithDimension(t, nil, fakeEmbeddingDimension)
	serviceConfig := harness.childConfig()
	serviceConfig.OpenAIBaseURL = embedServer.URL
	serviceConfig.EmbeddingDimension = fakeEmbeddingDimension
	if serviceConfig.EmbeddingModel == "" {
		t.Fatal("The harness configuration has an empty embedding model. The test requires a configured model to distinguish a null model from the configured model.")
	}
	service, err := semantic.NewService(harness.milvusContext, serviceConfig)
	if err != nil {
		t.Fatalf("The semantic service failed to open with error %v.", err)
	}
	t.Cleanup(func() { _ = service.Close(context.Background()) })

	codebasePath := filepath.Join(harness.stateRoot, "copy-chunks-model-"+randomID())
	collectionName := service.CollectionName(codebasePath)
	harness.trackCollectionFamily(collectionName)
	harness.trackTemporaryCollection(semantic.ReuseCatalogCollectionName(serviceConfig))

	sourceRelativePath := "renamed/before.txt"
	destinationRelativePath := "renamed/after.txt"
	knownModelContent := "copied content with a stored model " + randomID()
	legacyContent := "copied content without a stored model " + randomID()
	legacyVector := markedVector(fakeEmbeddingDimension, 1, 2)
	if err := service.StageReindex(
		context.Background(),
		codebasePath,
		[]model.StoredChunk{{Content: knownModelContent, RelativePath: sourceRelativePath}},
		semantic.Removal{},
		nil,
		map[string]model.ReuseVector{},
		semantic.CodeColumns(),
	); err != nil {
		t.Fatalf("The source row with a stored model failed to stage for codebase %s with error %v.", codebasePath, err)
	}
	if err := service.PromoteStaging(context.Background(), codebasePath); err != nil {
		t.Fatalf("The staged collection promotion for codebase %s failed with error %v.", codebasePath, err)
	}
	insertLegacyRows(t, harness, collectionName, []legacyReuseFixtureRow{{
		id:            "legacy-copy-" + randomID(),
		content:       legacyContent,
		relativePath:  sourceRelativePath,
		startLine:     3,
		endLine:       4,
		fileExtension: "txt",
		metadata:      "{}",
		vector:        legacyVector,
	}})
	knownModelSource := onlyRowForContent(t, harness, collectionName, knownModelContent)
	if knownModelSource.embeddingModel != serviceConfig.EmbeddingModel {
		t.Fatalf("Before the copy, the staged source row stores model %q instead of the configured model %q.", knownModelSource.embeddingModel, serviceConfig.EmbeddingModel)
	}
	if legacySource := onlyRowForContent(t, harness, collectionName, legacyContent); legacySource.embeddingModelKnown {
		t.Fatalf("Before the copy, the legacy source row stores model %q instead of a null model.", legacySource.embeddingModel)
	}

	copiedRows, err := service.CopyChunks(
		context.Background(),
		codebasePath,
		[]semantic.ChunkCopy{{Source: sourceRelativePath, Destination: destinationRelativePath}},
	)
	if err != nil {
		t.Fatalf("CopyChunks from source path %s to destination path %s in collection %s returned error %v.", sourceRelativePath, destinationRelativePath, collectionName, err)
	}
	if !slices.Equal(copiedRows, []int{2}) {
		t.Fatalf("CopyChunks reported %v copied rows instead of [2].", copiedRows)
	}

	legacyCopy := onlyRowForContent(t, harness, collectionName, legacyContent)
	if legacyCopy.relativePath != destinationRelativePath {
		t.Fatalf("The row copied from the source row without a model has relative path %q instead of %q.", legacyCopy.relativePath, destinationRelativePath)
	}
	if legacyCopy.embeddingModelKnown {
		t.Fatalf("The row copied from the source row without a model stores model %q instead of null.", legacyCopy.embeddingModel)
	}
	if legacyCopy.vectorChecksum != checksumVector(entity.FloatVector(legacyVector)) {
		t.Fatalf("The row copied from the source row without a model stores a vector with checksum %s instead of the source checksum %s.", legacyCopy.vectorChecksum, checksumVector(entity.FloatVector(legacyVector)))
	}
	knownModelCopy := onlyRowForContent(t, harness, collectionName, knownModelContent)
	if knownModelCopy.relativePath != destinationRelativePath {
		t.Fatalf("The row copied from the source row with a model has relative path %q instead of %q.", knownModelCopy.relativePath, destinationRelativePath)
	}
	if !knownModelCopy.embeddingModelKnown || knownModelCopy.embeddingModel != knownModelSource.embeddingModel {
		t.Fatalf("The row copied from the source row with a model stores model %q instead of the source model %q.", knownModelCopy.embeddingModel, knownModelSource.embeddingModel)
	}
	if knownModelCopy.vectorChecksum != knownModelSource.vectorChecksum {
		t.Fatalf("The row copied from the source row with a model stores a vector with checksum %s instead of the source checksum %s.", knownModelCopy.vectorChecksum, knownModelSource.vectorChecksum)
	}
}
