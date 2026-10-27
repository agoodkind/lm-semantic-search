//go:build live

package live

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/milvus-io/milvus/client/v3/entity"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestReuseFromSourceRowWithoutModelStoresNoModel(t *testing.T) {
	harness := newHarness(t)
	embedServer := newFakeEmbeddingServerWithDimension(t, nil, fakeEmbeddingDimension)
	serviceConfig := harness.childConfig()
	serviceConfig.OpenAIBaseURL = embedServer.URL
	serviceConfig.EmbeddingDimension = fakeEmbeddingDimension
	if serviceConfig.EmbeddingModel == "" {
		t.Fatal("The harness configuration has an empty embedding model. The test requires a configured model to distinguish a null model from the current model.")
	}
	service, err := semantic.NewService(harness.milvusContext, serviceConfig)
	if err != nil {
		t.Fatalf("The call to open the semantic service failed with error %v.", err)
	}
	t.Cleanup(func() { _ = service.Close(context.Background()) })

	sourcePath := filepath.Join(harness.stateRoot, "legacy-model-source-"+randomID())
	targetPath := filepath.Join(harness.stateRoot, "legacy-model-target-"+randomID())
	sourceCollection := service.CollectionName(sourcePath)
	targetCollection := service.CollectionName(targetPath)
	harness.trackCollectionFamily(sourceCollection)
	harness.trackCollectionFamily(targetCollection)
	harness.trackTemporaryCollection(semantic.ReuseCatalogCollectionName(serviceConfig))

	knownModelContent := "source content with a stored model " + randomID()
	legacyContent := "source content without a stored model " + randomID()
	freshContent := "target content without a source row " + randomID()
	legacyVector := markedVector(fakeEmbeddingDimension, 1, 2)
	stageAndPromoteContents(t, service, sourcePath, []string{knownModelContent})
	insertLegacyRows(t, harness, sourceCollection, []legacyReuseFixtureRow{{
		id:            "legacy-model-" + randomID(),
		content:       legacyContent,
		relativePath:  "legacy/source.txt",
		startLine:     1,
		endLine:       2,
		fileExtension: "txt",
		metadata:      "{}",
		vector:        legacyVector,
	}})

	targetChunks := []model.StoredChunk{
		{Content: knownModelContent, RelativePath: "known.txt"},
		{Content: legacyContent, RelativePath: "legacy.txt"},
		{Content: freshContent, RelativePath: "fresh.txt"},
	}
	reuse, err := service.LoadReuseVectorsForContentsFrom(
		context.Background(),
		[]string{targetCollection, sourceCollection},
		targetChunks,
	)
	if err != nil {
		t.Fatalf("The lookup in the target and source collections failed with error %v.", err)
	}
	wantSourceModelKnown := map[string]bool{knownModelContent: true, legacyContent: false}
	for reusedContent, wantKnown := range wantSourceModelKnown {
		reused, found := reuse[semantic.ContentVectorKey(reusedContent)]
		if !found {
			t.Fatalf("The lookup returned no vector for content %q stored in the source collection.", reusedContent)
		}
		if reused.SourceModelKnown != wantKnown {
			t.Fatalf("The lookup for content %q returned SourceModelKnown %t instead of %t.", reusedContent, reused.SourceModelKnown, wantKnown)
		}
	}
	if _, found := reuse[semantic.ContentVectorKey(freshContent)]; found {
		t.Fatalf("The lookup returned a vector for content %q that no collection stores.", freshContent)
	}
	if err := service.StageReindex(
		context.Background(),
		targetPath,
		targetChunks,
		semantic.Removal{},
		nil,
		reuse,
		semantic.CodeColumns(),
	); err != nil {
		t.Fatalf("The attempt to stage the target rows failed with error %v.", err)
	}
	if err := service.PromoteStaging(context.Background(), targetPath); err != nil {
		t.Fatalf("The attempt to promote the staged target collection failed with error %v.", err)
	}

	legacyRow := onlyRowForContent(t, harness, targetCollection, legacyContent)
	if legacyRow.vectorChecksum != checksumVector(entity.FloatVector(legacyVector)) {
		t.Fatalf("The target row for the legacy content stores a vector with checksum %s instead of the legacy source vector's checksum %s.", legacyRow.vectorChecksum, checksumVector(entity.FloatVector(legacyVector)))
	}
	if legacyRow.embeddingModelKnown {
		t.Fatalf("The target row stores model %q after reusing a vector from a source row without a model.", legacyRow.embeddingModel)
	}
	for _, currentModelContent := range []string{knownModelContent, freshContent} {
		row := onlyRowForContent(t, harness, targetCollection, currentModelContent)
		if row.embeddingModel != serviceConfig.EmbeddingModel {
			t.Fatalf("The target row for content %q stores model %q instead of configured model %q.", currentModelContent, row.embeddingModel, serviceConfig.EmbeddingModel)
		}
	}

	targetReuse, err := service.LoadReuseVectorsForContentsFrom(
		context.Background(),
		[]string{targetCollection},
		[]model.StoredChunk{{Content: legacyContent}},
	)
	if err != nil {
		t.Fatalf("The lookup in the target collection failed with error %v.", err)
	}
	if _, found := targetReuse[semantic.ContentVectorKey(legacyContent)]; !found {
		t.Fatalf("The lookup returned no vector for target content %q stored without a model.", legacyContent)
	}
}

func onlyRowForContent(
	t *testing.T,
	harness *harness,
	collectionName string,
	content string,
) storedRowSnapshot {
	t.Helper()
	rows := snapshotsForContent(t, harness, collectionName, content)
	if len(rows) != 1 {
		t.Fatalf("Collection %s stores content %q in %d rows instead of one.", collectionName, content, len(rows))
	}
	return rows[0]
}
