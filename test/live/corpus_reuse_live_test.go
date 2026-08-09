//go:build live

package live

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestConversationContentReusesVectorAcrossCorpus(t *testing.T) {
	gate := &embedGate{arrived: make(chan int), release: make(chan struct{})}
	var mutex sync.Mutex
	embedCalls := 0
	stop := make(chan struct{})
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		for {
			select {
			case <-stop:
				return
			case <-gate.arrived:
				mutex.Lock()
				embedCalls++
				mutex.Unlock()
				select {
				case gate.release <- struct{}{}:
				case <-stop:
					return
				}
			}
		}
	}()
	t.Cleanup(func() {
		close(stop)
		<-stopped
	})

	harness := newHarnessWithGate(t, gate)
	secondCollectionID := "live-reuse-" + randomID()
	secondCodebase, err := harness.manager.RegisterConversationCollection(
		context.Background(),
		secondCollectionID,
	)
	if err != nil {
		t.Fatalf("RegisterConversationCollection for second corpus returned error: %v", err)
	}
	harness.trackCollectionFamily(secondCodebase.CollectionName)
	secondHarness := *harness
	secondHarness.collectionID = secondCollectionID
	secondHarness.collectionName = secondCodebase.CollectionName
	secondHarness.codebaseID = secondCodebase.ID
	sharedContent := "cross conversation reuse sentinel"
	uniqueContent := "cross conversation unique control"

	first := harness.upsert(
		map[string][]*pb.ConversationDocument{
			"reuse-first": {{
				ConversationId: "reuse-first",
				MessageIndex:   0,
				Role:           "user",
				Text:           sharedContent,
			}},
		},
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		false,
		false,
	)
	requireCompleted(t, first, "first reuse ingest")
	if first.Progress.ChunksEmbedded != 1 {
		t.Fatalf("first ingest embedded = %d, want 1", first.Progress.ChunksEmbedded)
	}

	second := secondHarness.upsert(
		map[string][]*pb.ConversationDocument{
			"reuse-second": {
				{
					ConversationId: "reuse-second",
					MessageIndex:   0,
					Role:           "user",
					Text:           sharedContent,
				},
				{
					ConversationId: "reuse-second",
					MessageIndex:   1,
					Role:           "assistant",
					Text:           uniqueContent,
				},
			},
		},
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		false,
		false,
	)
	requireCompleted(t, second, "second reuse ingest")
	if second.Progress.ChunksReused != 1 {
		t.Fatalf("second ingest reused = %d, want 1", second.Progress.ChunksReused)
	}
	if second.Progress.ChunksEmbedded != 1 {
		t.Fatalf("second ingest embedded = %d, want 1", second.Progress.ChunksEmbedded)
	}
	if count := harness.countRowsContaining(sharedContent); count != 1 {
		t.Fatalf("first corpus rows with shared content = %d, want 1", count)
	}
	if count := secondHarness.countRowsContaining(sharedContent); count != 1 {
		t.Fatalf("second corpus rows with shared content = %d, want 1", count)
	}
	if count := secondHarness.countRowsContaining(uniqueContent); count != 1 {
		t.Fatalf("second corpus rows with unique content = %d, want 1", count)
	}
	catalogCount, err := harness.milvus.Query(
		context.Background(),
		milvusclient.NewQueryOption(harness.reuseCatalogName).
			WithOutputFields(countOutputField).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("count reuse catalog rows: %v", err)
	}
	countColumn := catalogCount.GetColumn(countOutputField)
	if countColumn == nil {
		t.Fatal("reuse catalog count query returned no count column")
	}
	count, err := countColumn.GetAsInt64(0)
	if err != nil {
		t.Fatalf("read reuse catalog count: %v", err)
	}
	if count != 2 {
		t.Fatalf("reuse catalog rows = %d, want 2", count)
	}
	mutex.Lock()
	callCount := embedCalls
	mutex.Unlock()
	if callCount != 2 {
		t.Fatalf("embedding calls = %d, want 2", callCount)
	}
}

func TestUntaggedReuseAcrossCorpusPreservesSourceRow(t *testing.T) {
	harness := newHarness(t)
	seed := harness.upsert(
		map[string][]*pb.ConversationDocument{
			"catalog-seed": {{
				ConversationId: "catalog-seed",
				MessageIndex:   0,
				Role:           "user",
				Text:           "current identity catalog seed",
			}},
		},
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		false,
		false,
	)
	requireCompleted(t, seed, "catalog seed ingest")

	legacyContent := "untagged legacy vector"
	legacyVector := make([]float32, fakeEmbeddingDimension)
	legacyVector[0] = 1
	legacyID := insertLegacyRow(t, harness, legacyContent, legacyVector)
	legacyBefore := snapshotRow(t, harness, harness.collectionName, legacyID)
	if legacyBefore.contentHashKnown || legacyBefore.embeddingModelKnown {
		t.Fatalf("legacy identity = hash:%t model:%t, want both absent", legacyBefore.contentHashKnown, legacyBefore.embeddingModelKnown)
	}
	searchConfig := harness.childConfig()
	searchService, err := semantic.NewService(harness.milvusContext, searchConfig)
	if err != nil {
		t.Fatalf("open search service: %v", err)
	}
	t.Cleanup(func() { _ = searchService.Close(context.Background()) })
	searchResults, err := searchService.SearchConversationCollectionCapped(
		context.Background(),
		harness.collectionName,
		"untagged legacy search probe",
		10,
		10,
		-1,
		semantic.ConversationFilter{},
	)
	if err != nil {
		t.Fatalf("search collection containing untagged row: %v", err)
	}
	if !slices.ContainsFunc(searchResults, func(chunk model.StoredChunk) bool {
		return chunk.Content == legacyContent
	}) {
		t.Fatalf("search results omitted untagged content: %+v", searchResults)
	}

	secondCollectionID := "live-legacy-reuse-" + randomID()
	secondCodebase, err := harness.manager.RegisterConversationCollection(
		context.Background(),
		secondCollectionID,
	)
	if err != nil {
		t.Fatalf("RegisterConversationCollection for second corpus returned error: %v", err)
	}
	harness.trackCollectionFamily(secondCodebase.CollectionName)
	secondHarness := *harness
	secondHarness.collectionID = secondCollectionID
	secondHarness.collectionName = secondCodebase.CollectionName
	secondHarness.codebaseID = secondCodebase.ID

	secondDocuments := map[string][]*pb.ConversationDocument{
		"legacy-second": {{
			ConversationId: "legacy-second",
			MessageIndex:   0,
			Role:           "user",
			Text:           legacyContent,
		}},
	}
	second := secondHarness.upsert(
		secondDocuments,
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		false,
		false,
	)
	requireCompleted(t, second, "second corpus after legacy reuse")
	if second.Progress.ChunksReused != 1 || second.Progress.ChunksEmbedded != 0 {
		t.Fatalf(
			"cross-corpus legacy reused/embedded = %d/%d, want 1/0",
			second.Progress.ChunksReused,
			second.Progress.ChunksEmbedded,
		)
	}
	legacyAfter := snapshotRow(t, harness, harness.collectionName, legacyID)
	if legacyAfter != legacyBefore {
		t.Fatalf("legacy row changed: before=%+v after=%+v", legacyBefore, legacyAfter)
	}
	secondRows := snapshotsForContent(t, harness, secondHarness.collectionName, legacyContent)
	if len(secondRows) != 1 {
		t.Fatalf("second corpus rows = %d, want 1", len(secondRows))
	}
	inserted := secondRows[0]
	if !inserted.contentHashKnown || inserted.contentHash != semantic.ContentVectorKey(legacyContent) {
		t.Fatalf("new row content hash = %q known=%t", inserted.contentHash, inserted.contentHashKnown)
	}
	if !inserted.embeddingModelKnown || inserted.embeddingModel != harness.config.EmbeddingModel {
		t.Fatalf("new row embedding model = %q known=%t", inserted.embeddingModel, inserted.embeddingModelKnown)
	}
	if inserted.vectorChecksum != legacyBefore.vectorChecksum {
		t.Fatalf("new row vector checksum = %s, want legacy %s", inserted.vectorChecksum, legacyBefore.vectorChecksum)
	}
	t.Logf(
		"legacy_id=%s legacy_vector_sha256=%s new_id=%s new_vector_sha256=%s",
		legacyID,
		legacyBefore.vectorChecksum,
		inserted.id,
		inserted.vectorChecksum,
	)

	repeatBefore := snapshotsForContent(t, harness, secondHarness.collectionName, legacyContent)
	repeat := secondHarness.upsert(
		secondDocuments,
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		false,
		false,
	)
	requireCompleted(t, repeat, "repeat legacy reuse")
	if repeat.Progress.FilesModified != 0 || repeat.Progress.FilesEmbedded != 0 ||
		repeat.Progress.ChunksReused != 0 || repeat.Progress.ChunksEmbedded != 0 {
		t.Fatalf("repeat wrote work: %s", progressString(repeat))
	}
	repeatAfter := snapshotsForContent(t, harness, secondHarness.collectionName, legacyContent)
	if !slices.Equal(repeatAfter, repeatBefore) {
		t.Fatalf("repeat changed rows: before=%+v after=%+v", repeatBefore, repeatAfter)
	}

	cfg := harness.childConfig()
	cfg.EmbeddingModel = "known-unequal-model"
	unequalService, err := semantic.NewService(harness.milvusContext, cfg)
	if err != nil {
		t.Fatalf("open unequal-model service: %v", err)
	}
	t.Cleanup(func() { _ = unequalService.Close(context.Background()) })
	untaggedReuse, err := unequalService.LoadReuseVectorsForContents(
		context.Background(),
		harness.collectionName,
		[]model.StoredChunk{{Content: legacyContent}},
	)
	if err != nil {
		t.Fatalf("load untagged reuse under unequal current model: %v", err)
	}
	if len(untaggedReuse) != 1 {
		t.Fatalf("untagged reuse vectors = %d, want 1", len(untaggedReuse))
	}
	knownUnequal, err := unequalService.LoadReuseVectorsForContents(
		context.Background(),
		harness.collectionName,
		[]model.StoredChunk{{Content: "current identity catalog seed"}},
	)
	if err != nil {
		t.Fatalf("load known unequal reuse: %v", err)
	}
	if len(knownUnequal) != 0 {
		t.Fatalf("known unequal reuse vectors = %d, want 0", len(knownUnequal))
	}
	if count := reuseCatalogRowCount(t, harness); count != 1 {
		t.Fatalf("reuse catalog rows after read-only legacy reuse = %d, want 1", count)
	}
}

func TestReuseCatalogStoresEachKnownEmbeddingModel(t *testing.T) {
	harness := newHarness(t)
	content := "two known model catalog sentinel"
	emptyModelContent := "empty model catalog sentinel"
	seed := harness.upsert(
		map[string][]*pb.ConversationDocument{
			"model-a": {{
				ConversationId: "model-a",
				MessageIndex:   0,
				Role:           "user",
				Text:           content,
			}},
		},
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		false,
		false,
	)
	requireCompleted(t, seed, "model A catalog seed")

	cfgA := harness.childConfig()
	serviceA, err := semantic.NewService(harness.milvusContext, cfgA)
	if err != nil {
		t.Fatalf("open model A service: %v", err)
	}
	t.Cleanup(func() { _ = serviceA.Close(context.Background()) })

	cfgB := cfgA
	cfgB.EmbeddingModel = "known-model-b"
	serviceB, err := semantic.NewService(harness.milvusContext, cfgB)
	if err != nil {
		t.Fatalf("open model B service: %v", err)
	}
	t.Cleanup(func() { _ = serviceB.Close(context.Background()) })
	modelBPath := filepath.Join(harness.stateRoot, "model-b-"+randomID())
	modelBCollection := serviceB.CollectionName(modelBPath)
	harness.trackCollectionFamily(modelBCollection)
	if err := serviceB.StageReindex(
		context.Background(),
		modelBPath,
		[]model.StoredChunk{{Content: content, RelativePath: "model-b.txt"}},
		semantic.Removal{},
		nil,
		map[string][]float32{},
		semantic.StoreColumnSetCode,
	); err != nil {
		t.Fatalf("stage model B row: %v", err)
	}
	if err := serviceB.PromoteStaging(context.Background(), modelBPath); err != nil {
		t.Fatalf("promote model B row: %v", err)
	}
	insertEmptyModelCatalogRow(t, harness, emptyModelContent)

	models := reuseCatalogModels(t, harness, content)
	wantModels := []string{"known-model-b", harness.config.EmbeddingModel}
	if !slices.Equal(models, wantModels) {
		t.Fatalf("catalog models = %v, want %v", models, wantModels)
	}
	for name, service := range map[string]*semantic.Service{
		"model A": serviceA,
		"model B": serviceB,
	} {
		reuse, loadErr := service.LoadReuseVectorsForContents(
			context.Background(),
			"",
			[]model.StoredChunk{{Content: content}},
		)
		if loadErr != nil {
			t.Fatalf("load %s catalog reuse: %v", name, loadErr)
		}
		if len(reuse) != 1 {
			t.Fatalf("%s catalog reuse vectors = %d, want 1", name, len(reuse))
		}
	}

	cfgC := cfgA
	cfgC.EmbeddingModel = "known-model-c"
	serviceC, err := semantic.NewService(harness.milvusContext, cfgC)
	if err != nil {
		t.Fatalf("open model C service: %v", err)
	}
	t.Cleanup(func() { _ = serviceC.Close(context.Background()) })
	knownUnequal, err := serviceC.LoadReuseVectorsForContents(
		context.Background(),
		"",
		[]model.StoredChunk{{Content: content}},
	)
	if err != nil {
		t.Fatalf("load model C catalog reuse: %v", err)
	}
	if len(knownUnequal) != 0 {
		t.Fatalf("model C catalog reuse vectors = %d, want 0", len(knownUnequal))
	}
	emptyIdentity, err := serviceC.LoadReuseVectorsForContents(
		context.Background(),
		"",
		[]model.StoredChunk{{Content: emptyModelContent}},
	)
	if err != nil {
		t.Fatalf("load empty model catalog reuse: %v", err)
	}
	if len(emptyIdentity) != 1 {
		t.Fatalf("empty model catalog reuse vectors = %d, want 1", len(emptyIdentity))
	}
}

func TestCompleteCatalogHitSkipsCollectionFallback(t *testing.T) {
	harness := newHarness(t)
	content := "complete catalog hit sentinel"
	seed := harness.upsert(
		map[string][]*pb.ConversationDocument{
			"complete-hit": {{
				ConversationId: "complete-hit",
				MessageIndex:   0,
				Role:           "user",
				Text:           content,
			}},
		},
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		false,
		false,
	)
	requireCompleted(t, seed, "complete catalog hit seed")

	cfg := harness.childConfig()
	cfg.RegistryPath = filepath.Join(t.TempDir(), "missing-registry.json")
	service, err := semantic.NewService(harness.milvusContext, cfg)
	if err != nil {
		t.Fatalf("open complete catalog hit service: %v", err)
	}
	t.Cleanup(func() { _ = service.Close(context.Background()) })

	reuse, err := service.LoadReuseVectorsForContents(
		context.Background(),
		harness.collectionName,
		[]model.StoredChunk{{Content: content}},
	)
	if err != nil {
		t.Fatalf("load complete catalog hit: %v", err)
	}
	if len(reuse) != 1 {
		t.Fatalf("complete catalog hit reuse vectors = %d, want 1", len(reuse))
	}
}

func TestUnknownConfiguredDimensionScopesCatalogByReturnedVectorWidth(t *testing.T) {
	const (
		initialDimension = 1536
		targetDimension  = 4096
	)

	harness := newHarness(t)
	initialServer := newFakeEmbeddingServerWithDimension(t, nil, initialDimension)
	targetServer := newFakeEmbeddingServerWithDimension(t, nil, targetDimension)

	initialConfig := harness.childConfig()
	initialConfig.OpenAIBaseURL = initialServer.URL
	initialConfig.EmbeddingDimension = 0
	targetConfig := initialConfig
	targetConfig.OpenAIBaseURL = targetServer.URL

	initialService, err := semantic.NewService(harness.milvusContext, initialConfig)
	if err != nil {
		t.Fatalf("open initial dimension service: %v", err)
	}
	t.Cleanup(func() { _ = initialService.Close(context.Background()) })
	targetService, err := semantic.NewService(harness.milvusContext, targetConfig)
	if err != nil {
		t.Fatalf("open target dimension service: %v", err)
	}
	t.Cleanup(func() { _ = targetService.Close(context.Background()) })

	configuredCatalogName := semantic.ReuseCatalogCollectionName(initialConfig)
	initialCatalogConfig := initialConfig
	initialCatalogConfig.EmbeddingDimension = initialDimension
	initialCatalogName := semantic.ReuseCatalogCollectionName(initialCatalogConfig)
	targetCatalogConfig := targetConfig
	targetCatalogConfig.EmbeddingDimension = targetDimension
	targetCatalogName := semantic.ReuseCatalogCollectionName(targetCatalogConfig)
	initialPath := filepath.Join(harness.stateRoot, "dimension-1536-"+randomID())
	targetPath := filepath.Join(harness.stateRoot, "dimension-4096-"+randomID())
	sharedContent := "dimension transition shared content"
	newContent := "dimension transition new content"
	for _, collectionName := range []string{
		configuredCatalogName,
		initialCatalogName,
		targetCatalogName,
	} {
		harness.trackTemporaryCollection(collectionName)
	}
	harness.trackCollectionFamily(initialService.CollectionName(initialPath))
	harness.trackCollectionFamily(targetService.CollectionName(targetPath))

	if err := initialService.StageReindex(
		context.Background(),
		initialPath,
		[]model.StoredChunk{{Content: sharedContent, RelativePath: "seed.txt"}},
		semantic.Removal{},
		nil,
		map[string][]float32{},
		semantic.StoreColumnSetCode,
	); err != nil {
		t.Fatalf("stage initial dimension row: %v", err)
	}
	if err := initialService.PromoteStaging(context.Background(), initialPath); err != nil {
		t.Fatalf("promote initial dimension row: %v", err)
	}

	var targetProgress semantic.Progress
	if err := targetService.StageReindex(
		context.Background(),
		targetPath,
		[]model.StoredChunk{
			{Content: sharedContent, RelativePath: "shared.txt"},
			{Content: newContent, RelativePath: "new.txt"},
		},
		semantic.Removal{},
		func(progress semantic.Progress) { targetProgress = progress },
		map[string][]float32{},
		semantic.StoreColumnSetCode,
	); err != nil {
		t.Fatalf("stage target dimension row: %v", err)
	}
	if err := targetService.PromoteStaging(context.Background(), targetPath); err != nil {
		t.Fatalf("promote target dimension row: %v", err)
	}
	if initialCatalogName == targetCatalogName {
		t.Fatal("initial and target dimensions share a reuse catalog")
	}
	for _, collectionName := range []string{initialCatalogName, targetCatalogName} {
		exists, existsErr := harness.milvus.HasCollection(
			context.Background(),
			milvusclient.NewHasCollectionOption(collectionName),
		)
		if existsErr != nil {
			t.Fatalf("check reuse catalog %s: %v", collectionName, existsErr)
		}
		if !exists {
			t.Fatalf("reuse catalog %s does not exist", collectionName)
		}
	}
	configuredCatalogExists, err := harness.milvus.HasCollection(
		context.Background(),
		milvusclient.NewHasCollectionOption(configuredCatalogName),
	)
	if err != nil {
		t.Fatalf("check configured-zero reuse catalog: %v", err)
	}
	if configuredCatalogExists {
		t.Fatalf("configured-zero reuse catalog %s exists", configuredCatalogName)
	}
	if targetProgress.ChunksEmbedded != 2 || targetProgress.ChunksReused != 0 {
		t.Fatalf(
			"target embedded/reused = %d/%d, want 2/0",
			targetProgress.ChunksEmbedded,
			targetProgress.ChunksReused,
		)
	}

	targetCollectionName := targetService.CollectionName(targetPath)
	targetRows := snapshotsForContent(t, harness, targetCollectionName, sharedContent)
	if len(targetRows) != 1 {
		t.Fatalf("target dimension rows = %d, want 1", len(targetRows))
	}
	result, err := harness.milvus.Query(
		context.Background(),
		milvusclient.NewQueryOption(targetCollectionName).
			WithFilter(fmt.Sprintf(`content == "%s"`, sharedContent)).
			WithOutputFields("vector").
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("query target dimension vector: %v", err)
	}
	vectorColumn := result.GetColumn("vector")
	if vectorColumn == nil || result.ResultCount != 1 {
		t.Fatalf("target dimension vector rows = %d, want 1", result.ResultCount)
	}
	vectorValue, err := vectorColumn.Get(0)
	if err != nil {
		t.Fatalf("read target dimension vector: %v", err)
	}
	vector, ok := vectorValue.(entity.FloatVector)
	if !ok {
		t.Fatalf("target dimension vector has type %T", vectorValue)
	}
	if len(vector) != targetDimension {
		t.Fatalf("target vector dimension = %d, want %d", len(vector), targetDimension)
	}
}

func insertLegacyRow(
	t *testing.T,
	harness *harness,
	content string,
	vector []float32,
) string {
	t.Helper()
	rowID := "legacy-" + randomID()
	result, err := harness.milvus.Insert(
		context.Background(),
		milvusclient.NewColumnBasedInsertOption(harness.collectionName).
			WithVarcharColumn("id", []string{rowID}).
			WithVarcharColumn("content", []string{content}).
			WithVarcharColumn("relativePath", []string{"conv/legacy/0/0"}).
			WithInt64Column("startLine", []int64{0}).
			WithInt64Column("endLine", []int64{0}).
			WithVarcharColumn("fileExtension", []string{"txt"}).
			WithVarcharColumn("metadata", []string{"{}"}).
			WithFloatVectorColumn("vector", len(vector), [][]float32{vector}),
	)
	if err != nil {
		t.Fatalf("insert legacy row: %v", err)
	}
	if result.InsertCount != 1 {
		t.Fatalf("insert legacy row count = %d, want 1", result.InsertCount)
	}
	flushTask, err := harness.milvus.Flush(
		context.Background(),
		milvusclient.NewFlushOption(harness.collectionName),
	)
	if err != nil {
		t.Fatalf("flush legacy row: %v", err)
	}
	if err := flushTask.Await(context.Background()); err != nil {
		t.Fatalf("await legacy row flush: %v", err)
	}
	return rowID
}

type storedRowSnapshot struct {
	id                  string
	content             string
	relativePath        string
	startLine           int64
	endLine             int64
	fileExtension       string
	metadata            string
	contentHash         string
	contentHashKnown    bool
	embeddingModel      string
	embeddingModelKnown bool
	vectorChecksum      string
}

func snapshotRow(
	t *testing.T,
	harness *harness,
	collectionName string,
	rowID string,
) storedRowSnapshot {
	t.Helper()
	rows := queryRowSnapshots(
		t,
		harness,
		collectionName,
		fmt.Sprintf(`id == "%s"`, rowID),
	)
	if len(rows) != 1 {
		t.Fatalf("row %s snapshots = %d, want 1", rowID, len(rows))
	}
	return rows[0]
}

func snapshotsForContent(
	t *testing.T,
	harness *harness,
	collectionName string,
	content string,
) []storedRowSnapshot {
	t.Helper()
	return queryRowSnapshots(
		t,
		harness,
		collectionName,
		fmt.Sprintf(`content == "%s"`, strings.ReplaceAll(content, `"`, `\"`)),
	)
}

func queryRowSnapshots(
	t *testing.T,
	harness *harness,
	collectionName string,
	filter string,
) []storedRowSnapshot {
	t.Helper()
	result, err := harness.milvus.Query(
		context.Background(),
		milvusclient.NewQueryOption(collectionName).
			WithFilter(filter).
			WithOutputFields(
				"id",
				"content",
				"relativePath",
				"startLine",
				"endLine",
				"fileExtension",
				"metadata",
				"contentHash",
				"embeddingModel",
				"vector",
			).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("query row snapshots from %s: %v", collectionName, err)
	}
	idColumn := result.GetColumn("id")
	contentColumn := result.GetColumn("content")
	pathColumn := result.GetColumn("relativePath")
	startLineColumn := result.GetColumn("startLine")
	endLineColumn := result.GetColumn("endLine")
	fileExtensionColumn := result.GetColumn("fileExtension")
	metadataColumn := result.GetColumn("metadata")
	hashColumn := result.GetColumn("contentHash")
	modelColumn := result.GetColumn("embeddingModel")
	vectorColumn := result.GetColumn("vector")
	if idColumn == nil || contentColumn == nil || pathColumn == nil ||
		startLineColumn == nil || endLineColumn == nil || fileExtensionColumn == nil ||
		metadataColumn == nil || vectorColumn == nil {
		t.Fatal("row snapshot query omitted a required column")
	}
	rows := make([]storedRowSnapshot, 0, result.ResultCount)
	for rowIndex := range result.ResultCount {
		rowID, idErr := idColumn.GetAsString(rowIndex)
		if idErr != nil {
			t.Fatalf("read row id at %d: %v", rowIndex, idErr)
		}
		content, contentErr := contentColumn.GetAsString(rowIndex)
		if contentErr != nil {
			t.Fatalf("read row content at %d: %v", rowIndex, contentErr)
		}
		relativePath, pathErr := pathColumn.GetAsString(rowIndex)
		if pathErr != nil {
			t.Fatalf("read row path at %d: %v", rowIndex, pathErr)
		}
		startLine, startLineErr := startLineColumn.GetAsInt64(rowIndex)
		if startLineErr != nil {
			t.Fatalf("read row start line at %d: %v", rowIndex, startLineErr)
		}
		endLine, endLineErr := endLineColumn.GetAsInt64(rowIndex)
		if endLineErr != nil {
			t.Fatalf("read row end line at %d: %v", rowIndex, endLineErr)
		}
		fileExtension, fileExtensionErr := fileExtensionColumn.GetAsString(rowIndex)
		if fileExtensionErr != nil {
			t.Fatalf("read row file extension at %d: %v", rowIndex, fileExtensionErr)
		}
		metadata, metadataErr := metadataColumn.GetAsString(rowIndex)
		if metadataErr != nil {
			t.Fatalf("read row metadata at %d: %v", rowIndex, metadataErr)
		}
		contentHash, contentHashKnown := nullableSnapshotString(t, hashColumn, rowIndex)
		embeddingModel, embeddingModelKnown := nullableSnapshotString(t, modelColumn, rowIndex)
		vectorValue, vectorErr := vectorColumn.Get(rowIndex)
		if vectorErr != nil {
			t.Fatalf("read row vector at %d: %v", rowIndex, vectorErr)
		}
		vector, ok := vectorValue.(entity.FloatVector)
		if !ok {
			t.Fatalf("row vector at %d has type %T", rowIndex, vectorValue)
		}
		rows = append(rows, storedRowSnapshot{
			id:                  rowID,
			content:             content,
			relativePath:        relativePath,
			startLine:           startLine,
			endLine:             endLine,
			fileExtension:       fileExtension,
			metadata:            metadata,
			contentHash:         contentHash,
			contentHashKnown:    contentHashKnown,
			embeddingModel:      embeddingModel,
			embeddingModelKnown: embeddingModelKnown,
			vectorChecksum:      checksumVector(vector),
		})
	}
	slices.SortFunc(rows, func(left storedRowSnapshot, right storedRowSnapshot) int {
		return strings.Compare(left.id, right.id)
	})
	return rows
}

func nullableSnapshotString(
	t *testing.T,
	field interface {
		IsNull(int) (bool, error)
		GetAsString(int) (string, error)
	},
	rowIndex int,
) (string, bool) {
	t.Helper()
	if field == nil {
		return "", false
	}
	isNull, err := field.IsNull(rowIndex)
	if err != nil {
		t.Fatalf("read nullable marker at %d: %v", rowIndex, err)
	}
	if isNull {
		return "", false
	}
	value, err := field.GetAsString(rowIndex)
	if err != nil {
		t.Fatalf("read nullable string at %d: %v", rowIndex, err)
	}
	return value, true
}

func checksumVector(vector entity.FloatVector) string {
	hash := sha256.New()
	buffer := make([]byte, 4)
	for _, value := range vector {
		binary.LittleEndian.PutUint32(buffer, math.Float32bits(value))
		_, _ = hash.Write(buffer)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func reuseCatalogRowCount(t *testing.T, harness *harness) int64 {
	t.Helper()
	result, err := harness.milvus.Query(
		context.Background(),
		milvusclient.NewQueryOption(harness.reuseCatalogName).
			WithOutputFields(countOutputField).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("count reuse catalog rows: %v", err)
	}
	countColumn := result.GetColumn(countOutputField)
	if countColumn == nil {
		t.Fatal("reuse catalog count query returned no count column")
	}
	count, err := countColumn.GetAsInt64(0)
	if err != nil {
		t.Fatalf("read reuse catalog count: %v", err)
	}
	return count
}

func reuseCatalogModels(t *testing.T, harness *harness, content string) []string {
	t.Helper()
	contentHash := semantic.ContentVectorKey(content)
	result, err := harness.milvus.Query(
		context.Background(),
		milvusclient.NewQueryOption(harness.reuseCatalogName).
			WithFilter(fmt.Sprintf(`contentHash == "%s"`, contentHash)).
			WithOutputFields("embeddingModel").
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("query reuse catalog models: %v", err)
	}
	modelColumn := result.GetColumn("embeddingModel")
	if modelColumn == nil && result.ResultCount > 0 {
		t.Fatal("reuse catalog model query returned no model column")
	}
	models := make([]string, 0, result.ResultCount)
	for rowIndex := range result.ResultCount {
		modelName, known := nullableSnapshotString(t, modelColumn, rowIndex)
		if !known {
			models = append(models, "")
			continue
		}
		models = append(models, modelName)
	}
	slices.Sort(models)
	return models
}

func insertEmptyModelCatalogRow(t *testing.T, harness *harness, content string) {
	t.Helper()
	contentHash := semantic.ContentVectorKey(content)
	rowKeySum := sha256.Sum256([]byte(contentHash + "\x00"))
	rowKey := hex.EncodeToString(rowKeySum[:])
	embeddingModelColumn, err := column.NewNullableColumnVarChar(
		"embeddingModel",
		[]string{""},
		[]bool{false},
		column.WithSparseNullableMode[string](true),
	)
	if err != nil {
		t.Fatalf("build empty model catalog column: %v", err)
	}
	vector := make([]float32, fakeEmbeddingDimension)
	vector[0] = 1
	result, err := harness.milvus.Insert(
		context.Background(),
		milvusclient.NewColumnBasedInsertOption(harness.reuseCatalogName).
			WithVarcharColumn("catalogKey", []string{rowKey}).
			WithVarcharColumn("contentHash", []string{contentHash}).
			WithColumns(embeddingModelColumn).
			WithFloatVectorColumn("vector", len(vector), [][]float32{vector}),
	)
	if err != nil {
		t.Fatalf("insert empty model catalog row: %v", err)
	}
	if result.InsertCount != 1 {
		t.Fatalf("empty model catalog insert count = %d, want 1", result.InsertCount)
	}
	flushTask, err := harness.milvus.Flush(
		context.Background(),
		milvusclient.NewFlushOption(harness.reuseCatalogName),
	)
	if err != nil {
		t.Fatalf("flush empty model catalog row: %v", err)
	}
	if err := flushTask.Await(context.Background()); err != nil {
		t.Fatalf("await empty model catalog flush: %v", err)
	}
}

func dropLiveCollection(t *testing.T, harness *harness, collectionName string) {
	t.Helper()
	dropCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := harness.milvus.DropCollection(
		dropCtx,
		milvusclient.NewDropCollectionOption(collectionName),
	); err != nil &&
		!strings.Contains(err.Error(), "not exist") &&
		!strings.Contains(err.Error(), "not found") {
		t.Errorf("DropCollection(%s) returned error: %v", collectionName, err)
	}
}

func TestCorpusReuseLookupP95BelowConfiguredEmbedding(t *testing.T) {
	if os.Getenv("LMS_CORPUS_REUSE_PERF") != "1" {
		t.Skip("set LMS_CORPUS_REUSE_PERF=1 to measure the configured embedder")
	}
	actualConfig, err := config.Default()
	if err != nil {
		t.Fatalf("load configured embedder: %v", err)
	}
	provider, err := embedding.NewProvider(context.Background(), actualConfig)
	if err != nil {
		t.Fatalf("create configured embedder: %v", err)
	}

	harness := newHarness(t)
	lookupConfig := harness.childConfig()
	const sampleCount = 20
	contents := make([]string, 0, sampleCount)
	documents := make([]*pb.ConversationDocument, 0, sampleCount)
	for index := range sampleCount {
		content := fmt.Sprintf("harmless corpus reuse performance control %02d", index)
		contents = append(contents, content)
		documents = append(documents, &pb.ConversationDocument{
			ConversationId: "reuse-performance",
			MessageIndex:   int32(index),
			Role:           "user",
			Text:           content,
		})
	}
	completed := harness.upsert(
		map[string][]*pb.ConversationDocument{"reuse-performance": documents},
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		false,
		false,
	)
	requireCompleted(t, completed, "reuse performance seed")

	service, err := semantic.NewService(harness.milvusContext, lookupConfig)
	if err != nil {
		t.Fatalf("open semantic service for lookup measurement: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := service.Close(context.Background()); closeErr != nil {
			t.Errorf("close lookup measurement service: %v", closeErr)
		}
	})

	if _, err := provider.EmbedBatch(context.Background(), []string{"harmless corpus reuse warmup"}); err != nil {
		t.Fatalf("warm configured embedder: %v", err)
	}
	if _, err := service.LoadReuseVectorsForContents(
		context.Background(),
		harness.collectionName,
		[]model.StoredChunk{{Content: contents[0]}},
	); err != nil {
		t.Fatalf("warm candidate lookup: %v", err)
	}

	lookupDurations := make([]time.Duration, 0, sampleCount)
	embedDurations := make([]time.Duration, 0, sampleCount)
	for _, content := range contents {
		candidate := []model.StoredChunk{{Content: content}}
		lookupStarted := time.Now()
		reuse, lookupErr := service.LoadReuseVectorsForContents(
			context.Background(),
			harness.collectionName,
			candidate,
		)
		lookupDurations = append(lookupDurations, time.Since(lookupStarted))
		if lookupErr != nil {
			t.Fatalf("candidate lookup for %q: %v", content, lookupErr)
		}
		if len(reuse) != 1 {
			t.Fatalf("candidate lookup for %q returned %d vectors, want 1", content, len(reuse))
		}

		embedStarted := time.Now()
		result, embedErr := provider.EmbedBatch(context.Background(), []string{content})
		embedDurations = append(embedDurations, time.Since(embedStarted))
		if embedErr != nil {
			t.Fatalf("embed performance control %q: %v", content, embedErr)
		}
		if len(result.Vectors) != 1 || len(result.Vectors[0]) == 0 {
			t.Fatalf("embed performance control %q returned no vector", content)
		}
	}

	lookupP95 := durationPercentile(lookupDurations, 95)
	embedP95 := durationPercentile(embedDurations, 95)
	ratio := float64(lookupP95) / float64(embedP95)
	t.Logf(
		"lookup_ms min=%.3f p50=%.3f p95=%.3f max=%.3f; embed_ms min=%.3f p50=%.3f p95=%.3f max=%.3f; ratio=%.4f",
		durationMilliseconds(durationPercentile(lookupDurations, 0)),
		durationMilliseconds(durationPercentile(lookupDurations, 50)),
		durationMilliseconds(lookupP95),
		durationMilliseconds(durationPercentile(lookupDurations, 100)),
		durationMilliseconds(durationPercentile(embedDurations, 0)),
		durationMilliseconds(durationPercentile(embedDurations, 50)),
		durationMilliseconds(embedP95),
		durationMilliseconds(durationPercentile(embedDurations, 100)),
		ratio,
	)
	if lookupP95*10 >= embedP95 {
		t.Fatalf("candidate lookup p95 %s is %.2f%% of embedding p95 %s, want below 10%%", lookupP95, ratio*100, embedP95)
	}
}

func durationPercentile(values []time.Duration, percentile int) time.Duration {
	sorted := append([]time.Duration(nil), values...)
	slices.Sort(sorted)
	if percentile <= 0 {
		return sorted[0]
	}
	index := (len(sorted)*percentile+99)/100 - 1
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

func durationMilliseconds(duration time.Duration) float64 {
	return float64(duration) / float64(time.Millisecond)
}
