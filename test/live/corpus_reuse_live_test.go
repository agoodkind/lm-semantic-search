//go:build live

package live

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

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
	secondHarness := *harness
	secondHarness.collectionID = secondCollectionID
	secondHarness.collectionName = secondCodebase.CollectionName
	secondHarness.codebaseID = secondCodebase.ID
	t.Cleanup(func() {
		dropCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if dropErr := harness.milvus.DropCollection(
			dropCtx,
			milvusclient.NewDropCollectionOption(secondHarness.collectionName),
		); dropErr != nil &&
			!strings.Contains(dropErr.Error(), "not exist") &&
			!strings.Contains(dropErr.Error(), "not found") {
			t.Errorf(
				"DropCollection(%s) returned error: %v",
				secondHarness.collectionName,
				dropErr,
			)
		}
	})
	sharedContent := "cross conversation reuse sentinel"

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
			"reuse-second": {{
				ConversationId: "reuse-second",
				MessageIndex:   0,
				Role:           "user",
				Text:           sharedContent,
			}},
		},
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		false,
		false,
	)
	requireCompleted(t, second, "second reuse ingest")
	if second.Progress.ChunksReused != 1 {
		t.Fatalf("second ingest reused = %d, want 1", second.Progress.ChunksReused)
	}
	if second.Progress.ChunksEmbedded != 0 {
		t.Fatalf("second ingest embedded = %d, want 0", second.Progress.ChunksEmbedded)
	}
	if count := harness.countRowsContaining(sharedContent); count != 1 {
		t.Fatalf("first corpus rows with shared content = %d, want 1", count)
	}
	if count := secondHarness.countRowsContaining(sharedContent); count != 1 {
		t.Fatalf("second corpus rows with shared content = %d, want 1", count)
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
	if count != 1 {
		t.Fatalf("reuse catalog rows = %d, want 1", count)
	}
	mutex.Lock()
	callCount := embedCalls
	mutex.Unlock()
	if callCount != 1 {
		t.Fatalf("embedding calls = %d, want 1", callCount)
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
	lookupConfig, err := config.Default()
	if err != nil {
		t.Fatalf("load isolated lookup config: %v", err)
	}
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

	service, err := semantic.NewService(context.Background(), lookupConfig)
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
