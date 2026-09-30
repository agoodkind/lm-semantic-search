//go:build live

package live

import (
	"context"
	"os"
	"sync"
	"testing"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/embedding"
	"goodkind.io/lm-semantic-search/library/milvus"
	"goodkind.io/lm-semantic-search/library/observation"
)

type operationEvents struct {
	mutex  sync.Mutex
	events []observation.Event
}

func (collector *operationEvents) Observe(event observation.Event) {
	collector.mutex.Lock()
	defer collector.mutex.Unlock()
	collector.events = append(collector.events, event)
}

func (collector *operationEvents) completed(t *testing.T, runID string) []observation.Event {
	t.Helper()
	collector.mutex.Lock()
	defer collector.mutex.Unlock()
	starts := make(map[uint64]observation.Event)
	completed := make([]observation.Event, 0)
	for _, event := range collector.events {
		if event.Scope.RunID != runID {
			continue
		}
		if event.Scope.ProcessID != os.Getpid() || event.Scope.Generation != 1 {
			t.Fatalf("incorrect run metadata: %+v", event.Scope)
		}
		switch event.Phase {
		case observation.Started:
			if _, exists := starts[event.Scope.OperationID]; exists {
				t.Fatalf("duplicate operation start: %+v", event)
			}
			starts[event.Scope.OperationID] = event
		case observation.Completed:
			start, exists := starts[event.Scope.OperationID]
			if !exists || start.Operation != event.Operation {
				t.Fatalf("completion without matching start: %+v", event)
			}
			if event.Duration < 0 || event.Outcome == "" {
				t.Fatalf("invalid operation completion: %+v", event)
			}
			delete(starts, event.Scope.OperationID)
			completed = append(completed, event)
		}
	}
	if len(starts) != 0 || len(completed) == 0 {
		t.Fatalf("run %s has incomplete observation: %d starts, %d completions", runID, len(starts), len(completed))
	}
	return completed
}

func observedRunContext(ctx context.Context, runID string) context.Context {
	return observation.WithScope(ctx, observation.Scope{RunID: runID, Generation: 1, ProcessID: os.Getpid(), Purpose: observation.Ingestion})
}

func TestLibraryObservationReportsRealWritesReuseAndUnchangedPass(t *testing.T) {
	harness := newLibraryHarness(t)
	collector := &operationEvents{}
	ctx := harness.context()
	environment := harness.environment
	embedder, err := embedding.NewOpenAI(ctx, embedding.OpenAIConfig{
		Observer: collector, BaseURL: environment.EmbeddingURL, APIKey: environment.APIKey,
		Model: environment.EmbeddingModel, Dimension: libraryLiveDimension,
		RequestTimeout: libraryEmbeddingRequestTimeout,
	})
	if err != nil {
		t.Fatalf("construct observed embedder: %v", err)
	}
	const collection = "observed_pool"
	vectors, err := milvus.New(harness.milvus, milvus.Config{Observer: collector, Database: harness.database, Collection: collection})
	if err != nil {
		t.Fatalf("construct observed vector store: %v", err)
	}
	descriptor := harness.descriptor("observed")
	opened, err := library.Open(ctx, library.Config{Observer: collector, Store: descriptor, Vectors: vectors, Embedder: embedder})
	if err != nil {
		t.Fatalf("open observed library: %v", err)
	}
	t.Cleanup(func() {
		if err := opened.Close(); err != nil {
			t.Errorf("close observed library: %v", err)
		}
	})
	if err := opened.RegisterNamespace(ctx, conversationSpec()); err != nil {
		t.Fatalf("register observed namespace: %v", err)
	}
	first := appendBatch("observed-owner", 1, "observed-first", messageRow("one", "The archive contains a verified checksum.", 1), messageRow("two", "The archive contains a verified checksum.", 2))
	receipt := mustApply(t, observedRunContext(ctx, "first"), opened, first)
	assertCatalogConsistent(t, harness, descriptor, collection)
	requireFirstObservedPass(t, collector.completed(t, "first"))
	reuse := appendBatch("another-owner", 1, "reuse", messageRow("three", first.Rows[0].EmbeddingInput, 3))
	mustApply(t, observedRunContext(ctx, "reuse"), opened, reuse)
	requireObservedReuse(t, collector.completed(t, "reuse"))
	replayed := mustApply(t, observedRunContext(ctx, "unchanged"), opened, first)
	if replayed != receipt {
		t.Fatalf("unchanged public receipt differs: %+v, want %+v", replayed, receipt)
	}
	requireObservedUnchangedPass(t, collector.completed(t, "unchanged"))
	requireObservedPublicFailures(t, harness, opened, collector, collection)
}

func requireObservedStageOutcome(t *testing.T, events []observation.Event, outcome observation.Outcome) {
	t.Helper()
	for _, event := range events {
		if event.Operation == observation.Stage {
			if event.Outcome != outcome {
				t.Fatalf("stage outcome = %s, want %s", event.Outcome, outcome)
			}
			return
		}
	}
	t.Fatal("public stage lacks its completion event")
}

type firstPassObservations struct {
	requests, writes, duplicateInputs, validated int
	admission, transaction, verification, stage  bool
}

func requireFirstObservedPass(t *testing.T, firstEvents []observation.Event) {
	t.Helper()
	got := firstPassObservations{}
	for _, event := range firstEvents {
		if event.Outcome != observation.Success || event.Scope.Purpose != observation.Ingestion {
			t.Fatalf("first pass failed: %+v", event)
		}
		switch event.Operation {
		case observation.EmbeddingAttempt:
			got.requests++
			if event.Data.Embedding.Requested != 1 || event.Data.Embedding.Returned != 1 {
				t.Fatalf("actual embedding counts: %+v", event)
			}
		case observation.EmbeddingValidation:
			if event.Data.Embedding.Validation == observation.CanonicalValidation {
				got.validated += event.Data.Embedding.Validated
			}
		case observation.UpsertCall:
			got.writes++
			if event.Data.Vector.Requested != 1 || event.Data.Vector.Acknowledged != 1 {
				t.Fatalf("actual upsert counts: %+v", event)
			}
		case observation.IdentitySelection:
			got.duplicateInputs += event.Data.Identity.DuplicateInputs
		case observation.WriterAdmission:
			got.admission = true
		case observation.CatalogTransaction:
			got.transaction = event.Data.Transaction.Boundary == observation.TransactionCommit
		case observation.StrongVerification:
			got.verification = event.Data.Vector.Verified > 0
		case observation.Stage:
			got.stage = !event.Data.Stage.CommittedReceipt && event.Data.Stage.Rows == 2
		}
	}
	want := firstPassObservations{requests: 1, writes: 1, duplicateInputs: 1, validated: 1, admission: true, transaction: true, verification: true, stage: true}
	if got != want {
		t.Fatalf("first pass observations = %+v, want %+v", got, want)
	}
}

func requireObservedReuse(t *testing.T, events []observation.Event) {
	t.Helper()
	reused := 0
	for _, event := range events {
		if event.Operation == observation.EmbeddingAttempt || event.Operation == observation.UpsertCall {
			t.Fatalf("verified catalog reuse performed a request: %+v", event)
		}
		if event.Operation == observation.IdentitySelection && !event.Data.Identity.SecondLookup {
			reused += event.Data.Identity.VerifiedReuse
		}
	}
	if reused != 1 {
		t.Fatalf("verified catalog reuse count = %d, want 1", reused)
	}
}

func requireObservedUnchangedPass(t *testing.T, events []observation.Event) {
	t.Helper()
	completedStage := false
	for _, event := range events {
		if event.Operation == observation.EmbeddingAttempt || event.Operation == observation.UpsertCall {
			t.Fatalf("unchanged pass performed a request: %+v", event)
		}
		if event.Operation == observation.Stage {
			completedStage = event.Data.Stage.CommittedReceipt && event.Outcome == observation.Success
		}
	}
	if !completedStage {
		t.Fatal("unchanged pass lacks the committed-receipt completion")
	}
}

func requireObservedPublicFailures(t *testing.T, harness *libraryHarness, opened *library.Library, collector *operationEvents, collection string) {
	t.Helper()
	ctx := harness.context()
	cancelled, cancel := context.WithCancel(observedRunContext(ctx, "cancelled"))
	cancel()
	if _, err := opened.Apply(cancelled, appendBatch("cancelled-owner", 1, "cancelled", messageRow("cancelled", "cancelled request", 1))); err == nil {
		t.Fatal("cancelled public apply succeeded")
	}
	requireObservedStageOutcome(t, collector.completed(t, "cancelled"), observation.Cancelled)
	if err := harness.milvus.DropCollection(ctx, milvusclient.NewDropCollectionOption(collection)); err != nil {
		t.Fatalf("drop isolated collection: %v", err)
	}
	failedBatch := appendBatch("failed-owner", 1, "failed", messageRow("failed", "A different archive must reject a missing vector pool.", 1))
	if _, err := opened.Apply(observedRunContext(ctx, "backend-failure"), failedBatch); err == nil {
		t.Fatal("public apply succeeded without its vector collection")
	}
	failureEvents := collector.completed(t, "backend-failure")
	requireObservedStageOutcome(t, failureEvents, observation.Failure)
	upsertFailed := false
	for _, event := range failureEvents {
		if event.Operation == observation.UpsertCall {
			upsertFailed = event.Outcome == observation.Failure && event.Data.Vector.Acknowledged == 0
		}
	}
	if !upsertFailed {
		t.Fatal("real failed SDK upsert lacks a failed zero-acknowledgment event")
	}
	if _, err := opened.Apply(observedRunContext(ctx, "pending-retry"), failedBatch); err == nil {
		t.Fatal("pending retry succeeded without its vector collection")
	}
	requireObservedPendingReuse(t, collector.completed(t, "pending-retry"))
}

func requireObservedPendingReuse(t *testing.T, events []observation.Event) {
	t.Helper()
	requireObservedStageOutcome(t, events, observation.Failure)
	reused := 0
	for _, event := range events {
		if event.Operation == observation.EmbeddingAttempt {
			t.Fatalf("pending vector retry embedded its saved input: %+v", event)
		}
		if event.Operation == observation.IdentitySelection && !event.Data.Identity.SecondLookup {
			reused += event.Data.Identity.PendingReuse
		}
	}
	if reused != 1 {
		t.Fatalf("pending catalog reuse count = %d, want 1", reused)
	}
}
