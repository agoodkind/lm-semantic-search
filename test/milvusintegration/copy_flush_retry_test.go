//go:build milvusintegration

package milvusintegration

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v3/entity"
	"github.com/milvus-io/milvus/client/v3/milvusclient"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

const (
	flushRetryMilvusMemory = "4g"
	// flushRetryExpectedAttempts repeats the value of flushRateLimitMaxAttempts in
	// internal/semantic/copy_chunks.go. The test package cannot read that
	// unexported constant. Changing the production value makes
	// TestCopyChunksFailsAfterEveryFlushAttemptIsRateLimited fail.
	flushRetryExpectedAttempts = 8
	// flushRetryMinimumWait totals the seven waits of 2 seconds that requestFlush
	// performs between eight attempts. A shorter elapsed time means CopyChunks
	// waited fewer times than required.
	flushRetryMinimumWait = 14 * time.Second
	// flushRetryCancelAfterAttempts sets the number of Flush requests the
	// recorder must observe before the second test waits flushRetryCancelDelay
	// and cancels the context. Milvus rejects that request at once. The test
	// cancels during the 2 second wait before the next attempt.
	flushRetryCancelAfterAttempts = 2
	flushRetryCancelDelay         = time.Second
	// flushRetryPromptReturn sets the longest time the second test allows between
	// context cancellation and the return of CopyChunks. This limit is shorter
	// than one 2 second retry wait.
	flushRetryPromptReturn    = time.Second
	flushRetryAttemptWait     = 2 * time.Minute
	flushRetryAttemptPoll     = 25 * time.Millisecond
	flushRetryRows            = 3
	flushRetrySourcePath      = "retry/source.txt"
	flushRetryDestinationPath = "retry/destination.txt"
	flushRetryContentFormat   = "flush retry source content part %d"
	flushRetryCodebaseName    = "flush-retry-codebase"
	flushRetryRateLimitText   = "rate limit exceeded"
	flushMethod               = "Flush"
	deleteMethod              = "Delete"
	// flushRequestAnyCollection is the empty collection name filter that
	// recorder.count uses to match every Flush request. The Milvus FlushRequest
	// lists its collections in a repeated field and does not have a
	// GetCollectionName method; the recorder does not store a collection name for
	// Flush requests, and a nonempty collection name filter does not match them.
	// Only the service under test sends requests through the recorded client.
	flushRequestAnyCollection = ""
)

// flushRetryMilvusConfig supplies the throwaway stack's Milvus user.yaml and
// limits each collection to 0.01 flushes per second. After one flush, Milvus
// rejects later flushes for that collection with rate limit exceeded for longer
// than the 14 second retry window of requestFlush. The Milvus default of 0.1
// flushes per second permits another flush after 10 seconds, within that
// window.
const flushRetryMilvusConfig = `quotaAndLimits:
  flushRate:
    enabled: true
    collection:
      max: 0.01
`

// flushRetryFixture provides a production semantic.Service connected to a
// throwaway Milvus stack, a recorder for the service's Milvus requests, and a
// test-owned Milvus client. Its collection has used its one allowed flush.
// sourceRows records the number of rows under flushRetrySourcePath before
// CopyChunks runs.
type flushRetryFixture struct {
	service        *semantic.Service
	recorder       *milvusCallRecorder
	milvus         *milvusclient.Client
	codebasePath   string
	collectionName string
	sourceRows     int64
}

// TestCopyChunksFailsAfterEveryFlushAttemptIsRateLimited calls CopyChunks after
// the collection uses its allowed flush, and Milvus rejects every flush attempt
// with a rate limit error. The test requires CopyChunks to return that error
// after exactly eight Flush requests and at least 14 seconds. It also checks
// that CopyChunks does not send a Delete request and that every source row
// remains in the collection.
func TestCopyChunksFailsAfterEveryFlushAttemptIsRateLimited(t *testing.T) {
	requireIntegration(t)
	fixture := startFlushRetryFixture(t)

	fixture.recorder.reset()
	startedAt := time.Now()
	copied, err := fixture.service.CopyChunks(context.Background(), fixture.codebasePath, flushRetryCopies())
	elapsed := time.Since(startedAt)
	attempts := fixture.recorder.count(flushMethod, flushRequestAnyCollection)
	t.Logf("CopyChunks sent %d Flush requests, returned after %s, and returned the error %v.", attempts, elapsed.Round(time.Millisecond), err)

	if err == nil {
		t.Fatalf("CopyChunks succeeded with copied row counts %v, but the test expected an error because Milvus rejects every flush with a rate limit error.", copied)
	}
	if !strings.Contains(strings.ToLower(err.Error()), flushRetryRateLimitText) {
		t.Fatalf("CopyChunks returned the error %v, but the test expected an error containing the Milvus text rate limit exceeded.", err)
	}
	if attempts != flushRetryExpectedAttempts {
		t.Fatalf("CopyChunks sent %d Flush requests, but the test expected %d.", attempts, flushRetryExpectedAttempts)
	}
	if elapsed < flushRetryMinimumWait {
		t.Fatalf("CopyChunks returned after %s, but the test expected at least %s of retry waits.", elapsed.Round(time.Millisecond), flushRetryMinimumWait)
	}
	if len(copied) != 1 || copied[0] != flushRetryRows {
		t.Fatalf("CopyChunks acknowledged row counts %v, want [%d] despite the later flush failure", copied, flushRetryRows)
	}
	if destinationRows := fixture.pathRowCount(t, flushRetryDestinationPath); destinationRows != flushRetryRows {
		t.Fatalf("Milvus stored %d acknowledged destination rows, want %d", destinationRows, flushRetryRows)
	}
	fixture.assertSourceRowsKept(t)
}

// TestCopyChunksStopsFlushRetryWhenContextEnds cancels the context of CopyChunks
// during a rate limit wait. The test requires CopyChunks to return an error that
// wraps context.Canceled within flushRetryPromptReturn and after fewer than
// eight Flush requests. It also checks that CopyChunks does not send a Delete
// request and that every source row remains in the collection.
func TestCopyChunksStopsFlushRetryWhenContextEnds(t *testing.T) {
	requireIntegration(t)
	fixture := startFlushRetryFixture(t)

	fixture.recorder.reset()
	copyContext, cancel := context.WithCancel(context.Background())
	defer cancel()
	results := make(chan copyChunksResult, 1)
	go func() {
		copied, err := fixture.service.CopyChunks(copyContext, fixture.codebasePath, flushRetryCopies())
		results <- copyChunksResult{copied: copied, err: err, returnedAt: time.Now()}
	}()

	fixture.waitForFlushAttempts(t, flushRetryCancelAfterAttempts, results)
	// This sleep delays context cancellation until partway through the 2 second
	// retry wait after the rejected Flush request.
	time.Sleep(flushRetryCancelDelay)
	cancelledAt := time.Now()
	cancel()

	var result copyChunksResult
	select {
	case result = <-results:
	case <-time.After(flushRetryMinimumWait):
		t.Fatalf("CopyChunks did not return within %s after the test canceled the context.", flushRetryMinimumWait)
	}
	returnDelay := result.returnedAt.Sub(cancelledAt)
	attempts := fixture.recorder.count(flushMethod, flushRequestAnyCollection)
	t.Logf("CopyChunks sent %d Flush requests, returned %s after context cancellation, and returned the error %v.", attempts, returnDelay.Round(time.Millisecond), result.err)

	if result.err == nil {
		t.Fatalf("CopyChunks succeeded with copied row counts %v, but the test expected an error after it canceled the context.", result.copied)
	}
	if !errors.Is(result.err, context.Canceled) {
		t.Fatalf("CopyChunks returned the error %v, but the test expected an error that wraps context.Canceled.", result.err)
	}
	if !strings.Contains(strings.ToLower(result.err.Error()), flushRetryRateLimitText) {
		t.Fatalf("CopyChunks returned the error %v. The test expected that error to include both the context cancellation and the Milvus text rate limit exceeded from the last rejected Flush request.", result.err)
	}
	if returnDelay > flushRetryPromptReturn {
		t.Fatalf("CopyChunks returned %s after context cancellation, but the test expected a return within %s.", returnDelay.Round(time.Millisecond), flushRetryPromptReturn)
	}
	if attempts >= flushRetryExpectedAttempts {
		t.Fatalf("CopyChunks sent %d Flush requests, but the test expected fewer than %d because it canceled the context during a retry wait.", attempts, flushRetryExpectedAttempts)
	}
	fixture.assertSourceRowsKept(t)
}

type copyChunksResult struct {
	copied     []int
	err        error
	returnedAt time.Time
}

// startFlushRetryFixture starts a throwaway Milvus stack with
// flushRetryMilvusConfig and writes flushRetryRows source rows through
// StageReindex and PromoteStaging with the real ONNX embedder. It then flushes
// the collection once through the test-owned client and records the source row
// count.
func startFlushRetryFixture(t *testing.T) flushRetryFixture {
	t.Helper()
	return startCopyFixtureWithConfig(t, flushRetryMilvusConfig)
}

func startCopyFixtureWithConfig(t *testing.T, userConfig string) flushRetryFixture {
	t.Helper()
	stack := startThrowawayStackWithMilvusConfig(t, flushRetryMilvusMemory, userConfig)
	milvus := dialMilvus(t, stack)
	recorder := &milvusCallRecorder{}
	service := newService(t, integrationConfig(t, stack, nil), recorder)
	codebasePath := filepath.Join(t.TempDir(), flushRetryCodebaseName)
	collectionName := service.CollectionName(codebasePath)

	chunks := make([]model.StoredChunk, 0, flushRetryRows)
	for partIndex := range int32(flushRetryRows) {
		chunks = append(chunks, model.StoredChunk{
			Content:      fmt.Sprintf(flushRetryContentFormat, partIndex),
			RelativePath: flushRetrySourcePath,
			StartLine:    partIndex + 1,
			EndLine:      partIndex + 1,
		})
	}
	setupContext, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	if err := service.StageReindex(setupContext, codebasePath, chunks, semantic.Removal{}, nil, map[string]model.ReuseVector{}, semantic.CodeColumns()); err != nil {
		t.Fatalf("StageReindex for codebase path %s returned the error %v while the test wrote the source rows.", codebasePath, err)
	}
	if err := service.PromoteStaging(setupContext, codebasePath); err != nil {
		t.Fatalf("PromoteStaging for codebase path %s returned the error %v while the test wrote the source rows.", codebasePath, err)
	}
	flushCollection(t, setupContext, milvus, collectionName)

	fixture := flushRetryFixture{
		service:        service,
		recorder:       recorder,
		milvus:         milvus,
		codebasePath:   codebasePath,
		collectionName: collectionName,
		sourceRows:     0,
	}
	fixture.sourceRows = fixture.pathRowCount(t, flushRetrySourcePath)
	if fixture.sourceRows != flushRetryRows {
		t.Fatalf("Source path %s has %d rows after setup, but the test wrote %d rows.", flushRetrySourcePath, fixture.sourceRows, flushRetryRows)
	}
	return fixture
}

func TestCopyChunksPreservesAcknowledgedRowsWhenSourceDeleteFails(t *testing.T) {
	requireIntegration(t)
	fixture := startCopyFixtureWithConfig(t, readMilvusTestConfig(t, "delete-limit.yaml"))
	fixture.recorder.reset()
	copied, err := fixture.service.CopyChunks(context.Background(), fixture.codebasePath, flushRetryCopies())
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "rate limit") {
		t.Fatalf("Milvus must reject source deletion at its configured rate limit: %v", err)
	}
	if len(copied) != 1 || copied[0] != flushRetryRows {
		t.Fatalf("CopyChunks acknowledged row counts %v, want [%d] despite the later deletion failure", copied, flushRetryRows)
	}
	if fixture.recorder.count(deleteMethod, fixture.collectionName) == 0 {
		t.Fatal("CopyChunks did not request source deletion.")
	}
	for _, relativePath := range []string{flushRetrySourcePath, flushRetryDestinationPath} {
		if count := fixture.pathRowCount(t, relativePath); count != flushRetryRows {
			t.Fatalf("Milvus stores %d rows for %s, want %d", count, relativePath, flushRetryRows)
		}
	}
	t.Logf("Milvus rejected source deletion after acknowledging %d copied rows. Both source and destination retain %d rows.", copied[0], flushRetryRows)
}

func flushRetryCopies() []semantic.ChunkCopy {
	return []semantic.ChunkCopy{{Source: flushRetrySourcePath, Destination: flushRetryDestinationPath}}
}

// waitForFlushAttempts polls the recorder until the service sends the given
// number of Flush requests. It fails the test if CopyChunks returns first or
// flushRetryAttemptWait passes.
func (fixture flushRetryFixture) waitForFlushAttempts(t *testing.T, attempts int, results <-chan copyChunksResult) {
	t.Helper()
	deadline := time.Now().Add(flushRetryAttemptWait)
	for fixture.recorder.count(flushMethod, flushRequestAnyCollection) < attempts {
		select {
		case result := <-results:
			t.Fatalf("CopyChunks returned before it sent %d Flush requests, with copied row counts %v and the error %v.", attempts, result.copied, result.err)
		default:
		}
		if time.Now().After(deadline) {
			t.Fatalf("CopyChunks did not send %d Flush requests within %s.", attempts, flushRetryAttemptWait)
		}
		time.Sleep(flushRetryAttemptPoll)
	}
}

// assertSourceRowsKept fails the test if the service sent a Delete request for
// the collection after the recorder reset. It also fails the test if the
// current number of rows under flushRetrySourcePath differs from the number
// before CopyChunks.
func (fixture flushRetryFixture) assertSourceRowsKept(t *testing.T) {
	t.Helper()
	if deletes := fixture.recorder.count(deleteMethod, fixture.collectionName); deletes != 0 {
		t.Fatalf("CopyChunks sent %d Delete requests for collection %s, but the test requires CopyChunks to skip deletion after a flush failure.", deletes, fixture.collectionName)
	}
	if remaining := fixture.pathRowCount(t, flushRetrySourcePath); remaining != fixture.sourceRows {
		t.Fatalf("Source path %s has %d rows after CopyChunks, but it had %d rows before CopyChunks.", flushRetrySourcePath, remaining, fixture.sourceRows)
	}
}

// pathRowCount counts the rows under one relative path through the test-owned
// client with Strong consistency. The count includes rows that Milvus has not
// flushed.
func (fixture flushRetryFixture) pathRowCount(t *testing.T, relativePath string) int64 {
	t.Helper()
	queryContext, cancel := context.WithTimeout(context.Background(), milvusCallTimeout)
	defer cancel()
	filter := fmt.Sprintf("%s == %q", milvusstore.RelativePathField, relativePath)
	result, err := fixture.milvus.Query(
		queryContext,
		milvusclient.NewQueryOption(fixture.collectionName).
			WithFilter(filter).
			WithOutputFields(milvusstore.CountOutputField).
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		t.Fatalf("The count query on collection %s with filter %q returned the error %v.", fixture.collectionName, filter, err)
	}
	countColumn := result.GetColumn(milvusstore.CountOutputField)
	if countColumn == nil {
		t.Fatalf("The count query on collection %s with filter %q did not return a count(*) column.", fixture.collectionName, filter)
	}
	count, err := countColumn.GetAsInt64(0)
	if err != nil {
		t.Fatalf("Reading the count(*) value from collection %s with filter %q returned the error %v.", fixture.collectionName, filter, err)
	}
	return count
}
