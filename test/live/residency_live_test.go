//go:build live

package live

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
)

const residencyLiveIdleTimeout = 500 * time.Millisecond

func TestPublicCodeSearchLoadsAnIdleCollection(t *testing.T) {
	harness := newResidencyHarness(t, residencyLiveIdleTimeout)
	const sentinel = "public idle collection residency sentinel"
	codebasePath, collectionName := indexResidencyCodebase(t, harness, "idle-code", sentinel)

	statusResponse, err := harness.client.GetIndex(
		correlatedContext(),
		&pb.GetIndexRequest{
			Path:   codebasePath,
			Client: &pb.ClientInfo{Name: "residency-live-harness"},
		},
	)
	if err != nil {
		t.Fatalf("get public live index: %v", err)
	}
	if got := statusResponse.GetCodebase().GetCollectionName(); got != collectionName {
		t.Fatalf("public collection name = %q, want %q", got, collectionName)
	}
	waitForLoadState(t, harness, collectionName, entity.LoadStateNotLoad)

	searchResponse, err := harness.client.SearchCode(
		correlatedContext(),
		&pb.SearchCodeRequest{
			Path:   codebasePath,
			Query:  "public idle collection residency sentinel",
			Limit:  5,
			Client: &pb.ClientInfo{Name: "residency-live-harness"},
		},
	)
	if err != nil {
		t.Fatalf("public code search from idle: %v", err)
	}
	if len(searchResponse.GetResults()) == 0 {
		t.Fatalf("public code search from idle returned no results: %s", searchResponse.GetDisplayText())
	}
	requireKnownReleaseAttribution(t, harness, collectionName)
}

func TestConcurrentColdSearchesShareOneLoadAndLastUseUnloads(t *testing.T) {
	harness := newResidencyHarness(t, residencyLiveIdleTimeout)
	const sentinel = "shared cold load residency sentinel"
	codebasePath, collectionName := indexResidencyCodebase(t, harness, "shared-load", sentinel)
	waitForLoadState(t, harness, collectionName, entity.LoadStateNotLoad)
	harness.callRecorder.reset()

	type searchResult struct {
		response *pb.SearchCodeResponse
		err      error
	}
	start := make(chan struct{})
	results := make(chan searchResult, 2)
	for range 2 {
		go func() {
			<-start
			response, err := harness.client.SearchCode(
				correlatedContext(),
				&pb.SearchCodeRequest{
					Path:   codebasePath,
					Query:  sentinel,
					Limit:  5,
					Client: &pb.ClientInfo{Name: "residency-live-harness"},
				},
			)
			results <- searchResult{response: response, err: err}
		}()
	}
	close(start)
	for range 2 {
		result := <-results
		if result.err != nil {
			t.Fatalf("concurrent public code search from idle: %v", result.err)
		}
		if len(result.response.GetResults()) == 0 {
			t.Fatalf(
				"concurrent public code search from idle returned no results: %s",
				result.response.GetDisplayText(),
			)
		}
	}
	if calls := harness.callRecorder.count("LoadCollection", collectionName); calls != 1 {
		t.Fatalf("shared cold load calls = %d, want 1", calls)
	}
	waitForLoadState(t, harness, collectionName, entity.LoadStateNotLoad)
	if calls := harness.callRecorder.count("ReleaseCollection", collectionName); calls != 1 {
		t.Fatalf("last-use release calls = %d, want 1", calls)
	}
}

func TestActiveSearchPreventsIdleUnload(t *testing.T) {
	gate := &embedGate{arrived: make(chan int), release: make(chan struct{})}
	harness := newHarnessWithOptions(t, gate, residencyLiveIdleTimeout, true)
	t.Cleanup(func() {
		select {
		case gate.release <- struct{}{}:
		default:
		}
	})
	const sentinel = "active search residency sentinel"
	codebasePath, collectionName, jobID := startResidencyCodebase(
		t,
		harness,
		"active-search",
		sentinel,
	)
	if batchSize := <-gate.arrived; batchSize != 1 {
		t.Fatalf("setup embedding batch size = %d, want 1", batchSize)
	}
	gate.release <- struct{}{}
	job := waitForPublicJob(t, harness, jobID)
	requirePublicJobCompleted(t, job)
	waitForLoadState(t, harness, collectionName, entity.LoadStateNotLoad)
	harness.callRecorder.reset()

	type searchResult struct {
		response *pb.SearchCodeResponse
		err      error
	}
	result := make(chan searchResult, 1)
	go func() {
		response, err := harness.client.SearchCode(
			correlatedContext(),
			&pb.SearchCodeRequest{
				Path:   codebasePath,
				Query:  sentinel,
				Limit:  5,
				Client: &pb.ClientInfo{Name: "residency-live-harness"},
			},
		)
		result <- searchResult{response: response, err: err}
	}()
	select {
	case batchSize := <-gate.arrived:
		if batchSize != 1 {
			t.Fatalf("search embedding batch size = %d, want 1", batchSize)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("public search did not reach the embedding endpoint")
	}
	time.Sleep(residencyLiveIdleTimeout + 250*time.Millisecond)
	requireLoadState(t, harness, collectionName, entity.LoadStateLoaded)
	if calls := harness.callRecorder.count("ReleaseCollection", collectionName); calls != 0 {
		t.Fatalf("release calls during active search = %d, want 0", calls)
	}

	gate.release <- struct{}{}
	search := <-result
	if search.err != nil {
		t.Fatalf("active public code search: %v", search.err)
	}
	if len(search.response.GetResults()) == 0 {
		t.Fatalf("active public code search returned no results: %s", search.response.GetDisplayText())
	}
	waitForLoadState(t, harness, collectionName, entity.LoadStateNotLoad)
}

func TestPinnedStagingStaysLoadedThroughPublicIndex(t *testing.T) {
	harness := newHarnessWithOptions(t, nil, residencyLiveIdleTimeout, true)
	_, collectionName, jobID := startResidencyCodebaseFiles(
		t,
		harness,
		"staging-pin",
		"staging pin residency sentinel",
		3,
	)
	job := waitForPublicJob(t, harness, jobID)
	requirePublicJobCompleted(t, job)
	stagingName := collectionName + "_stg"
	calls := harness.callRecorder.snapshot()
	lastWrite := -1
	for callIndex, call := range calls {
		if call.method == "Insert" && slices.Contains(call.collectionNames, stagingName) {
			lastWrite = callIndex
		}
	}
	if lastWrite < 0 {
		t.Fatal("public index recorded no staging writes")
	}
	for _, call := range calls[:lastWrite] {
		if call.method == "ReleaseCollection" && slices.Contains(call.collectionNames, stagingName) {
			t.Fatalf(
				"public index released pinned staging before its final write:\n%s",
				collectionCallTrace(calls, stagingName),
			)
		}
	}
	t.Logf(
		"pinned staging trace with zero releases before the final write:\n%s",
		collectionCallTrace(calls, stagingName),
	)
}

func requireKnownReleaseAttribution(
	t *testing.T,
	harness *harness,
	collectionName string,
) {
	t.Helper()
	harness.callRecorder.reset()
	releaseContext, cancelRelease := context.WithTimeout(context.Background(), 15*time.Second)
	err := harness.milvus.ReleaseCollection(
		releaseContext,
		milvusclient.NewReleaseCollectionOption(collectionName),
	)
	cancelRelease()
	if err != nil {
		t.Fatalf("known direct ReleaseCollection: %v", err)
	}

	calls := harness.callRecorder.snapshot()
	for _, call := range calls {
		if call.method != "ReleaseCollection" ||
			!slices.Contains(call.collectionNames, collectionName) ||
			!strings.Contains(call.caller, "/test/live.") {
			continue
		}
		if call.databaseName != harness.databaseName {
			t.Fatalf(
				"known ReleaseCollection database = %q, want %q",
				call.databaseName,
				harness.databaseName,
			)
		}
		if call.recordedAt.IsZero() {
			t.Fatal("known ReleaseCollection recorded no timestamp")
		}
		t.Logf("known ReleaseCollection trace:\n%s", collectionCallTrace(calls, collectionName))
		return
	}
	t.Fatalf(
		"known ReleaseCollection recorded no direct caller:\n%s",
		collectionCallTrace(calls, collectionName),
	)
}

func collectionCallTrace(calls []milvusCall, collectionName string) string {
	var trace strings.Builder
	var startedAt time.Time
	for callIndex, call := range calls {
		if !slices.Contains(call.collectionNames, collectionName) {
			continue
		}
		if startedAt.IsZero() {
			startedAt = call.recordedAt
		}
		_, _ = fmt.Fprintf(
			&trace,
			"%d at=%s elapsed=%s method=%s database=%q caller=%q collections=%v\n",
			callIndex,
			call.recordedAt.Format(time.RFC3339Nano),
			call.recordedAt.Sub(startedAt).Round(time.Millisecond),
			call.method,
			call.databaseName,
			call.caller,
			call.collectionNames,
		)
	}
	return trace.String()
}

func TestActiveMutationPreventsIdleUnloadAndUnpinnedStagingPreservesRows(t *testing.T) {
	gate := &embedGate{arrived: make(chan int), release: make(chan struct{})}
	harness := newHarnessWithOptions(t, gate, residencyLiveIdleTimeout, true)
	t.Cleanup(func() {
		select {
		case gate.release <- struct{}{}:
		default:
		}
	})
	_, collectionName, jobID := startResidencyCodebaseFiles(
		t,
		harness,
		"active-mutation",
		"active mutation residency sentinel",
		2,
	)
	if batchSize := <-gate.arrived; batchSize != 1 {
		t.Fatalf("first staging embedding batch size = %d, want 1", batchSize)
	}
	gate.release <- struct{}{}
	if batchSize := <-gate.arrived; batchSize != 1 {
		t.Fatalf("second staging embedding batch size = %d, want 1", batchSize)
	}

	stagingName := collectionName + "_stg"
	waitForLoadState(t, harness, stagingName, entity.LoadStateLoaded)
	harness.callRecorder.reset()
	rowsBefore := queryRowSnapshots(t, harness, stagingName, `id != ""`)
	if len(rowsBefore) == 0 {
		t.Fatal("pinned staging collection contained no completed rows")
	}
	time.Sleep(residencyLiveIdleTimeout + 250*time.Millisecond)
	requireLoadState(t, harness, stagingName, entity.LoadStateLoaded)
	if calls := harness.callRecorder.count("ReleaseCollection", stagingName); calls != 0 {
		t.Fatalf("release calls while staging mutation was pinned = %d, want 0", calls)
	}

	cancelResponse, err := harness.client.CancelJob(
		correlatedContext(),
		&pb.CancelJobRequest{JobId: jobID},
	)
	if err != nil {
		t.Fatalf("cancel public staging job: %v", err)
	}
	if !cancelResponse.GetCancelled() {
		t.Fatalf("cancel public staging job returned not cancelled: %s", cancelResponse.GetDisplayText())
	}
	gate.release <- struct{}{}
	job := waitForPublicJob(t, harness, jobID)
	if job.GetState() != string(model.JobStateCancelled) {
		t.Fatalf("cancelled staging job state = %q, want cancelled", job.GetState())
	}
	waitForLoadState(t, harness, stagingName, entity.LoadStateNotLoad)

	loadContext, cancelLoad := context.WithTimeout(context.Background(), 15*time.Second)
	if _, err := harness.milvus.LoadCollection(
		loadContext,
		milvusclient.NewLoadCollectionOption(stagingName),
	); err != nil {
		cancelLoad()
		t.Fatalf("load unpinned staging collection for preservation check: %v", err)
	}
	cancelLoad()
	waitForLoadState(t, harness, stagingName, entity.LoadStateLoaded)
	rowsAfter := queryRowSnapshots(t, harness, stagingName, `id != ""`)
	if !reflect.DeepEqual(rowsAfter, rowsBefore) {
		t.Fatalf("staging rows changed across idle unload\nbefore: %+v\nafter: %+v", rowsBefore, rowsAfter)
	}
	releaseContext, cancelRelease := context.WithTimeout(context.Background(), 15*time.Second)
	if err := harness.milvus.ReleaseCollection(
		releaseContext,
		milvusclient.NewReleaseCollectionOption(stagingName),
	); err != nil {
		cancelRelease()
		t.Fatalf("release staging collection after preservation check: %v", err)
	}
	cancelRelease()
}

func TestPublicConversationSearchLoadsAnIdleCollection(t *testing.T) {
	harness := newResidencyHarness(t, residencyLiveIdleTimeout)
	conversationID := "idle-conversation-" + randomID()
	sentinel := "public idle conversation residency sentinel"
	job := harness.upsert(
		map[string][]*pb.ConversationDocument{
			conversationID: {
				{
					ConversationId: conversationID,
					MessageIndex:   0,
					Role:           "user",
					TimestampUnix:  1712345000,
					Text:           sentinel,
				},
			},
		},
		pb.ConversationReconcileMode_CONVERSATION_RECONCILE_MODE_RETAIN,
		true,
		false,
	)
	requireCompleted(t, job, "public conversation idle setup")
	waitForLoadState(t, harness, harness.collectionName, entity.LoadStateNotLoad)
	harness.callRecorder.reset()

	response, err := harness.client.SearchConversations(
		correlatedContext(),
		&pb.SearchConversationsRequest{
			CollectionId: harness.collectionID,
			Query:        sentinel,
			Limit:        5,
		},
	)
	if err != nil {
		t.Fatalf("public conversation search from idle: %v", err)
	}
	if len(response.GetResults()) == 0 {
		t.Fatalf(
			"public conversation search from idle returned no results: %s",
			response.GetDisplayText(),
		)
	}
	if got := response.GetResults()[0].GetConversationId(); got != conversationID {
		t.Fatalf("public conversation result id = %q, want %q", got, conversationID)
	}
	if calls := harness.callRecorder.count("LoadCollection", harness.collectionName); calls != 1 {
		t.Fatalf("conversation idle load calls = %d, want 1", calls)
	}
}

func TestColdResidencyTransitionPreservesRowsMmapAndJobs(t *testing.T) {
	harness := newResidencyHarness(t, residencyLiveIdleTimeout)
	const sentinel = "cold residency preservation sentinel"
	codebasePath, collectionName := indexResidencyCodebase(
		t,
		harness,
		"cold-preservation",
		sentinel,
	)
	waitForLoadState(t, harness, collectionName, entity.LoadStateNotLoad)

	loadCollectionDirect(t, harness, collectionName)
	rowsBefore := queryRowSnapshots(t, harness, collectionName, `id != ""`)
	if len(rowsBefore) == 0 {
		t.Fatal("cold residency collection contained no rows")
	}
	releaseCollectionDirect(t, harness, collectionName)

	jobIDsBefore := publicJobIDs(t, harness)
	embeddingCallsBefore := harness.embeddingRecorder.snapshot()
	response, err := harness.client.SearchCode(
		correlatedContext(),
		&pb.SearchCodeRequest{
			Path:   codebasePath,
			Query:  sentinel,
			Limit:  5,
			Client: &pb.ClientInfo{Name: "residency-live-harness"},
		},
	)
	if err != nil {
		t.Fatalf("public cold preservation search: %v", err)
	}
	if len(response.GetResults()) == 0 {
		t.Fatalf("public cold preservation search returned no results: %s", response.GetDisplayText())
	}
	waitForLoadState(t, harness, collectionName, entity.LoadStateNotLoad)

	jobIDsAfter := publicJobIDs(t, harness)
	if !slices.Equal(jobIDsAfter, jobIDsBefore) {
		t.Fatalf("public jobs changed across residency transition\nbefore: %v\nafter: %v", jobIDsBefore, jobIDsAfter)
	}
	embeddingCallsAfter := harness.embeddingRecorder.snapshot()
	transitionCalls := embeddingCallsAfter[len(embeddingCallsBefore):]
	if !reflect.DeepEqual(transitionCalls, [][]string{{sentinel}}) {
		t.Fatalf(
			"residency transition embedding calls = %v, want one query-only call %q",
			transitionCalls,
			sentinel,
		)
	}

	loadCollectionDirect(t, harness, collectionName)
	rowsAfter := queryRowSnapshots(t, harness, collectionName, `id != ""`)
	if !reflect.DeepEqual(rowsAfter, rowsBefore) {
		t.Fatalf(
			"rows changed across cold residency transition\nbefore: %+v\nafter: %+v",
			rowsBefore,
			rowsAfter,
		)
	}
	requirePresentMmapTargets(t, harness)
	releaseCollectionDirect(t, harness, collectionName)

	vectorChecksums := make([]string, 0, len(rowsAfter))
	for _, row := range rowsAfter {
		vectorChecksums = append(vectorChecksums, row.vectorChecksum)
	}
	t.Logf(
		"cold residency preserved rows=%d dense_vector_sha256=%v jobs=%v embedding_calls=%v",
		len(rowsAfter),
		vectorChecksums,
		jobIDsAfter,
		transitionCalls,
	)
}

func publicJobIDs(t *testing.T, harness *harness) []string {
	t.Helper()
	response, err := harness.client.ListJobs(correlatedContext(), &pb.ListJobsRequest{})
	if err != nil {
		t.Fatalf("list public jobs: %v", err)
	}
	jobIDs := make([]string, 0, len(response.GetJobs()))
	for _, job := range response.GetJobs() {
		jobIDs = append(jobIDs, job.GetId())
	}
	slices.Sort(jobIDs)
	return jobIDs
}

func loadCollectionDirect(t *testing.T, harness *harness, collectionName string) {
	t.Helper()
	loadContext, cancelLoad := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancelLoad()
	if _, err := harness.milvus.LoadCollection(
		loadContext,
		milvusclient.NewLoadCollectionOption(collectionName),
	); err != nil {
		t.Fatalf("load collection %s directly: %v", collectionName, err)
	}
	waitForLoadState(t, harness, collectionName, entity.LoadStateLoaded)
}

func releaseCollectionDirect(t *testing.T, harness *harness, collectionName string) {
	t.Helper()
	releaseContext, cancelRelease := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancelRelease()
	if err := harness.milvus.ReleaseCollection(
		releaseContext,
		milvusclient.NewReleaseCollectionOption(collectionName),
	); err != nil {
		t.Fatalf("release collection %s directly: %v", collectionName, err)
	}
	waitForLoadState(t, harness, collectionName, entity.LoadStateNotLoad)
}

func requirePresentMmapTargets(t *testing.T, harness *harness) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	collectionNames, err := harness.milvus.ListCollections(
		ctx,
		milvusclient.NewListCollectionOption(),
	)
	if err != nil {
		t.Fatalf("list collections for mmap audit: %v", err)
	}
	if len(collectionNames) == 0 {
		t.Fatal("mmap audit found no collections")
	}

	targetCount := 0
	for _, collectionName := range collectionNames {
		description, describeErr := harness.milvus.DescribeCollection(
			ctx,
			milvusclient.NewDescribeCollectionOption(collectionName),
		)
		if describeErr != nil {
			t.Fatalf("describe collection %s for mmap audit: %v", collectionName, describeErr)
		}
		collectionTargetCount := 0
		for _, field := range description.Schema.Fields {
			supported := !field.PrimaryKey && field.Name != "id" && field.Name != "sparse_vector"
			if supported && field.Name != "vector" && field.DataType.IsVectorType() {
				supported = false
			}
			if !supported {
				continue
			}
			collectionTargetCount++
			if got := field.TypeParams["mmap.enabled"]; got != "true" {
				t.Fatalf(
					"collection %s field %s mmap.enabled = %q, want true",
					collectionName,
					field.Name,
					got,
				)
			}
		}
		for _, fieldName := range []string{"vector", "contentHash", "sparse_vector"} {
			indexNames, listErr := harness.milvus.ListIndexes(
				ctx,
				milvusclient.NewListIndexOption(collectionName).WithFieldName(fieldName),
			)
			if listErr != nil {
				t.Fatalf("list indexes on %s for %s: %v", fieldName, collectionName, listErr)
			}
			for _, indexName := range indexNames {
				indexDescription, indexErr := harness.milvus.DescribeIndex(
					ctx,
					milvusclient.NewDescribeIndexOption(collectionName, indexName),
				)
				if indexErr != nil {
					t.Fatalf("describe index %s on %s: %v", indexName, collectionName, indexErr)
				}
				params := indexDescription.Params()
				supported := fieldName == "vector" ||
					(fieldName == "contentHash" && params["index_type"] == "INVERTED") ||
					(fieldName == "sparse_vector" && params["index_type"] == "SPARSE_INVERTED_INDEX")
				if !supported {
					continue
				}
				collectionTargetCount++
				if got := params["mmap.enabled"]; got != "true" {
					t.Fatalf(
						"collection %s index %s mmap.enabled = %q, want true",
						collectionName,
						indexName,
						got,
					)
				}
			}
		}
		if collectionTargetCount == 0 {
			t.Fatalf("collection %s exposed no mmap targets", collectionName)
		}
		targetCount += collectionTargetCount
	}
	t.Logf("mmap audit collections=%d targets=%d all_enabled=true", len(collectionNames), targetCount)
}

func waitForPublicJob(t *testing.T, harness *harness, jobID string) *pb.Job {
	t.Helper()
	if jobID == "" {
		t.Fatal("public live index returned an empty job id")
	}
	deadline := time.Now().Add(jobPollTimeout)
	for time.Now().Before(deadline) {
		response, err := harness.client.GetJob(
			correlatedContext(),
			&pb.GetJobRequest{JobId: jobID},
		)
		if err != nil {
			t.Fatalf("get public live job %s: %v", jobID, err)
		}
		job := response.GetJob()
		if job != nil {
			switch job.GetState() {
			case string(model.JobStateCompleted),
				string(model.JobStateFailed),
				string(model.JobStateCancelled):
				return job
			}
		}
		time.Sleep(jobPollInterval)
	}
	t.Fatalf("public live job %s did not finish within %s", jobID, jobPollTimeout)
	return nil
}

func indexResidencyCodebase(
	t *testing.T,
	harness *harness,
	directoryPrefix string,
	sentinel string,
) (string, string) {
	t.Helper()
	codebasePath, collectionName, jobID := startResidencyCodebase(
		t,
		harness,
		directoryPrefix,
		sentinel,
	)
	job := waitForPublicJob(t, harness, jobID)
	requirePublicJobCompleted(t, job)
	return codebasePath, collectionName
}

func startResidencyCodebase(
	t *testing.T,
	harness *harness,
	directoryPrefix string,
	sentinel string,
) (string, string, string) {
	return startResidencyCodebaseFiles(t, harness, directoryPrefix, sentinel, 1)
}

func startResidencyCodebaseFiles(
	t *testing.T,
	harness *harness,
	directoryPrefix string,
	sentinel string,
	fileCount int,
) (string, string, string) {
	t.Helper()
	codebasePath := filepath.Join(harness.stateRoot, directoryPrefix+"-"+randomID())
	if err := os.MkdirAll(codebasePath, 0o755); err != nil {
		t.Fatalf("create live codebase: %v", err)
	}
	for fileIndex := range fileCount {
		fileSentinel := sentinel
		if fileCount > 1 {
			fileSentinel = fmt.Sprintf("%s %d", sentinel, fileIndex)
		}
		source := fmt.Sprintf(
			"package residency\n\nfunc ResidencySentinel%d() string {\n\treturn %q\n}\n",
			fileIndex,
			fileSentinel,
		)
		filename := fmt.Sprintf("residency_%02d.go", fileIndex)
		if err := os.WriteFile(filepath.Join(codebasePath, filename), []byte(source), 0o644); err != nil {
			t.Fatalf("write live codebase fixture: %v", err)
		}
	}
	collectionName := harness.trackCodebasePath(codebasePath)
	startResponse, err := harness.client.StartIndex(
		correlatedContext(),
		&pb.StartIndexRequest{
			Path: codebasePath,
			Splitter: &pb.SplitterConfig{
				Type: "ast",
			},
			Client: &pb.ClientInfo{Name: "residency-live-harness"},
		},
	)
	if err != nil {
		t.Fatalf("start public live index: %v", err)
	}
	return codebasePath, collectionName, startResponse.GetJobId()
}

func requirePublicJobCompleted(t *testing.T, job *pb.Job) {
	t.Helper()
	if job.GetState() != string(model.JobStateCompleted) {
		t.Fatalf(
			"public live index state = %q, want completed: %s",
			job.GetState(),
			job.GetError().GetMessage(),
		)
	}
}

func waitForLoadState(
	t *testing.T,
	harness *harness,
	collectionName string,
	want entity.LoadStateCode,
) {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		state, err := harness.milvus.GetLoadState(
			context.Background(),
			milvusclient.NewGetLoadStateOption(collectionName),
		)
		if err != nil {
			t.Fatalf("get load state for %s: %v", collectionName, err)
		}
		if state.State == want {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("collection %s did not reach load state %v", collectionName, want)
}

func requireLoadState(
	t *testing.T,
	harness *harness,
	collectionName string,
	want entity.LoadStateCode,
) {
	t.Helper()
	state, err := harness.milvus.GetLoadState(
		context.Background(),
		milvusclient.NewGetLoadStateOption(collectionName),
	)
	if err != nil {
		t.Fatalf("get load state for %s: %v", collectionName, err)
	}
	if state.State != want {
		t.Fatalf("collection %s load state = %v, want %v", collectionName, state.State, want)
	}
}
