//go:build restartacceptance

package restartacceptance

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/tshash"
)

const cloneConfirmationCase = "h-confirmation"

func (driver *realAcceptanceDriver) newCloneHarness(run acceptanceRun) *harness {
	return &harness{
		paths:          run.Paths,
		composeProject: run.ComposeProject,
		runner:         driver.runner,
		valueEntropy:   rand.Reader,
		archiveSizes:   run.ArchiveSizes,
		census: func(ctx context.Context) (cloneMilvusCensus, error) {
			return readCloneMilvusCensus(ctx, cloneMilvusSettings{
				Address:     cloneMilvusAddress,
				Database:    cloneMilvusDatabase,
				SkipSamples: true,
			})
		},
		readiness: func(ctx context.Context) error {
			client, err := newCloneMilvusClient(ctx)
			if err != nil {
				return err
			}
			defer closeMilvusClient(client)
			_, err = client.ListCollections(ctx, milvusclient.NewListCollectionOption())
			return err
		},
	}
}

func (driver *realAcceptanceDriver) confirmClone(
	ctx context.Context,
	run acceptanceRun,
) (runErr error) {
	defer func() {
		runErr = errors.Join(runErr, driver.takeTeardownError())
	}()
	h := driver.newCloneHarness(run)
	return h.withCompose(ctx, cloneConfirmationCase, func(caseContext context.Context) (scenarioErr error) {
		defer func() {
			if scenarioErr != nil {
				scenarioErr = errors.Join(
					scenarioErr,
					preserveCaseDiagnostics(run.Paths, cloneConfirmationCase),
				)
			}
		}()
		if err := resetIsolatedRuntime(run.Paths); err != nil {
			return err
		}
		proxies, err := startCaseProxies(caseContext)
		if err != nil {
			return err
		}
		defer func() {
			scenarioErr = errors.Join(scenarioErr, proxies.close())
		}()
		fixture, err := createAcceptanceFixture(
			filepath.Join(run.Paths.Cases, cloneConfirmationCase, "fixture"),
			"confirmation",
		)
		if err != nil {
			return err
		}
		if err := writeClydeConversation(run.Paths, fixture.marker, "initial"); err != nil {
			return err
		}
		lms, err := startDaemonRuntime(caseContext, installedLMSProcess(run), run.Paths.LMSSocket)
		if err != nil {
			return err
		}
		defer func() { driver.stopDaemonRuntime(lms) }()
		if _, err := waitForCompletedIndex(caseContext, lms.client, fixture.root); err != nil {
			return fmt.Errorf("seed clone confirmation code collection: %w", err)
		}
		clyde, err := driver.startClydeRuntime(caseContext, run)
		if err != nil {
			return err
		}
		defer driver.stopInstalledProcess(clyde.process)
		if _, err := waitForSemanticSuccess(
			caseContext,
			func(searchContext context.Context) (semanticSearchObservation, error) {
				return driver.searchClyde(searchContext, run, fixture.marker)
			},
			maximumClydeSearchRecovery,
			defaultScenarioPollInterval,
		); err != nil {
			return fmt.Errorf("seed clone confirmation conversation collection: %w", err)
		}
		milvus, err := newCloneMilvusClient(caseContext)
		if err != nil {
			return err
		}
		defer closeMilvusClient(milvus)

		collectionID := "restart-" + filepath.Base(run.Paths.RunRoot)
		conversationCollection := "conv_chunks_" + tshash.PathPrefix(collectionID)
		if err := prepareCloneConfirmationColdTargets(
			caseContext,
			lms,
			[]string{codeCollectionName(fixture.root), conversationCollection},
			stopDaemonRuntime,
			func(releaseContext context.Context, collectionName string) error {
				return releaseCloneCollection(releaseContext, milvus, collectionName)
			},
		); err != nil {
			return err
		}
		lms = nil
		lms, err = startDaemonRuntime(caseContext, installedLMSProcess(run), run.Paths.LMSSocket)
		if err != nil {
			return fmt.Errorf("restart LMS for cold clone confirmation: %w", err)
		}
		result, err := runCloneConfirmation(caseContext, cloneConfirmationInput{
			Census: h.census,
			ScalarDebt: func(debtContext context.Context) ([]string, error) {
				return readCloneConversationScalarDebt(debtContext, milvus)
			},
			ColdCodeSearch: func(searchContext context.Context) (semanticSearchObservation, error) {
				return searchCodeObservationForQuery(
					searchContext,
					lms.client,
					fixture.root,
					fixture.marker,
				)
			},
			ColdConversationSearch: func(
				searchContext context.Context,
			) (semanticSearchObservation, error) {
				return searchConversationObservation(
					searchContext,
					lms.client,
					collectionID,
					fixture.marker,
				)
			},
			Health: func(healthContext context.Context) error {
				return driver.checkCloneHealth(
					healthContext,
					run,
					h,
					lms,
					clyde,
					milvus,
				)
			},
			ActiveJobs: func(jobsContext context.Context) (int, error) {
				return countActiveCloneJobs(jobsContext, lms.client)
			},
		})
		if err != nil {
			return err
		}
		return newEvidenceRecorder(run.Paths, time.Now).Record(
			"clone-confirmation",
			"passed",
			map[string]string{
				"collections": strconv.Itoa(len(result.Census.Collections)),
				"samples":     strconv.Itoa(len(result.Census.Samples)),
			},
		)
	})
}

func searchConversationObservation(
	ctx context.Context,
	client pb.SemanticSearchDaemonServiceClient,
	collectionID string,
	query string,
) (semanticSearchObservation, error) {
	response, err := client.SearchConversations(ctx, &pb.SearchConversationsRequest{
		CollectionId: collectionID,
		Query:        query,
		Limit:        3,
	})
	if err != nil {
		return semanticSearchObservation{Code: classifySearchError(err)}, nil
	}
	resultIDs := make([]string, 0, len(response.GetResults()))
	for _, result := range response.GetResults() {
		resultIDs = append(
			resultIDs,
			result.GetConversationId()+":"+strconv.Itoa(int(result.GetMessageIndex())),
		)
	}
	return semanticSearchObservation{
		Succeeded: true,
		Source:    "semantic",
		Matches:   len(response.GetResults()),
		ResultIDs: resultIDs,
	}, nil
}

func prepareCloneConfirmationColdTargets(
	ctx context.Context,
	runtime *daemonRuntime,
	collectionNames []string,
	stop func(*daemonRuntime) error,
	release func(context.Context, string) error,
) error {
	if err := stop(runtime); err != nil {
		return fmt.Errorf("stop clone confirmation seed daemon: %w", err)
	}
	for _, collectionName := range collectionNames {
		if err := release(ctx, collectionName); err != nil {
			return fmt.Errorf("release clone confirmation collection %q: %w", collectionName, err)
		}
	}
	return nil
}

func releaseCloneCollection(
	ctx context.Context,
	client *milvusclient.Client,
	collectionName string,
) error {
	state, err := client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(collectionName))
	if err != nil {
		return fmt.Errorf("get clone collection %q load state before release: %w", collectionName, err)
	}
	if state.State != entity.LoadStateNotLoad {
		if err := client.ReleaseCollection(
			ctx,
			milvusclient.NewReleaseCollectionOption(collectionName),
		); err != nil {
			return fmt.Errorf("release clone collection %q: %w", collectionName, err)
		}
	}
	return waitForCloneLoadState(ctx, client, collectionName, entity.LoadStateNotLoad)
}

func loadCloneCollection(
	ctx context.Context,
	client *milvusclient.Client,
	collectionName string,
) error {
	state, err := client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(collectionName))
	if err != nil {
		return fmt.Errorf("get clone collection %q load state before load: %w", collectionName, err)
	}
	if state.State != entity.LoadStateLoaded {
		if _, err := client.LoadCollection(
			ctx,
			milvusclient.NewLoadCollectionOption(collectionName),
		); err != nil {
			return fmt.Errorf("load clone collection %q: %w", collectionName, err)
		}
	}
	return waitForCloneLoadState(ctx, client, collectionName, entity.LoadStateLoaded)
}

func waitForCloneLoadState(
	ctx context.Context,
	client *milvusclient.Client,
	collectionName string,
	want entity.LoadStateCode,
) error {
	for {
		state, err := client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(collectionName))
		if err == nil && state.State == want {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf(
				"wait for clone collection %q load state %v: %w",
				collectionName,
				want,
				context.Cause(ctx),
			)
		case <-time.After(defaultScenarioPollInterval):
		}
	}
}

func readCloneConversationScalarDebt(
	ctx context.Context,
	client *milvusclient.Client,
) ([]string, error) {
	collections, err := client.ListCollections(ctx, milvusclient.NewListCollectionOption())
	if err != nil {
		return nil, fmt.Errorf("list clone conversation collections: %w", err)
	}
	slices.Sort(collections)
	debt := make([]string, 0)
	for _, collectionName := range collections {
		if !strings.HasPrefix(collectionName, "conv_chunks_") {
			continue
		}
		hasDebt, checkErr := cloneConversationHasScalarDebt(ctx, client, collectionName)
		if checkErr != nil {
			return nil, checkErr
		}
		if hasDebt {
			debt = append(debt, collectionName)
		}
	}
	return debt, nil
}

func cloneConversationHasScalarDebt(
	ctx context.Context,
	client *milvusclient.Client,
	collectionName string,
) (hasDebt bool, resultErr error) {
	description, err := client.DescribeCollection(
		ctx,
		milvusclient.NewDescribeCollectionOption(collectionName),
	)
	if err != nil {
		return false, fmt.Errorf("describe clone conversation collection %q: %w", collectionName, err)
	}
	hasProvider := false
	if description.Schema != nil {
		for _, field := range description.Schema.Fields {
			if field.Name == "provider" {
				hasProvider = true
				break
			}
		}
	}
	if !hasProvider {
		return true, nil
	}
	state, err := client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(collectionName))
	if err != nil {
		return false, fmt.Errorf("get clone conversation collection %q load state: %w", collectionName, err)
	}
	restoreCold := state.State != entity.LoadStateLoaded
	if restoreCold {
		if err := loadCloneCollection(ctx, client, collectionName); err != nil {
			return false, err
		}
		defer func() {
			cleanupContext, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			resultErr = errors.Join(
				resultErr,
				releaseCloneCollection(cleanupContext, client, collectionName),
			)
		}()
	}
	iterator, err := client.QueryIterator(
		ctx,
		milvusclient.NewQueryIteratorOption(collectionName).
			WithBatchSize(1).
			WithIteratorLimit(1).
			WithFilter("provider is null").
			WithOutputFields("id").
			WithConsistencyLevel(entity.ClStrong),
	)
	if err != nil {
		return false, fmt.Errorf(
			"open clone conversation scalar debt query for %q: %w",
			collectionName,
			err,
		)
	}
	result, err := iterator.Next(ctx)
	if errors.Is(err, io.EOF) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf(
			"query clone conversation scalar debt for %q: %w",
			collectionName,
			err,
		)
	}
	return result.ResultCount > 0, nil
}

func countActiveCloneJobs(
	ctx context.Context,
	client pb.SemanticSearchDaemonServiceClient,
) (int, error) {
	response, err := client.ListJobs(ctx, &pb.ListJobsRequest{})
	if err != nil {
		return 0, err
	}
	active := 0
	for _, job := range response.GetJobs() {
		switch job.GetState() {
		case "queued", "running", "cancelling":
			active++
		case "completed", "failed", "cancelled":
		default:
			return 0, fmt.Errorf("clone job %q has unknown state %q", job.GetId(), job.GetState())
		}
	}
	return active, nil
}

func (driver *realAcceptanceDriver) checkCloneHealth(
	ctx context.Context,
	run acceptanceRun,
	h *harness,
	lms *daemonRuntime,
	clyde *clydeRuntime,
	milvus *milvusclient.Client,
) error {
	if _, err := lms.client.GetStatus(ctx, &pb.GetStatusRequest{}); err != nil {
		return fmt.Errorf("get clone LMS status: %w", err)
	}
	indexes, err := lms.client.ListIndexes(ctx, &pb.ListIndexesRequest{})
	if err != nil {
		return fmt.Errorf("list clone indexes: %w", err)
	}
	if health := indexes.GetDependencyHealth(); health.GetDegraded() || health.GetMode() != "" {
		return fmt.Errorf("clone index dependencies are degraded: %s", health.GetMode())
	}
	jobs, err := lms.client.ListJobs(ctx, &pb.ListJobsRequest{})
	if err != nil {
		return fmt.Errorf("list clone jobs: %w", err)
	}
	if health := jobs.GetDependencyHealth(); health.GetDegraded() || health.GetMode() != "" {
		return fmt.Errorf("clone job dependencies are degraded: %s", health.GetMode())
	}
	status, err := driver.clydeStatus(ctx, run, clyde.process.Process.Pid)
	if err != nil {
		return err
	}
	if !status.Responding {
		return fmt.Errorf("isolated Clyde is not responding")
	}
	if _, err := milvus.ListCollections(ctx, milvusclient.NewListCollectionOption()); err != nil {
		return fmt.Errorf("list clone collections during health check: %w", err)
	}
	if err := verifyEmbeddingReadiness(
		ctx,
		fmt.Sprintf("http://127.0.0.1:%d", embeddingProxyPort),
		cloneEmbeddingModel,
		cloneEmbeddingDimension,
	); err != nil {
		return err
	}
	composeEnvironment, err := h.composeEnvironment()
	if err != nil {
		return err
	}
	output, err := h.runner.Run(
		ctx,
		composeEnvironment,
		"docker",
		"compose",
		"-p",
		h.composeProject,
		"-f",
		h.paths.ComposeFile,
		"ps",
		"--status",
		"running",
		"--services",
	)
	if err != nil {
		return fmt.Errorf("inspect clone services: %w", err)
	}
	services := strings.Fields(string(output))
	slices.Sort(services)
	wantServices := []string{"etcd", "minio", "standalone"}
	if !slices.Equal(services, wantServices) {
		return fmt.Errorf("running clone services = %v, want %v", services, wantServices)
	}
	return nil
}
