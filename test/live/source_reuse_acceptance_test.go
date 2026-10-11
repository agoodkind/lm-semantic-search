//go:build live && acceptance

package live

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
)

const (
	acceptanceClientName           = "source-reuse-acceptance"
	acceptanceJobTimeout           = 5 * time.Minute
	acceptanceProgressPollInterval = 10 * time.Millisecond
	acceptanceReportVariable       = "LMS_ACCEPTANCE_REPORT"
	acceptanceLoadsMetric          = "milvus_collection_loads_total"
	acceptanceCatalogAppendCaller  = ".loadReuseCatalogRowKeys:"

	acceptanceSearchLimit int32 = 3

	acceptanceMainDirectory = "main"
	acceptanceTargetBranch  = "target"
	acceptanceSiblingBranch = "sibling"
	acceptanceSubdirectory  = "component"

	acceptanceTargetOnlyFile  = "orbit.go"
	acceptanceTargetOnlyQuery = "days until the next launch window between two planets"
	acceptanceAddedFile       = "ledger.go"
	acceptanceModifiedFile    = "retry.go"
	acceptanceComponentFile   = "thermostat.go"

	acceptanceTargetOnlyContent = `package fixture

func OrbitalTransferWindowDays(innerPeriodDays float64, outerPeriodDays float64) float64 {
	return 1 / (1/innerPeriodDays - 1/outerPeriodDays)
}
`
	acceptanceAddedContent = `package fixture

func LedgerBalance(credits []int64, debits []int64) int64 {
	var balance int64
	for _, credit := range credits {
		balance += credit
	}
	for _, debit := range debits {
		balance -= debit
	}
	return balance
}
`
	acceptanceModifiedContent = `package fixture

import "time"

func RetryDelay(attempt int, base time.Duration) time.Duration {
	return base << attempt
}

func RetryBudgetExhausted(attempt int, maximumAttempts int) bool {
	return attempt >= maximumAttempts
}
`
	acceptanceComponentContent = `package component

func ThermostatShouldHeat(currentCelsius float64, targetCelsius float64) bool {
	return currentCelsius < targetCelsius
}
`
)

type acceptanceFile struct {
	name    string
	content string
}

var acceptanceBaseFiles = []acceptanceFile{
	{name: "invoice.go", content: `package fixture

func InvoiceTotalWithSalesTax(prices []float64, taxRate float64) float64 {
	total := 0.0
	for _, price := range prices {
		total += price
	}
	return total * (1 + taxRate)
}
`},
	{name: acceptanceModifiedFile, content: `package fixture

import "time"

func RetryDelay(attempt int, base time.Duration) time.Duration {
	return base << attempt
}
`},
	{name: "checksum.go", content: `package fixture

import "crypto/sha256"

func ArchiveChecksumMatches(archive []byte, expected [sha256.Size]byte) bool {
	return sha256.Sum256(archive) == expected
}
`},
}

var acceptanceSiblingCounts = []int{1, 10, 66}

type acceptanceMeasurement struct {
	SiblingCount                  int      `json:"sibling_count"`
	EmbeddingProvider             string   `json:"embedding_provider"`
	EmbeddingModel                string   `json:"embedding_model"`
	FirstFileSeen                 bool     `json:"first_file_seen"`
	FirstFileMilliseconds         int64    `json:"first_file_ms"`
	FirstFileObservedMilliseconds int64    `json:"first_file_observed_ms"`
	TotalMilliseconds             int64    `json:"total_ms"`
	TotalObservedMilliseconds     int64    `json:"total_observed_ms"`
	ChunksEmbedded                int32    `json:"chunks_embedded"`
	ChunksReused                  int32    `json:"chunks_reused"`
	ReuseVectorsLoaded            int32    `json:"reuse_vectors_loaded"`
	SourceChunks                  int32    `json:"source_chunks"`
	CollectionsRead               int      `json:"collections_read"`
	CollectionsReadNames          []string `json:"collections_read_names"`
	CollectionsNamed              int      `json:"collections_named"`
	ReadCalls                     int      `json:"read_calls"`
	CollectionLoads               int64    `json:"milvus_collection_loads_delta"`
	SiblingBuildsMilliseconds     int64    `json:"sibling_builds_ms"`
}

type acceptanceJobTiming struct {
	job               *pb.Job
	firstFileSeen     bool
	firstFile         time.Duration
	firstFileObserved time.Duration
	totalObserved     time.Duration
}

type acceptanceReadSummary struct {
	readNames  []string
	namedNames []string
	readCalls  int
}

func acceptanceGit(t *testing.T, directory string, arguments ...string) string {
	t.Helper()

	command := exec.CommandContext(
		context.Background(),
		"git",
		append([]string{"-C", directory}, arguments...)...,
	)
	command.Env = append(
		os.Environ(),
		"GIT_AUTHOR_NAME="+acceptanceClientName,
		"GIT_AUTHOR_EMAIL="+acceptanceClientName+"@example.invalid",
		"GIT_COMMITTER_NAME="+acceptanceClientName,
		"GIT_COMMITTER_EMAIL="+acceptanceClientName+"@example.invalid",
	)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("The git command with arguments %s in directory %s failed with error %v and output %s.", strings.Join(arguments, " "), directory, err, output)
	}
	return strings.TrimSpace(string(output))
}

func writeAcceptanceFile(t *testing.T, directory string, name string, content string) {
	t.Helper()

	path := filepath.Join(directory, name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("Writing the fixture file at %s failed with error %v.", path, err)
	}
}

func newAcceptanceRepository(t *testing.T) string {
	t.Helper()

	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("Resolving symlinks of the temporary directory failed with error %v.", err)
	}
	repository := filepath.Join(root, acceptanceMainDirectory)
	if err := os.Mkdir(repository, 0o755); err != nil {
		t.Fatalf("Creating the repository directory %s failed with error %v.", repository, err)
	}
	acceptanceGit(t, repository, "init", "--quiet", "--initial-branch", acceptanceMainDirectory)
	for _, file := range acceptanceBaseFiles {
		writeAcceptanceFile(t, repository, file.name, file.content)
	}
	acceptanceCommitAll(t, repository)
	return repository
}

func acceptanceCommitAll(t *testing.T, repository string) {
	t.Helper()

	acceptanceGit(t, repository, "add", "--all")
	acceptanceGit(t, repository, "-c", "commit.gpgsign=false", "commit", "--quiet", "--message", "fixture")
}

func addAcceptanceWorktree(t *testing.T, repository string, branch string) string {
	t.Helper()

	worktree := filepath.Join(filepath.Dir(repository), branch)
	acceptanceGit(t, repository, "worktree", "add", "--quiet", "-b", branch, worktree)
	return worktree
}

func acceptanceClientInfo() *pb.ClientInfo {
	return &pb.ClientInfo{Name: acceptanceClientName}
}

func acceptanceStartIndex(
	t *testing.T,
	liveHarness *harness,
	path string,
	sourcePath string,
	startingCommit string,
) *pb.StartIndexResponse {
	t.Helper()

	response, err := liveHarness.client.StartIndex(correlatedContext(), &pb.StartIndexRequest{
		Path:           path,
		Splitter:       &pb.SplitterConfig{Type: "ast"},
		Client:         acceptanceClientInfo(),
		SourcePath:     sourcePath,
		StartingCommit: startingCommit,
	})
	if err != nil {
		t.Fatalf("StartIndex for path %s with source path %q returned error %v.", path, sourcePath, err)
	}
	return response
}

func acceptanceObserveJob(
	t *testing.T,
	liveHarness *harness,
	jobID string,
	requestedAt time.Time,
	pollInterval time.Duration,
) acceptanceJobTiming {
	t.Helper()

	var timing acceptanceJobTiming
	deadline := time.Now().Add(acceptanceJobTimeout)
	for time.Now().Before(deadline) {
		response, err := liveHarness.client.GetJob(correlatedContext(), &pb.GetJobRequest{JobId: jobID})
		if err != nil {
			t.Fatalf("GetJob for job %s returned error %v.", jobID, err)
		}
		job := response.GetJob()
		if job == nil {
			t.Fatalf("GetJob for job %s returned no job.", jobID)
		}
		if !timing.firstFileSeen && job.GetProgress().GetFilesProcessed() > 0 {
			timing.firstFileSeen = true
			timing.firstFileObserved = time.Since(requestedAt)
			timing.firstFile = job.GetUpdatedAt().AsTime().Sub(job.GetStartedAt().AsTime())
		}
		if publicJobTerminal(job) {
			timing.job = job
			timing.totalObserved = time.Since(requestedAt)
			return timing
		}
		time.Sleep(pollInterval)
	}
	t.Fatalf("Job %s did not finish within %s.", jobID, acceptanceJobTimeout)
	return timing
}

func acceptanceRequireCompleted(t *testing.T, job *pb.Job, path string) {
	t.Helper()

	if job.GetState() != "completed" {
		t.Fatalf("Job %s for path %s ended in state %q instead of completed. The display error was %s.", job.GetId(), path, job.GetState(), job.GetDisplayError())
	}
	jobConfig := job.GetConfig()
	if jobConfig.GetEmbeddingProvider() != config.EmbeddingProviderONNX.String() {
		t.Fatalf("Job %s reports embedding provider %q instead of %q.", job.GetId(), jobConfig.GetEmbeddingProvider(), config.EmbeddingProviderONNX)
	}
	if jobConfig.GetVectorBackend() != config.IndexBackendMilvus.String() {
		t.Fatalf("Job %s reports vector backend %q instead of %q.", job.GetId(), jobConfig.GetVectorBackend(), config.IndexBackendMilvus)
	}
}

func acceptanceIndex(
	t *testing.T,
	liveHarness *harness,
	path string,
	sourcePath string,
) *pb.Job {
	t.Helper()

	requestedAt := time.Now()
	response := acceptanceStartIndex(t, liveHarness, path, sourcePath, "")
	if response.GetJobId() == "" {
		t.Fatalf("StartIndex for path %s with source path %q returned no job identifier.", path, sourcePath)
	}
	job := acceptanceObserveJob(t, liveHarness, response.GetJobId(), requestedAt, jobPollInterval).job
	acceptanceRequireCompleted(t, job, path)
	return job
}

func acceptanceSync(t *testing.T, liveHarness *harness, path string) *pb.Job {
	t.Helper()

	requestedAt := time.Now()
	response, err := liveHarness.client.SyncIndex(correlatedContext(), &pb.SyncIndexRequest{
		Path:   path,
		Client: acceptanceClientInfo(),
	})
	if err != nil {
		t.Fatalf("SyncIndex for path %s returned error %v.", path, err)
	}
	if response.GetJobId() == "" {
		t.Fatalf("SyncIndex for path %s returned no job identifier. The display text was %s.", path, response.GetDisplayText())
	}
	job := acceptanceObserveJob(t, liveHarness, response.GetJobId(), requestedAt, jobPollInterval).job
	acceptanceRequireCompleted(t, job, path)
	return job
}

func acceptanceCodebase(t *testing.T, liveHarness *harness, path string) *pb.Codebase {
	t.Helper()

	response, err := liveHarness.client.GetIndex(correlatedContext(), &pb.GetIndexRequest{
		Path:   path,
		Client: acceptanceClientInfo(),
	})
	if err != nil {
		t.Fatalf("GetIndex for path %s returned error %v.", path, err)
	}
	codebase := response.GetCodebase()
	if !response.GetTracked() || codebase == nil || codebase.GetCanonicalPath() != path {
		t.Fatalf("GetIndex for path %s returned no tracked codebase with that canonical path. The display text was %s.", path, response.GetDisplayText())
	}
	return codebase
}

func acceptanceRequireRecordedSource(
	t *testing.T,
	liveHarness *harness,
	path string,
	wantSourcePath string,
	wantStartingCommit string,
) {
	t.Helper()

	codebase := acceptanceCodebase(t, liveHarness, path)
	if codebase.GetWorktreeSourcePath() != wantSourcePath {
		t.Fatalf("The codebase at path %s records source path %q instead of %q.", path, codebase.GetWorktreeSourcePath(), wantSourcePath)
	}
	if codebase.GetWorktreeStartingCommit() != wantStartingCommit {
		t.Fatalf("The codebase at path %s records starting commit %q instead of %q.", path, codebase.GetWorktreeStartingCommit(), wantStartingCommit)
	}
}

func acceptanceCollectionFamily(collectionNames ...string) []string {
	family := make([]string, 0, 3*len(collectionNames))
	for _, collectionName := range collectionNames {
		family = append(
			family,
			collectionName,
			collectionName+"_stg",
			collectionName+"_swap_previous",
		)
	}
	return family
}

func acceptanceReadMethod(method string) bool {
	switch method {
	case "Query", "Search", "HybridSearch":
		return true
	default:
		return false
	}
}

func acceptanceCallsSince(liveHarness *harness, mark int) []milvusCall {
	return liveHarness.callRecorder.snapshot()[mark:]
}

func acceptanceCallMark(liveHarness *harness) int {
	return len(liveHarness.callRecorder.snapshot())
}

func summarizeAcceptanceReads(calls []milvusCall) acceptanceReadSummary {
	var summary acceptanceReadSummary
	for _, call := range calls {
		for _, collectionName := range call.collectionNames {
			summary.namedNames = appendNonEmpty(summary.namedNames, collectionName)
		}
		if !acceptanceReadMethod(call.method) {
			continue
		}
		summary.readCalls++
		for _, collectionName := range call.collectionNames {
			summary.readNames = appendNonEmpty(summary.readNames, collectionName)
		}
	}
	slices.Sort(summary.readNames)
	slices.Sort(summary.namedNames)
	return summary
}

func acceptanceRequireReads(
	t *testing.T,
	calls []milvusCall,
	allowedReads []string,
	forbidden []string,
	catalogName string,
) {
	t.Helper()

	readCalls := 0
	for _, call := range calls {
		for _, collectionName := range call.collectionNames {
			if slices.Contains(forbidden, collectionName) {
				t.Fatalf("Milvus method %s used an unrelated sibling's collection %s for caller %s.", call.method, collectionName, call.caller)
			}
		}
		if !acceptanceReadMethod(call.method) {
			continue
		}
		readCalls++
		for _, collectionName := range call.collectionNames {
			if collectionName == catalogName && strings.Contains(call.caller, acceptanceCatalogAppendCaller) {
				continue
			}
			if !slices.Contains(allowedReads, collectionName) {
				t.Fatalf("Milvus read method %s used collection %s outside the allowed collections %v for caller %s.", call.method, collectionName, allowedReads, call.caller)
			}
		}
	}
	if readCalls == 0 {
		t.Fatalf("The recorder observed no row read during the job among %d recorded Milvus calls.", len(calls))
	}
}

func acceptanceMetric(t *testing.T, liveHarness *harness, name string) int64 {
	t.Helper()

	response, err := liveHarness.client.GetStatus(correlatedContext(), &pb.GetStatusRequest{})
	if err != nil {
		t.Fatalf("GetStatus returned error %v.", err)
	}
	for _, metric := range response.GetMetrics() {
		if metric.GetName() == name {
			return metric.GetIntValue()
		}
	}
	t.Fatalf("GetStatus returned no metric with identifier %s.", name)
	return 0
}

func acceptanceSearch(
	t *testing.T,
	liveHarness *harness,
	path string,
	query string,
) []*pb.SearchResult {
	t.Helper()

	response, err := liveHarness.client.SearchCode(correlatedContext(), &pb.SearchCodeRequest{
		Path:   path,
		Query:  query,
		Limit:  acceptanceSearchLimit,
		Client: acceptanceClientInfo(),
	})
	if err != nil {
		t.Fatalf("SearchCode for path %s with query %q returned error %v.", path, query, err)
	}
	return response.GetResults()
}

func TestAcceptanceSourceRegistrationPersistsAcrossRestartAndLaterRequest(t *testing.T) {
	liveHarness := newONNXHarness(t)
	repository := newAcceptanceRepository(t)
	target := addAcceptanceWorktree(t, repository, acceptanceTargetBranch)
	startingCommit := acceptanceGit(t, repository, "rev-parse", "HEAD")
	liveHarness.trackCodebasePath(target)

	requestedAt := time.Now()
	response := acceptanceStartIndex(t, liveHarness, target, repository, startingCommit)
	if response.GetJobId() == "" {
		t.Fatalf("StartIndex for target %s with source path %q returned no job identifier.", target, repository)
	}
	firstRun := acceptanceObserveJob(t, liveHarness, response.GetJobId(), requestedAt, jobPollInterval).job
	acceptanceRequireCompleted(t, firstRun, target)
	acceptanceRequireRecordedSource(t, liveHarness, target, repository, startingCommit)

	liveHarness.restart(nil)
	acceptanceRequireRecordedSource(t, liveHarness, target, repository, startingCommit)

	laterRequestedAt := time.Now()
	laterResponse := acceptanceStartIndex(t, liveHarness, target, "", "")
	if laterResponse.GetJobId() != "" {
		laterRun := acceptanceObserveJob(t, liveHarness, laterResponse.GetJobId(), laterRequestedAt, jobPollInterval).job
		acceptanceRequireCompleted(t, laterRun, target)
	}
	acceptanceRequireRecordedSource(t, liveHarness, target, repository, startingCommit)

	liveHarness.restart(nil)
	acceptanceRequireRecordedSource(t, liveHarness, target, repository, startingCommit)
}

func TestAcceptanceTargetReusesDeclaredSourceAndSearchReturnsTargetContent(t *testing.T) {
	liveHarness := newONNXHarness(t)
	repository := newAcceptanceRepository(t)
	target := addAcceptanceWorktree(t, repository, acceptanceTargetBranch)
	writeAcceptanceFile(t, target, acceptanceTargetOnlyFile, acceptanceTargetOnlyContent)
	sourceCollection := liveHarness.trackCodebasePath(repository)
	targetCollection := liveHarness.trackCodebasePath(target)

	sourceRun := acceptanceIndex(t, liveHarness, repository, "")
	sourceChunks := sourceRun.GetProgress().GetChunksEmbedded()
	if sourceChunks <= 0 {
		t.Fatalf("The source build embedded %d chunks instead of more than 0.", sourceChunks)
	}

	mark := acceptanceCallMark(liveHarness)
	targetRun := acceptanceIndex(t, liveHarness, target, repository)
	targetCalls := acceptanceCallsSince(liveHarness, mark)
	targetCodebase := acceptanceCodebase(t, liveHarness, target)
	if reason := targetCodebase.GetWorktreeReuseUnavailableReason(); reason != "" {
		t.Fatalf("The target at %s reports reuse unavailable reason %q instead of an empty value.", target, reason)
	}
	if reused := targetRun.GetProgress().GetChunksReused(); reused != sourceChunks {
		t.Fatalf("The target build reused %d chunks instead of the source build's %d chunks.", reused, sourceChunks)
	}
	targetChunks := targetCodebase.GetLastSuccessfulRun().GetTotalChunks()
	wantEmbedded := targetChunks - sourceChunks
	embedded := targetRun.GetProgress().GetChunksEmbedded()
	if wantEmbedded <= 0 || embedded != wantEmbedded {
		t.Fatalf("The target build embedded %d chunks instead of the required positive count of %d. The chunk totals are %d for the target and %d for the source.", embedded, wantEmbedded, targetChunks, sourceChunks)
	}
	acceptanceRequireReads(
		t,
		targetCalls,
		acceptanceCollectionFamily(targetCollection, sourceCollection),
		nil,
		liveHarness.reuseCatalogName,
	)

	targetResults := acceptanceSearch(t, liveHarness, target, acceptanceTargetOnlyQuery)
	if len(targetResults) == 0 || targetResults[0].GetRelativePath() != acceptanceTargetOnlyFile {
		t.Fatalf("The search in target %s for query %q did not return file %s as the first result. The results were %v.", target, acceptanceTargetOnlyQuery, acceptanceTargetOnlyFile, targetResults)
	}
	for _, result := range acceptanceSearch(t, liveHarness, repository, acceptanceTargetOnlyQuery) {
		if result.GetRelativePath() == acceptanceTargetOnlyFile {
			t.Fatalf("The search in source %s returned file %s that exists only in the target.", repository, acceptanceTargetOnlyFile)
		}
	}
}

func TestAcceptanceTargetSyncsReadOnlyTargetAndSourceCollections(t *testing.T) {
	liveHarness := newONNXHarness(t)
	repository := newAcceptanceRepository(t)
	sibling := addAcceptanceWorktree(t, repository, acceptanceSiblingBranch)
	target := addAcceptanceWorktree(t, repository, acceptanceTargetBranch)
	writeAcceptanceFile(t, sibling, acceptanceAddedFile, acceptanceAddedContent)
	writeAcceptanceFile(t, sibling, acceptanceModifiedFile, acceptanceModifiedContent)
	sourceCollection := liveHarness.trackCodebasePath(repository)
	siblingCollection := liveHarness.trackCodebasePath(sibling)
	targetCollection := liveHarness.trackCodebasePath(target)

	acceptanceIndex(t, liveHarness, repository, "")
	acceptanceIndex(t, liveHarness, sibling, "")
	acceptanceIndex(t, liveHarness, target, repository)
	allowedReads := acceptanceCollectionFamily(targetCollection, sourceCollection)
	forbidden := acceptanceCollectionFamily(siblingCollection)

	writeAcceptanceFile(t, target, acceptanceAddedFile, acceptanceAddedContent)
	additionMark := acceptanceCallMark(liveHarness)
	additionRun := acceptanceSync(t, liveHarness, target)
	acceptanceRequireReads(
		t,
		acceptanceCallsSince(liveHarness, additionMark),
		allowedReads,
		forbidden,
		liveHarness.reuseCatalogName,
	)
	if embedded := additionRun.GetProgress().GetChunksEmbedded(); embedded <= 0 {
		t.Fatalf("The sync after adding file %s embedded %d chunks instead of more than 0.", acceptanceAddedFile, embedded)
	}

	writeAcceptanceFile(t, target, acceptanceModifiedFile, acceptanceModifiedContent)
	modificationMark := acceptanceCallMark(liveHarness)
	modificationRun := acceptanceSync(t, liveHarness, target)
	acceptanceRequireReads(
		t,
		acceptanceCallsSince(liveHarness, modificationMark),
		allowedReads,
		forbidden,
		liveHarness.reuseCatalogName,
	)
	if embedded := modificationRun.GetProgress().GetChunksEmbedded(); embedded <= 0 {
		t.Fatalf("The sync after modifying file %s embedded %d chunks instead of more than 0.", acceptanceModifiedFile, embedded)
	}
}

func TestAcceptanceTargetWithoutDeclaredSourceEmbedsEveryChunk(t *testing.T) {
	liveHarness := newONNXHarness(t)
	repository := newAcceptanceRepository(t)
	target := addAcceptanceWorktree(t, repository, acceptanceTargetBranch)
	siblingCollection := liveHarness.trackCodebasePath(repository)
	targetCollection := liveHarness.trackCodebasePath(target)

	siblingRun := acceptanceIndex(t, liveHarness, repository, "")
	siblingChunks := siblingRun.GetProgress().GetChunksEmbedded()
	if siblingChunks <= 0 {
		t.Fatalf("The sibling build embedded %d chunks instead of more than 0.", siblingChunks)
	}

	mark := acceptanceCallMark(liveHarness)
	targetRun := acceptanceIndex(t, liveHarness, target, "")
	acceptanceRequireReads(
		t,
		acceptanceCallsSince(liveHarness, mark),
		acceptanceCollectionFamily(targetCollection),
		acceptanceCollectionFamily(siblingCollection),
		liveHarness.reuseCatalogName,
	)
	progress := targetRun.GetProgress()
	if progress.GetChunksReused() != 0 || progress.GetChunksEmbedded() != siblingChunks {
		t.Fatalf("The target build's chunk counts were %d embedded and %d reused. The required counts are the sibling build's %d embedded chunks and 0 reused chunks.", progress.GetChunksEmbedded(), progress.GetChunksReused(), siblingChunks)
	}
	if sourcePath := acceptanceCodebase(t, liveHarness, target).GetWorktreeSourcePath(); sourcePath != "" {
		t.Fatalf("The target at %s records source path %q instead of an empty value.", target, sourcePath)
	}
}

func TestAcceptanceParentBuildReusesAndAbsorbsIndexedSubdirectory(t *testing.T) {
	liveHarness := newONNXHarness(t)
	repository := newAcceptanceRepository(t)
	subdirectory := filepath.Join(repository, acceptanceSubdirectory)
	if err := os.Mkdir(subdirectory, 0o755); err != nil {
		t.Fatalf("Creating the subdirectory %s failed with error %v.", subdirectory, err)
	}
	writeAcceptanceFile(t, subdirectory, acceptanceComponentFile, acceptanceComponentContent)
	acceptanceCommitAll(t, repository)
	liveHarness.trackCodebasePath(subdirectory)
	liveHarness.trackCodebasePath(repository)

	subdirectoryRun := acceptanceIndex(t, liveHarness, subdirectory, "")
	subdirectoryChunks := subdirectoryRun.GetProgress().GetChunksEmbedded()
	if subdirectoryChunks <= 0 {
		t.Fatalf("The subdirectory build embedded %d chunks instead of more than 0.", subdirectoryChunks)
	}

	parentRun := acceptanceIndex(t, liveHarness, repository, "")
	if reused := parentRun.GetProgress().GetChunksReused(); reused != subdirectoryChunks {
		t.Fatalf("The parent build reused %d chunks instead of the subdirectory build's %d chunks.", reused, subdirectoryChunks)
	}
	if embedded := parentRun.GetProgress().GetChunksEmbedded(); embedded <= 0 {
		t.Fatalf("The parent build embedded %d chunks instead of more than 0.", embedded)
	}
	indexes, err := liveHarness.client.ListIndexes(correlatedContext(), &pb.ListIndexesRequest{})
	if err != nil {
		t.Fatalf("ListIndexes returned error %v.", err)
	}
	for _, codebase := range indexes.GetIndexes() {
		if codebase.GetCanonicalPath() == subdirectory {
			t.Fatalf("The subdirectory codebase at %s remains registered with identifier %s.", subdirectory, codebase.GetId())
		}
	}
}

func TestAcceptanceSourceReuseMeasurement(t *testing.T) {
	measurements := make([]acceptanceMeasurement, 0, len(acceptanceSiblingCounts))
	for _, siblingCount := range acceptanceSiblingCounts {
		t.Run(fmt.Sprintf("siblings_%d", siblingCount), func(t *testing.T) {
			measurement := measureAcceptanceTargetBuild(t, siblingCount)
			appendAcceptanceReport(t, measurement)
			measurements = append(measurements, measurement)
		})
	}
	if len(measurements) != len(acceptanceSiblingCounts) {
		return
	}
	baseline := measurements[0]
	for _, measurement := range measurements[1:] {
		if measurement.CollectionsRead > baseline.CollectionsRead {
			t.Errorf(
				"The target build with %d siblings read %d collections listed in %v; the build with %d siblings read %d collections listed in %v. The collection count must not grow with the sibling count.",
				measurement.SiblingCount,
				measurement.CollectionsRead,
				measurement.CollectionsReadNames,
				baseline.SiblingCount,
				baseline.CollectionsRead,
				baseline.CollectionsReadNames,
			)
		}
	}
}

func measureAcceptanceTargetBuild(t *testing.T, siblingCount int) acceptanceMeasurement {
	t.Helper()

	liveHarness := newONNXHarness(t)
	repository := newAcceptanceRepository(t)
	liveHarness.trackCodebasePath(repository)
	sourceRun := acceptanceIndex(t, liveHarness, repository, "")

	siblingBuildsStartedAt := time.Now()
	for siblingIndex := range siblingCount {
		sibling := addAcceptanceWorktree(
			t,
			repository,
			fmt.Sprintf("%s-%02d", acceptanceSiblingBranch, siblingIndex),
		)
		writeAcceptanceFile(t, sibling, acceptanceTargetOnlyFile, acceptanceTargetOnlyContent)
		liveHarness.trackCodebasePath(sibling)
		acceptanceIndex(t, liveHarness, sibling, "")
	}
	siblingBuilds := time.Since(siblingBuildsStartedAt)

	target := addAcceptanceWorktree(t, repository, acceptanceTargetBranch)
	writeAcceptanceFile(t, target, acceptanceTargetOnlyFile, acceptanceTargetOnlyContent)
	liveHarness.trackCodebasePath(target)
	loadsBefore := acceptanceMetric(t, liveHarness, acceptanceLoadsMetric)
	mark := acceptanceCallMark(liveHarness)

	requestedAt := time.Now()
	response := acceptanceStartIndex(t, liveHarness, target, repository, "")
	if response.GetJobId() == "" {
		t.Fatalf("StartIndex for target %s with source path %q returned no job identifier.", target, repository)
	}
	timing := acceptanceObserveJob(
		t,
		liveHarness,
		response.GetJobId(),
		requestedAt,
		acceptanceProgressPollInterval,
	)
	summary := summarizeAcceptanceReads(acceptanceCallsSince(liveHarness, mark))
	loadsAfter := acceptanceMetric(t, liveHarness, acceptanceLoadsMetric)
	acceptanceRequireCompleted(t, timing.job, target)

	targetRun := timing.job
	progress := targetRun.GetProgress()
	total := targetRun.GetCompletedAt().AsTime().Sub(targetRun.GetStartedAt().AsTime())
	return acceptanceMeasurement{
		SiblingCount:                  siblingCount,
		EmbeddingProvider:             targetRun.GetConfig().GetEmbeddingProvider(),
		EmbeddingModel:                targetRun.GetConfig().GetEmbeddingModel(),
		FirstFileSeen:                 timing.firstFileSeen,
		FirstFileMilliseconds:         timing.firstFile.Milliseconds(),
		FirstFileObservedMilliseconds: timing.firstFileObserved.Milliseconds(),
		TotalMilliseconds:             total.Milliseconds(),
		TotalObservedMilliseconds:     timing.totalObserved.Milliseconds(),
		ChunksEmbedded:                progress.GetChunksEmbedded(),
		ChunksReused:                  progress.GetChunksReused(),
		ReuseVectorsLoaded:            progress.GetReuseVectorsLoaded(),
		SourceChunks:                  sourceRun.GetProgress().GetChunksEmbedded(),
		CollectionsRead:               len(summary.readNames),
		CollectionsReadNames:          summary.readNames,
		CollectionsNamed:              len(summary.namedNames),
		ReadCalls:                     summary.readCalls,
		CollectionLoads:               loadsAfter - loadsBefore,
		SiblingBuildsMilliseconds:     siblingBuilds.Milliseconds(),
	}
}

func appendAcceptanceReport(t *testing.T, measurement acceptanceMeasurement) {
	t.Helper()

	encoded, err := json.Marshal(measurement)
	if err != nil {
		t.Fatalf("Encoding the measurement for %d siblings as JSON failed with error %v.", measurement.SiblingCount, err)
	}
	t.Logf("The measurement JSON line is %s.", encoded)
	reportPath := os.Getenv(acceptanceReportVariable)
	if reportPath == "" {
		return
	}
	report, err := os.OpenFile(reportPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatalf("Opening the report file %s failed with error %v.", reportPath, err)
	}
	if _, err := report.Write(append(encoded, '\n')); err != nil {
		_ = report.Close()
		t.Fatalf("Writing the report file %s failed with error %v.", reportPath, err)
	}
	if err := report.Close(); err != nil {
		t.Fatalf("Closing the report file %s failed with error %v.", reportPath, err)
	}
}
