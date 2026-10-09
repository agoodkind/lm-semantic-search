package daemon_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/networkcost"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
	"goodkind.io/lm-semantic-search/internal/sandbox"
	"google.golang.org/grpc"
)

const (
	testModelArtifactName = "model.onnx"
	testTokenizerName     = "tokenizer.json"
	testModelBytes        = 256 * 1024
	testDownloadTimeout   = 30 * time.Second
	testPollInterval      = 20 * time.Millisecond
	testAttemptBackoff    = 10 * time.Millisecond

	stateDownloading = "downloading"
	stateDeferred    = "deferred"
	stateFailed      = "failed"
	stateComplete    = "complete"

	jobStateQueued        = "queued"
	jobPhaseModelDownload = "model_download"
)

//testdouble:external operating system network cost classification source
type fixedNetworkSource struct {
	classification networkcost.Classification
}

func (source fixedNetworkSource) Classify(context.Context) networkcost.Classification {
	return source.classification
}

type modelArtifactServer struct {
	server           *httptest.Server
	modelContent     []byte
	tokenizerContent []byte
	requests         atomic.Int64
	modelMissing     atomic.Bool
	dropModel        atomic.Bool
	holding          atomic.Bool
	resumedRange     atomic.Value
	holdModel        chan struct{}
	releaseOnce      sync.Once
}

func newModelArtifactServer(t *testing.T, holdModel bool) *modelArtifactServer {
	t.Helper()
	artifacts := &modelArtifactServer{
		modelContent:     bytes.Repeat([]byte("lms-model-bytes."), testModelBytes/16),
		tokenizerContent: []byte(`{"version":"lms-test"}`),
		holdModel:        make(chan struct{}),
	}
	artifacts.holding.Store(holdModel)
	artifacts.resumedRange.Store("")
	artifacts.server = httptest.NewServer(http.HandlerFunc(artifacts.serve))
	t.Cleanup(func() {
		artifacts.release()
		artifacts.server.Close()
	})
	return artifacts
}

func (artifacts *modelArtifactServer) release() {
	artifacts.releaseOnce.Do(func() {
		close(artifacts.holdModel)
	})
}

func (artifacts *modelArtifactServer) serveModelRange(
	writer http.ResponseWriter,
	rangeHeader string,
) {
	artifacts.resumedRange.Store(rangeHeader)
	offsetText := strings.TrimSuffix(strings.TrimPrefix(rangeHeader, "bytes="), "-")
	offset, err := strconv.Atoi(offsetText)
	if err != nil || offset <= 0 || offset >= len(artifacts.modelContent) {
		http.Error(writer, "unsatisfiable range", http.StatusRequestedRangeNotSatisfiable)
		return
	}
	total := len(artifacts.modelContent)
	writer.Header().Set("Content-Range", "bytes "+strconv.Itoa(offset)+"-"+strconv.Itoa(total-1)+"/"+strconv.Itoa(total))
	writer.Header().Set("Content-Length", strconv.Itoa(total-offset))
	writer.WriteHeader(http.StatusPartialContent)
	_, _ = writer.Write(artifacts.modelContent[offset:])
}

func (artifacts *modelArtifactServer) serve(writer http.ResponseWriter, request *http.Request) {
	artifacts.requests.Add(1)
	switch request.URL.Path {
	case "/" + testTokenizerName:
		_, _ = writer.Write(artifacts.tokenizerContent)
	case "/" + testModelArtifactName:
		if artifacts.modelMissing.Load() {
			http.NotFound(writer, request)
			return
		}
		rangeHeader := request.Header.Get("Range")
		dropping := artifacts.dropModel.Load()
		if rangeHeader != "" && dropping {
			panic(http.ErrAbortHandler)
		}
		if rangeHeader != "" {
			artifacts.serveModelRange(writer, rangeHeader)
			return
		}
		half := len(artifacts.modelContent) / 2
		writer.Header().Set("Content-Length", strconv.Itoa(len(artifacts.modelContent)))
		_, _ = writer.Write(artifacts.modelContent[:half])
		if flusher, ok := writer.(http.Flusher); ok {
			flusher.Flush()
		}
		if dropping {
			panic(http.ErrAbortHandler)
		}
		if artifacts.holding.Load() {
			select {
			case <-artifacts.holdModel:
			case <-request.Context().Done():
				return
			}
		}
		_, _ = writer.Write(artifacts.modelContent[half:])
	default:
		http.NotFound(writer, request)
	}
}

func (artifacts *modelArtifactServer) registry() *offlinemodel.Registry {
	modelSum := sha256.Sum256(artifacts.modelContent)
	tokenizerSum := sha256.Sum256(artifacts.tokenizerContent)
	return offlinemodel.NewRegistry(offlinemodel.EmbeddingGemma, offlinemodel.Preset{
		Name:            offlinemodel.EmbeddingGemma,
		ModelONNXURL:    artifacts.server.URL + "/" + testModelArtifactName,
		ModelSHA256:     hex.EncodeToString(modelSum[:]),
		TokenizerURL:    artifacts.server.URL + "/" + testTokenizerName,
		TokenizerSHA256: hex.EncodeToString(tokenizerSum[:]),
		Dimension:       4,
		Pooling:         offlinemodel.PoolingMean,
		MaximumTokens:   16,
	})
}

func (artifacts *modelArtifactServer) options(
	classification networkcost.Classification,
	interval time.Duration,
) daemon.ModelDownloadOptions {
	return daemon.ModelDownloadOptions{
		NetworkSource:   fixedNetworkSource{classification: classification},
		HTTPClient:      artifacts.server.Client(),
		Presets:         artifacts.registry(),
		RecheckInterval: interval,
		RetryInterval:   interval,
		AttemptBackoff:  testAttemptBackoff,
	}
}

type modelDownloadHarness struct {
	config config.Config
	client pb.SemanticSearchDaemonServiceClient
}

func offlineDownloadTestConfig(t *testing.T) config.Config {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	root, err := os.MkdirTemp("", "lms-dl-root")
	if err != nil {
		t.Fatalf("MkdirTemp returned error: %v", err)
	}
	t.Cleanup(func() {
		deadline := time.Now().Add(testDownloadTimeout)
		for {
			removeErr := os.RemoveAll(root)
			if removeErr == nil {
				return
			}
			if time.Now().After(deadline) {
				t.Errorf("remove daemon root %s: %v", root, removeErr)
				return
			}
			time.Sleep(testPollInterval)
		}
	})
	socketDirectory, err := os.MkdirTemp("", "lms-dl")
	if err != nil {
		t.Fatalf("MkdirTemp returned error: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(socketDirectory) })
	t.Setenv("CLAUDE_CONTEXTD_SOCKET_PATH", filepath.Join(socketDirectory, "d.sock"))
	t.Setenv("CLAUDE_CONTEXTD_MODEL_CACHE_ROOT", filepath.Join(root, "models"))
	t.Setenv("CLAUDE_CONTEXT_MODEL_DOWNLOAD_NETWORK_POLICY", "")
	t.Setenv("CLAUDE_CONTEXT_MODEL_DOWNLOAD_NETWORK_OVERRIDE", "")
	t.Setenv("CLAUDE_CONTEXT_PROFILE", config.ProfileOffline)
	t.Setenv("OFFLINE_EMBEDDING_MODEL", "")
	t.Setenv("EMBEDDING_PROVIDER", "")
	for _, variable := range sandbox.Env(root) {
		if _, alreadySet := os.LookupEnv(variable.Name); alreadySet {
			continue
		}
		t.Setenv(variable.Name, variable.Value)
	}
	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("config.Default returned error: %v", err)
	}
	for _, directory := range sandbox.Directories(cfg) {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			t.Fatalf("MkdirAll(%s) returned error: %v", directory, err)
		}
	}
	return cfg
}

func startModelDownloadDaemon(
	t *testing.T,
	cfg config.Config,
	artifacts *modelArtifactServer,
	classification networkcost.Classification,
	interval time.Duration,
) *modelDownloadHarness {
	t.Helper()
	manager, err := daemon.NewManagerWithOptions(context.Background(), cfg, daemon.ManagerOptions{
		ModelDownload: artifacts.options(classification, interval),
	})
	if err != nil {
		t.Fatalf("NewManagerWithOptions returned error: %v", err)
	}
	listener, err := (&net.ListenConfig{}).Listen(context.Background(), "unix", cfg.SocketPath)
	if err != nil {
		t.Fatalf("listen on %s returned error: %v", cfg.SocketPath, err)
	}
	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(grpcutil.MaxMessageBytes),
		grpc.MaxSendMsgSize(grpcutil.MaxMessageBytes),
	)
	pb.RegisterSemanticSearchDaemonServiceServer(server, daemon.NewGRPCServer(manager, nil))
	go func() {
		_ = server.Serve(listener)
	}()
	connection, client, err := grpcutil.DialDaemon(context.Background(), cfg.SocketPath)
	if err != nil {
		server.Stop()
		t.Fatalf("DialDaemon returned error: %v", err)
	}
	var closeOnce sync.Once
	closeDaemon := func() {
		closeOnce.Do(func() {
			_ = connection.Close()
			server.GracefulStop()
			_ = listener.Close()
			closeContext, cancel := context.WithTimeout(context.Background(), testDownloadTimeout)
			defer cancel()
			if closeErr := manager.Close(closeContext); closeErr != nil {
				t.Errorf("Manager.Close returned error: %v", closeErr)
			}
		})
	}
	t.Cleanup(func() {
		artifacts.release()
		closeDaemon()
	})
	return &modelDownloadHarness{config: cfg, client: client}
}

func (harness *modelDownloadHarness) startIndex(t *testing.T, repoPath string) string {
	t.Helper()
	startContext, cancel := context.WithTimeout(grpcutil.WithCorrelation(context.Background()), 10*time.Second)
	defer cancel()
	started, err := harness.client.StartIndex(startContext, &pb.StartIndexRequest{
		Path:     repoPath,
		Splitter: &pb.SplitterConfig{Type: "ast"},
		Client:   &pb.ClientInfo{Name: "model-download-test"},
	})
	if err != nil {
		t.Fatalf("StartIndex returned error: %v", err)
	}
	return started.GetJobId()
}

func (harness *modelDownloadHarness) status(t *testing.T) (*pb.GetStatusResponse, map[string]*pb.Metric) {
	t.Helper()
	requestContext, cancel := context.WithTimeout(grpcutil.WithCorrelation(context.Background()), 5*time.Second)
	defer cancel()
	response, err := harness.client.GetStatus(requestContext, &pb.GetStatusRequest{})
	if err != nil {
		t.Fatalf("GetStatus returned error: %v", err)
	}
	metrics := make(map[string]*pb.Metric, len(response.GetMetrics()))
	for _, metric := range response.GetMetrics() {
		metrics[metric.GetName()] = metric
	}
	return response, metrics
}

func (harness *modelDownloadHarness) waitForMetrics(
	t *testing.T,
	description string,
	accept func(map[string]*pb.Metric) bool,
) map[string]*pb.Metric {
	t.Helper()
	deadline := time.Now().Add(testDownloadTimeout)
	for {
		_, metrics := harness.status(t)
		if accept(metrics) {
			return metrics
		}
		if time.Now().After(deadline) {
			t.Fatalf("status never showed %s; model_download.state = %q, last_error = %q",
				description,
				metrics["model_download.state"].GetStringValue(),
				metrics["model_download.last_error"].GetStringValue(),
			)
		}
		time.Sleep(testPollInterval)
	}
}

func (harness *modelDownloadHarness) waitForJob(
	t *testing.T,
	jobID string,
	description string,
	accept func(*pb.Job) bool,
) {
	t.Helper()
	deadline := time.Now().Add(testDownloadTimeout)
	for {
		response, err := harness.client.GetJob(
			grpcutil.WithCorrelation(context.Background()),
			&pb.GetJobRequest{JobId: jobID},
		)
		if err != nil {
			t.Fatalf("GetJob returned error: %v", err)
		}
		if accept(response.GetJob()) {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("job %s never showed %s; state = %q, phase = %q",
				jobID,
				description,
				response.GetJob().GetState(),
				response.GetJob().GetProgress().GetPhase(),
			)
		}
		time.Sleep(testPollInterval)
	}
}

func modelDownloadStateIs(state string) func(map[string]*pb.Metric) bool {
	return func(metrics map[string]*pb.Metric) bool {
		return metrics["model_download.state"].GetStringValue() == state
	}
}

func requireMetricString(t *testing.T, metrics map[string]*pb.Metric, name string, want string) {
	t.Helper()
	metric, found := metrics[name]
	if !found {
		t.Fatalf("status has no metric %q", name)
	}
	value, isString := metric.GetValue().(*pb.Metric_StringValue)
	if !isString || value.StringValue != want {
		t.Fatalf("metric %s = %v, want string %q", name, metric.GetValue(), want)
	}
}

func requireMetricAbsent(t *testing.T, metrics map[string]*pb.Metric, name string) {
	t.Helper()
	metric, found := metrics[name]
	if !found {
		t.Fatalf("status has no metric %q", name)
	}
	if metric.GetValue() != nil {
		t.Fatalf("metric %s = %v, want no value", name, metric.GetValue())
	}
}

func TestDaemonAnswersStatusWhileModelDownloadRuns(t *testing.T) {
	cfg := offlineDownloadTestConfig(t)
	artifacts := newModelArtifactServer(t, true)
	harness := startModelDownloadDaemon(t, cfg, artifacts, networkcost.ClassificationNormal, time.Hour)

	metrics := harness.waitForMetrics(t, "a download with byte progress", func(metrics map[string]*pb.Metric) bool {
		return metrics["model_download.state"].GetStringValue() == stateDownloading &&
			metrics["model_download.downloaded_bytes"].GetIntValue() > 0
	})
	requireMetricString(t, metrics, "model_download.artifact", testModelArtifactName)
	requireMetricString(t, metrics, "model_download.network_classification", string(networkcost.ClassificationNormal))
	requireMetricString(t, metrics, "model_download.decision", string(networkcost.DecisionDownload))
	requireMetricAbsent(t, metrics, "model_download.last_error")
	downloadedBytes := metrics["model_download.downloaded_bytes"].GetIntValue()
	totalBytes := metrics["model_download.total_bytes"].GetIntValue()
	if totalBytes != int64(len(artifacts.modelContent)) {
		t.Fatalf("model_download.total_bytes = %d, want %d", totalBytes, len(artifacts.modelContent))
	}
	if downloadedBytes >= totalBytes {
		t.Fatalf("model_download.downloaded_bytes = %d before the server sent the second half of %d bytes", downloadedBytes, totalBytes)
	}
	percent := metrics["model_download.percent"].GetDoubleValue()
	if percent <= 0 || percent >= 100 {
		t.Fatalf("model_download.percent = %v during the download, want between 0 and 100", percent)
	}

	response, _ := harness.status(t)
	if !strings.Contains(response.GetDisplayText(), "model_download.state") {
		t.Fatalf("status text omits model_download.state:\n%s", response.GetDisplayText())
	}
	listContext, cancel := context.WithTimeout(grpcutil.WithCorrelation(context.Background()), 5*time.Second)
	defer cancel()
	if _, err := harness.client.ListIndexes(listContext, &pb.ListIndexesRequest{}); err != nil {
		t.Fatalf("ListIndexes returned error during the download: %v", err)
	}

	artifacts.release()
	completed := harness.waitForMetrics(t, "a complete download", modelDownloadStateIs(stateComplete))
	requireMetricAbsent(t, completed, "model_download.last_error")
}

func TestModelDownloadDefersOnExpensiveNetworkUntilOverride(t *testing.T) {
	cfg := offlineDownloadTestConfig(t)
	if err := config.SetModelDownloadNetworkPolicy(cfg.ConfigPath, string(networkcost.PreferenceDefer)); err != nil {
		t.Fatalf("SetModelDownloadNetworkPolicy returned error: %v", err)
	}
	artifacts := newModelArtifactServer(t, false)
	harness := startModelDownloadDaemon(t, cfg, artifacts, networkcost.ClassificationExpensive, 50*time.Millisecond)

	deferred := harness.waitForMetrics(t, "a deferred download", modelDownloadStateIs(stateDeferred))
	requireMetricString(t, deferred, "model_download.network_classification", string(networkcost.ClassificationExpensive))
	requireMetricString(t, deferred, "model_download.network_policy", string(networkcost.PreferenceDefer))
	requireMetricString(t, deferred, "model_download.decision", string(networkcost.DecisionDefer))
	requireMetricAbsent(t, deferred, "model_download.downloaded_bytes")
	requireMetricAbsent(t, deferred, "model_download.total_bytes")
	time.Sleep(200 * time.Millisecond)
	if requests := artifacts.requests.Load(); requests != 0 {
		t.Fatalf("artifact server received %d requests while the download was deferred", requests)
	}

	if err := config.SetModelDownloadNetworkOverride(cfg.ConfigPath, true); err != nil {
		t.Fatalf("SetModelDownloadNetworkOverride returned error: %v", err)
	}
	completed := harness.waitForMetrics(t, "a complete download after the override", modelDownloadStateIs(stateComplete))
	requireMetricString(t, completed, "model_download.decision", string(networkcost.DecisionDownload))
	if !completed["model_download.network_override"].GetBoolValue() {
		t.Fatal("model_download.network_override = false after the override was set")
	}
	if requests := artifacts.requests.Load(); requests == 0 {
		t.Fatal("artifact server received no request after the override")
	}
}

func TestModelDownloadWarnsOnUnknownNetwork(t *testing.T) {
	cfg := offlineDownloadTestConfig(t)
	artifacts := newModelArtifactServer(t, false)
	harness := startModelDownloadDaemon(t, cfg, artifacts, networkcost.ClassificationUnknown, time.Hour)

	completed := harness.waitForMetrics(t, "a complete download", modelDownloadStateIs(stateComplete))
	requireMetricString(t, completed, "model_download.network_classification", string(networkcost.ClassificationUnknown))
	requireMetricString(t, completed, "model_download.network_policy", string(networkcost.PreferenceWarn))
	requireMetricString(t, completed, "model_download.decision", string(networkcost.DecisionDownloadWithWarning))
	if !completed["model_download.warning"].GetBoolValue() {
		t.Fatal("model_download.warning = false for a download that started with a warning")
	}
}

func TestFailedModelDownloadKeepsDaemonAnsweringAndRetryCompletes(t *testing.T) {
	cfg := offlineDownloadTestConfig(t)
	artifacts := newModelArtifactServer(t, false)
	artifacts.modelMissing.Store(true)
	harness := startModelDownloadDaemon(t, cfg, artifacts, networkcost.ClassificationNormal, 100*time.Millisecond)

	failed := harness.waitForMetrics(t, "a failed download", func(metrics map[string]*pb.Metric) bool {
		return metrics["model_download.state"].GetStringValue() == stateFailed &&
			metrics["model_download.last_error"].GetStringValue() != ""
	})
	if !strings.Contains(failed["model_download.last_error"].GetStringValue(), "404") {
		t.Fatalf("model_download.last_error = %q, want the HTTP 404 status", failed["model_download.last_error"].GetStringValue())
	}

	startContext, cancel := context.WithTimeout(grpcutil.WithCorrelation(context.Background()), 10*time.Second)
	defer cancel()
	started, err := harness.client.StartIndex(startContext, &pb.StartIndexRequest{
		Path:     t.TempDir(),
		Splitter: &pb.SplitterConfig{Type: "ast"},
		Client:   &pb.ClientInfo{Name: "model-download-test"},
	})
	if err != nil {
		t.Fatalf("StartIndex returned error while the download was failed: %v", err)
	}
	harness.waitForJob(t, started.GetJobId(), "the model download phase", func(job *pb.Job) bool {
		return job.GetState() == jobStateQueued && job.GetProgress().GetPhase() == jobPhaseModelDownload
	})

	artifacts.modelMissing.Store(false)
	harness.waitForMetrics(t, "a complete download after the retry", modelDownloadStateIs(stateComplete))
	harness.waitForJob(t, started.GetJobId(), "a phase after the model download", func(job *pb.Job) bool {
		return job.GetProgress().GetPhase() != jobPhaseModelDownload
	})
}

func TestFailedModelDownloadResumesPartialFileOnRetry(t *testing.T) {
	cfg := offlineDownloadTestConfig(t)
	artifacts := newModelArtifactServer(t, false)
	artifacts.dropModel.Store(true)
	harness := startModelDownloadDaemon(t, cfg, artifacts, networkcost.ClassificationNormal, time.Hour)

	failed := harness.waitForMetrics(t, "a failed download", func(metrics map[string]*pb.Metric) bool {
		return metrics["model_download.state"].GetStringValue() == stateFailed &&
			metrics["model_download.last_error"].GetStringValue() != ""
	})
	if downloadedBytes := failed["model_download.downloaded_bytes"].GetIntValue(); downloadedBytes <= 0 {
		t.Fatalf("model_download.downloaded_bytes = %d after dropped connections, want a positive count", downloadedBytes)
	}
	partialPath := filepath.Join(
		cfg.ModelCacheRoot,
		"embedding-models",
		offlinemodel.EmbeddingGemma,
		testModelArtifactName+".partial",
	)
	partialInfo, err := os.Stat(partialPath)
	if err != nil {
		t.Fatalf("stat partial file after the failed download: %v", err)
	}
	if partialInfo.Size() <= 0 || partialInfo.Size() >= int64(len(artifacts.modelContent)) {
		t.Fatalf("partial file size = %d bytes, want between 0 and %d", partialInfo.Size(), len(artifacts.modelContent))
	}

	artifacts.dropModel.Store(false)
	harness.startIndex(t, t.TempDir())
	harness.waitForMetrics(t, "a complete download after the retry", modelDownloadStateIs(stateComplete))
	wantRange := "bytes=" + strconv.FormatInt(partialInfo.Size(), 10) + "-"
	if resumedRange := artifacts.resumedRange.Load(); resumedRange != wantRange {
		t.Fatalf("retry request Range header = %q, want %q", resumedRange, wantRange)
	}
	if _, statErr := os.Stat(partialPath); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("partial file remains after the completed download: stat error = %v", statErr)
	}
}

func TestInstalledModelFilesCauseNoDownloadRequests(t *testing.T) {
	cfg := offlineDownloadTestConfig(t)
	artifacts := newModelArtifactServer(t, false)
	modelDirectory := filepath.Join(cfg.ModelCacheRoot, "embedding-models", offlinemodel.EmbeddingGemma)
	if err := os.MkdirAll(modelDirectory, 0o755); err != nil {
		t.Fatalf("MkdirAll returned error: %v", err)
	}
	if err := os.WriteFile(filepath.Join(modelDirectory, testModelArtifactName), artifacts.modelContent, 0o644); err != nil {
		t.Fatalf("WriteFile(model) returned error: %v", err)
	}
	if err := os.WriteFile(filepath.Join(modelDirectory, testTokenizerName), artifacts.tokenizerContent, 0o644); err != nil {
		t.Fatalf("WriteFile(tokenizer) returned error: %v", err)
	}
	harness := startModelDownloadDaemon(t, cfg, artifacts, networkcost.ClassificationExpensive, time.Hour)

	completed := harness.waitForMetrics(t, "a complete state", modelDownloadStateIs(stateComplete))
	requireMetricAbsent(t, completed, "model_download.decision")
	requireMetricAbsent(t, completed, "model_download.network_classification")
	requireMetricAbsent(t, completed, "model_download.downloaded_bytes")
	if requests := artifacts.requests.Load(); requests != 0 {
		t.Fatalf("artifact server received %d requests with the model files installed", requests)
	}
}
