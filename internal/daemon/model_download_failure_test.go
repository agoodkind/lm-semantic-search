package daemon_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/networkcost"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
)

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
		t.Fatalf("StartIndex returned an error after the download failed: %v", err)
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
		t.Fatalf("partial file size = %d bytes, want a size greater than 0 and less than %d", partialInfo.Size(), len(artifacts.modelContent))
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
