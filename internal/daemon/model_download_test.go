package daemon_test

import (
	"context"
	"strings"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/networkcost"
)

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
