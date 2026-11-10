package daemon_test

import (
	"context"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
)

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
		t.Fatalf("metric %q is not present in status response; cannot assert it has no value", name)
	}
	if metric.GetValue() != nil {
		t.Fatalf("metric %q has value %v, want no value", name, metric.GetValue())
	}
}
