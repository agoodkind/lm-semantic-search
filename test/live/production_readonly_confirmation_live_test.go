//go:build live && production

package live

import (
	"context"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
)

func TestProductionReadOnlySearchConfirmation(t *testing.T) {
	requireProductionOptIn(t)
	daemonSocket := requiredProductionEnvironment(t, "LMS_PRODUCTION_DAEMON_SOCKET")
	codePath := requiredProductionEnvironment(t, "LMS_PRODUCTION_CODE_PATH")
	conversationID := requiredProductionEnvironment(t, "LMS_PRODUCTION_CONVERSATION_ID")
	dialContext, dialCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer dialCancel()
	connection, client, err := grpcutil.DialDaemon(dialContext, daemonSocket)
	if err != nil {
		t.Fatalf("connect to production daemon: %v", err)
	}
	t.Cleanup(func() { _ = connection.Close() })

	requireProductionDaemonReady(t, client)
	codeContext, codeCancel := context.WithTimeout(context.Background(), 15*time.Second)
	codeResponse, err := client.SearchCode(codeContext, &pb.SearchCodeRequest{
		Path:  codePath,
		Query: "Milvus residency",
		Limit: 3,
	})
	codeCancel()
	if err != nil {
		t.Fatalf("search production code through public daemon: %v", err)
	}
	if len(codeResponse.GetResults()) == 0 {
		t.Fatalf("production code search returned no results: %s", codeResponse.GetDisplayText())
	}
	conversationContext, conversationCancel := context.WithTimeout(context.Background(), 15*time.Second)
	conversationResponse, err := client.SearchConversations(
		conversationContext,
		&pb.SearchConversationsRequest{
			CollectionId: conversationID,
			Query:        "Milvus residency",
			Limit:        3,
		},
	)
	conversationCancel()
	if err != nil {
		t.Fatalf("search production conversations through public daemon: %v", err)
	}
	if len(conversationResponse.GetResults()) == 0 {
		t.Fatalf("production conversation search returned no results: %s", conversationResponse.GetDisplayText())
	}
	requireProductionDaemonReady(t, client)
}

func requireProductionDaemonReady(t *testing.T, client pb.SemanticSearchDaemonServiceClient) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if _, err := client.GetStatus(ctx, &pb.GetStatusRequest{}); err != nil {
		t.Fatalf("production daemon status: %v", err)
	}
	indexes, err := client.ListIndexes(ctx, &pb.ListIndexesRequest{})
	if err != nil {
		t.Fatalf("list production indexes: %v", err)
	}
	if indexes.GetDependencyHealth().GetDegraded() || indexes.GetDependencyHealth().GetMode() != "" {
		t.Fatalf("production index dependencies are degraded: %s", indexes.GetDependencyHealth().GetMode())
	}
	jobs, err := client.ListJobs(ctx, &pb.ListJobsRequest{})
	if err != nil {
		t.Fatalf("list production jobs: %v", err)
	}
	if jobs.GetDependencyHealth().GetDegraded() || jobs.GetDependencyHealth().GetMode() != "" {
		t.Fatalf("production job dependencies are degraded: %s", jobs.GetDependencyHealth().GetMode())
	}
	for _, job := range jobs.GetJobs() {
		switch job.GetState() {
		case "completed", "failed", "cancelled":
		case "queued", "running", "cancelling":
			t.Fatalf("production job %q is active in state %q", job.GetId(), job.GetState())
		default:
			t.Fatalf("production job %q has unknown state %q", job.GetId(), job.GetState())
		}
	}
}
