//go:build live && production

package live

import (
	"context"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/semantic/milvusgrpc"
)

func TestProductionResidencyConfiguration(t *testing.T) {
	requireProductionOptIn(t)
	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("resolve production configuration: %v", err)
	}
	if cfg.MilvusCollectionIdleTimeoutMS != 900000 {
		t.Fatalf(
			"MilvusCollectionIdleTimeoutMS = %d, want 900000",
			cfg.MilvusCollectionIdleTimeoutMS,
		)
	}
	t.Logf("PRODUCTION_RESIDENCY_CONFIG idle_timeout_ms=%d", cfg.MilvusCollectionIdleTimeoutMS)
}

func TestProductionConversationSearch(t *testing.T) {
	requireProductionOptIn(t)
	daemonSocket := requiredProductionEnvironment(t, "LMS_PRODUCTION_DAEMON_SOCKET")
	conversationID := requiredProductionEnvironment(t, "LMS_PRODUCTION_CONVERSATION_ID")
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	connection, client, err := grpcutil.DialDaemon(ctx, daemonSocket)
	if err != nil {
		t.Fatalf("connect to production daemon: %v", err)
	}
	t.Cleanup(func() { _ = connection.Close() })

	startedAt := time.Now()
	response, err := client.SearchConversations(ctx, &pb.SearchConversationsRequest{
		CollectionId: conversationID,
		Query:        "Milvus residency",
		Limit:        3,
	})
	if err != nil {
		t.Fatalf("search production conversations: %v", err)
	}
	if len(response.GetResults()) == 0 {
		t.Fatalf("production conversation search returned no results: %s", response.GetDisplayText())
	}
	t.Logf(
		"PRODUCTION_CONVERSATION_SEARCH duration=%s results=%d",
		time.Since(startedAt),
		len(response.GetResults()),
	)
}

func TestProductionColdSearchRecovery(t *testing.T) {
	requireProductionOptIn(t)
	daemonSocket := requiredProductionEnvironment(t, "LMS_PRODUCTION_DAEMON_SOCKET")
	codePath := requiredProductionEnvironment(t, "LMS_PRODUCTION_CODE_PATH")
	codeCollection := requiredProductionEnvironment(t, "LMS_PRODUCTION_CODE_COLLECTION")
	conversationID := requiredProductionEnvironment(t, "LMS_PRODUCTION_CONVERSATION_ID")
	conversationCollection := requiredProductionEnvironment(
		t,
		"LMS_PRODUCTION_CONVERSATION_COLLECTION",
	)
	cfg, err := config.Default()
	if err != nil {
		t.Fatalf("resolve production configuration: %v", err)
	}
	dialCtx, dialCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer dialCancel()
	milvus, err := milvusclient.New(dialCtx, &milvusclient.ClientConfig{
		Address:     cfg.MilvusAddress,
		APIKey:      cfg.MilvusToken,
		DBName:      productionDatabaseName,
		DialOptions: milvusgrpc.DialOptions(dialCtx, slog.Default(), milvusgrpc.DefaultCallTimeouts()),
	})
	if err != nil {
		t.Fatalf("connect to production Milvus: %v", err)
	}
	t.Cleanup(func() { _ = milvus.Close(context.Background()) })
	releaseProductionCollection(t, milvus, codeCollection)
	releaseProductionCollection(t, milvus, conversationCollection)

	connection, client, err := grpcutil.DialDaemon(dialCtx, daemonSocket)
	if err != nil {
		t.Fatalf("connect to production daemon: %v", err)
	}
	t.Cleanup(func() { _ = connection.Close() })

	codeCtx, codeCancel := context.WithTimeout(context.Background(), 15*time.Second)
	codeStartedAt := time.Now()
	codeResponse, err := client.SearchCode(codeCtx, &pb.SearchCodeRequest{
		Path:  codePath,
		Query: "Milvus residency",
		Limit: 3,
	})
	codeDuration := time.Since(codeStartedAt)
	codeCancel()
	if err != nil {
		t.Fatalf("search cold production code collection: %v", err)
	}
	if len(codeResponse.GetResults()) == 0 {
		t.Fatalf("cold production code search returned no results: %s", codeResponse.GetDisplayText())
	}
	requireProductionLoadState(t, milvus, codeCollection, entity.LoadStateLoaded)

	conversationCtx, conversationCancel := context.WithTimeout(context.Background(), 15*time.Second)
	conversationStartedAt := time.Now()
	conversationResponse, err := client.SearchConversations(
		conversationCtx,
		&pb.SearchConversationsRequest{
			CollectionId: conversationID,
			Query:        "Milvus residency",
			Limit:        3,
		},
	)
	conversationDuration := time.Since(conversationStartedAt)
	conversationCancel()
	if err != nil {
		t.Fatalf("search cold production conversation collection: %v", err)
	}
	if len(conversationResponse.GetResults()) == 0 {
		t.Fatalf(
			"cold production conversation search returned no results: %s",
			conversationResponse.GetDisplayText(),
		)
	}
	requireProductionLoadState(
		t,
		milvus,
		conversationCollection,
		entity.LoadStateLoaded,
	)
	t.Logf(
		"PRODUCTION_COLD_SEARCH code_duration=%s code_results=%d conversation_duration=%s conversation_results=%d",
		codeDuration,
		len(codeResponse.GetResults()),
		conversationDuration,
		len(conversationResponse.GetResults()),
	)
}

func requiredProductionEnvironment(t *testing.T, name string) string {
	t.Helper()
	value := os.Getenv(name)
	if value == "" {
		t.Fatalf("%s is required", name)
	}
	return value
}

func requireProductionLoadState(
	t *testing.T,
	client *milvusclient.Client,
	collectionName string,
	want entity.LoadStateCode,
) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	state, err := client.GetLoadState(
		ctx,
		milvusclient.NewGetLoadStateOption(collectionName),
	)
	if err != nil {
		t.Fatalf("get load state for %s: %v", collectionName, err)
	}
	if state.State != want {
		t.Fatalf("load state for %s = %v, want %v", collectionName, state.State, want)
	}
}

func releaseProductionCollection(
	t *testing.T,
	client *milvusclient.Client,
	collectionName string,
) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	state, err := client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(collectionName))
	if err != nil {
		t.Fatalf("get load state for %s before release: %v", collectionName, err)
	}
	if state.State != entity.LoadStateNotLoad {
		if err := client.ReleaseCollection(
			ctx,
			milvusclient.NewReleaseCollectionOption(collectionName),
		); err != nil {
			t.Fatalf("release production collection %s: %v", collectionName, err)
		}
	}
	waitForClientLoadState(t, ctx, client, collectionName, entity.LoadStateNotLoad)
}
