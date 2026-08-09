package semantic

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
)

type promotionRecoveryServer struct {
	milvuspb.UnimplementedMilvusServiceServer

	mutex                 sync.Mutex
	collections           map[string]bool
	failRenameOld         string
	blockRename           string
	renameStarted         chan struct{}
	resumeRename          chan struct{}
	queryStarted          chan struct{}
	resumeQuery           chan struct{}
	deleteStarted         chan struct{}
	resumeDelete          chan struct{}
	dropStarted           chan struct{}
	resumeDrop            chan struct{}
	addStarted            chan struct{}
	resumeAdd             chan struct{}
	missingSchema         bool
	createStarted         chan struct{}
	resumeCreate          chan struct{}
	loadStates            map[string]commonpb.LoadState
	blockLoadState        string
	loadStateStarted      chan struct{}
	resumeLoadState       chan struct{}
	afterLoadState        string
	afterLoadStateStarted chan struct{}
	loadCollectionCalls   int
	renameCollectionCalls int
}

var (
	sharedPromotionRecoveryServer = &promotionRecoveryServer{
		collections: make(map[string]bool),
	}
	promotionRecoveryServerAddress string
)

func resetPromotionRecoveryServer() *promotionRecoveryServer {
	server := sharedPromotionRecoveryServer
	server.mutex.Lock()
	defer server.mutex.Unlock()
	server.collections = make(map[string]bool)
	server.failRenameOld = ""
	server.blockRename = ""
	server.renameStarted = nil
	server.resumeRename = nil
	server.queryStarted = nil
	server.resumeQuery = nil
	server.deleteStarted = nil
	server.resumeDelete = nil
	server.dropStarted = nil
	server.resumeDrop = nil
	server.addStarted = nil
	server.resumeAdd = nil
	server.missingSchema = false
	server.createStarted = nil
	server.resumeCreate = nil
	server.loadStates = nil
	server.blockLoadState = ""
	server.loadStateStarted = nil
	server.resumeLoadState = nil
	server.afterLoadState = ""
	server.afterLoadStateStarted = nil
	server.loadCollectionCalls = 0
	server.renameCollectionCalls = 0
	return server
}

func (server *promotionRecoveryServer) GetLoadState(
	ctx context.Context,
	request *milvuspb.GetLoadStateRequest,
) (*milvuspb.GetLoadStateResponse, error) {
	server.mutex.Lock()
	collectionName := request.GetCollectionName()
	state := commonpb.LoadState_LoadStateNotLoad
	if configuredState, ok := server.loadStates[collectionName]; ok {
		state = configuredState
	}
	shouldBlock := collectionName == server.blockLoadState
	loadStateStarted := server.loadStateStarted
	resumeLoadState := server.resumeLoadState
	shouldSignalAfter := collectionName == server.afterLoadState
	afterLoadStateStarted := server.afterLoadStateStarted
	server.mutex.Unlock()
	if shouldBlock {
		close(loadStateStarted)
		select {
		case <-resumeLoadState:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if shouldSignalAfter {
		close(afterLoadStateStarted)
	}
	return &milvuspb.GetLoadStateResponse{
		Status: promotionSuccessStatus(),
		State:  state,
	}, nil
}

func (server *promotionRecoveryServer) LoadCollection(
	context.Context,
	*milvuspb.LoadCollectionRequest,
) (*commonpb.Status, error) {
	server.mutex.Lock()
	server.loadCollectionCalls++
	server.mutex.Unlock()
	return promotionSuccessStatus(), nil
}

func (server *promotionRecoveryServer) Query(
	ctx context.Context,
	_ *milvuspb.QueryRequest,
) (*milvuspb.QueryResults, error) {
	server.mutex.Lock()
	queryStarted := server.queryStarted
	resumeQuery := server.resumeQuery
	server.mutex.Unlock()
	if queryStarted != nil {
		close(queryStarted)
		select {
		case <-resumeQuery:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return &milvuspb.QueryResults{
		Status: promotionSuccessStatus(),
		FieldsData: []*schemapb.FieldData{
			column.NewColumnInt64(countOutputField, []int64{42}).FieldData(),
		},
	}, nil
}

func (server *promotionRecoveryServer) Delete(
	ctx context.Context,
	_ *milvuspb.DeleteRequest,
) (*milvuspb.MutationResult, error) {
	server.mutex.Lock()
	deleteStarted := server.deleteStarted
	resumeDelete := server.resumeDelete
	server.mutex.Unlock()
	if deleteStarted != nil {
		close(deleteStarted)
		select {
		case <-resumeDelete:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return &milvuspb.MutationResult{Status: promotionSuccessStatus(), DeleteCnt: 1}, nil
}

func (server *promotionRecoveryServer) AddCollectionField(
	ctx context.Context,
	_ *milvuspb.AddCollectionFieldRequest,
) (*commonpb.Status, error) {
	server.mutex.Lock()
	addStarted := server.addStarted
	resumeAdd := server.resumeAdd
	server.addStarted = nil
	server.mutex.Unlock()
	if addStarted != nil {
		close(addStarted)
		select {
		case <-resumeAdd:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return promotionSuccessStatus(), nil
}

func (server *promotionRecoveryServer) CreateCollection(
	ctx context.Context,
	_ *milvuspb.CreateCollectionRequest,
) (*commonpb.Status, error) {
	server.mutex.Lock()
	createStarted := server.createStarted
	resumeCreate := server.resumeCreate
	server.mutex.Unlock()
	if createStarted != nil {
		close(createStarted)
		select {
		case <-resumeCreate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return promotionSuccessStatus(), nil
}

func (server *promotionRecoveryServer) DescribeIndex(
	context.Context,
	*milvuspb.DescribeIndexRequest,
) (*milvuspb.DescribeIndexResponse, error) {
	return &milvuspb.DescribeIndexResponse{
		Status: promotionSuccessStatus(),
		IndexDescriptions: []*milvuspb.IndexDescription{
			{
				IndexName: "content_hash_idx",
				FieldName: contentHashFieldName,
				State:     commonpb.IndexState_Finished,
			},
		},
	}, nil
}

func (server *promotionRecoveryServer) Connect(
	context.Context,
	*milvuspb.ConnectRequest,
) (*milvuspb.ConnectResponse, error) {
	return &milvuspb.ConnectResponse{
		Status:     promotionSuccessStatus(),
		Identifier: 1,
	}, nil
}

func (server *promotionRecoveryServer) HasCollection(
	_ context.Context,
	request *milvuspb.HasCollectionRequest,
) (*milvuspb.BoolResponse, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	return &milvuspb.BoolResponse{
		Status: promotionSuccessStatus(),
		Value:  server.collections[request.GetCollectionName()],
	}, nil
}

func (server *promotionRecoveryServer) DescribeCollection(
	_ context.Context,
	request *milvuspb.DescribeCollectionRequest,
) (*milvuspb.DescribeCollectionResponse, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	if !server.collections[request.GetCollectionName()] {
		return &milvuspb.DescribeCollectionResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_CollectionNotExists,
				Reason:    "collection not found",
			},
		}, nil
	}
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		{Name: splitPartFieldName},
		{Name: contentHashFieldName},
		{Name: embeddingModelFieldName},
	}}
	if server.missingSchema {
		schema = &schemapb.CollectionSchema{}
	}
	return &milvuspb.DescribeCollectionResponse{
		Status:         promotionSuccessStatus(),
		CollectionName: request.GetCollectionName(),
		Schema:         schema,
	}, nil
}

func (server *promotionRecoveryServer) ShowCollections(
	context.Context,
	*milvuspb.ShowCollectionsRequest,
) (*milvuspb.ShowCollectionsResponse, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	names := make([]string, 0, len(server.collections))
	for name := range server.collections {
		names = append(names, name)
	}
	sort.Strings(names)
	return &milvuspb.ShowCollectionsResponse{
		Status:          promotionSuccessStatus(),
		CollectionNames: names,
	}, nil
}

func (server *promotionRecoveryServer) RenameCollection(
	_ context.Context,
	request *milvuspb.RenameCollectionRequest,
) (*commonpb.Status, error) {
	server.mutex.Lock()
	server.renameCollectionCalls++
	if request.GetOldName() == server.failRenameOld {
		server.failRenameOld = ""
		server.mutex.Unlock()
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    "injected rename failure",
		}, nil
	}
	if !server.collections[request.GetOldName()] {
		server.mutex.Unlock()
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_CollectionNotExists,
			Reason:    "collection not found",
		}, nil
	}
	shouldBlock := request.GetOldName() == server.blockRename
	renameStarted := server.renameStarted
	resumeRename := server.resumeRename
	server.mutex.Unlock()
	if shouldBlock {
		close(renameStarted)
		<-resumeRename
	}
	server.mutex.Lock()
	defer server.mutex.Unlock()
	delete(server.collections, request.GetOldName())
	server.collections[request.GetNewName()] = true
	return promotionSuccessStatus(), nil
}

func (server *promotionRecoveryServer) DropCollection(
	ctx context.Context,
	request *milvuspb.DropCollectionRequest,
) (*commonpb.Status, error) {
	server.mutex.Lock()
	dropStarted := server.dropStarted
	resumeDrop := server.resumeDrop
	server.mutex.Unlock()
	if dropStarted != nil {
		close(dropStarted)
		select {
		case <-resumeDrop:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	server.mutex.Lock()
	delete(server.collections, request.GetCollectionName())
	server.mutex.Unlock()
	return promotionSuccessStatus(), nil
}

func (server *promotionRecoveryServer) setCollections(names ...string) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	server.collections = make(map[string]bool, len(names))
	for _, name := range names {
		server.collections[name] = true
	}
}

func (server *promotionRecoveryServer) failNextRenameFrom(collectionName string) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	server.failRenameOld = collectionName
}

func (server *promotionRecoveryServer) hasCollection(collectionName string) bool {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	return server.collections[collectionName]
}

func (server *promotionRecoveryServer) loadCallCount() int {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	return server.loadCollectionCalls
}

func (server *promotionRecoveryServer) renameCallCount() int {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	return server.renameCollectionCalls
}

func promotionSuccessStatus() *commonpb.Status {
	return &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}
}

func newPromotionTestService(
	t *testing.T,
	server *promotionRecoveryServer,
) *Service {
	t.Helper()
	service, client := newDisconnectedPromotionTestService(t, server)
	if err := service.publishClient(context.Background(), client); err != nil {
		t.Fatalf("publish fake Milvus client: %v", err)
	}
	return service
}

func newDisconnectedPromotionTestService(
	t *testing.T,
	server *promotionRecoveryServer,
) (*Service, *milvusclient.Client) {
	t.Helper()
	if server != sharedPromotionRecoveryServer {
		t.Fatal("promotion test did not use the registered fake Milvus server")
	}
	if promotionRecoveryServerAddress == "" {
		t.Fatal("promotion fake Milvus server address is empty")
	}
	service := &Service{cfg: config.Config{MilvusAddress: promotionRecoveryServerAddress}}
	service.initializeResidencyController()
	client, err := service.dialMilvus(context.Background())
	if err != nil {
		t.Fatalf("dial fake Milvus: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := service.Close(context.Background()); closeErr != nil {
			t.Errorf("Close returned error: %v", closeErr)
		}
	})
	return service, client
}

func TestPromoteStagingRestoresLiveWhenStagingRenameFails(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service := newPromotionTestService(t, server)
	codebasePath := t.TempDir()
	liveName := service.CollectionName(codebasePath)
	stagingName := stagingCollectionName(liveName)
	recoveryName := liveName + "_swap_previous"
	server.setCollections(liveName, stagingName)
	server.failNextRenameFrom(stagingName)

	err := service.PromoteStaging(context.Background(), codebasePath)
	if err == nil {
		t.Fatal("PromoteStaging returned nil after the staging rename failed")
	}
	if !server.hasCollection(liveName) {
		t.Fatal("failed promotion left the prior live collection unavailable")
	}
	if !server.hasCollection(stagingName) {
		t.Fatal("failed promotion removed the staging collection needed for retry")
	}
	if server.hasCollection(recoveryName) {
		t.Fatal("failed promotion left the prior live collection under the recovery name")
	}
}

func TestPromoteStagingRejectsIrreversibleRecoveryNameBeforeRename(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service := newPromotionTestService(t, server)
	service.cfg.CollectionNameOverride = strings.Repeat("x", maxCollectionNameLength)
	codebasePath := t.TempDir()
	liveName := service.CollectionName(codebasePath)
	if len(liveName) != maxCollectionNameLength {
		t.Fatalf("live name length = %d, want %d", len(liveName), maxCollectionNameLength)
	}
	stagingName := stagingCollectionName(liveName)
	server.setCollections(liveName, stagingName)

	if err := service.PromoteStaging(context.Background(), codebasePath); err == nil {
		t.Fatal("PromoteStaging accepted an irreversible recovery name")
	}
	if calls := server.renameCallCount(); calls != 0 {
		t.Fatalf("RenameCollection calls = %d, want 0", calls)
	}
	if !server.hasCollection(liveName) || !server.hasCollection(stagingName) {
		t.Fatal("rejected promotion changed live or staging collection presence")
	}
}

func TestRecoverInterruptedPromotionsUsesCollectionPresence(t *testing.T) {
	testCases := []struct {
		name        string
		initialLive bool
		initialStg  bool
		wantStg     bool
	}{
		{
			name:        "live renamed to recovery",
			initialLive: false,
			initialStg:  true,
			wantStg:     true,
		},
		{
			name:        "staging renamed to live",
			initialLive: true,
			initialStg:  false,
			wantStg:     false,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			server := resetPromotionRecoveryServer()
			service := newPromotionTestService(t, server)
			liveName := service.CollectionName(t.TempDir())
			stagingName := stagingCollectionName(liveName)
			recoveryName := liveName + recoveryCollectionSuffix
			initialNames := []string{recoveryName}
			if testCase.initialLive {
				initialNames = append(initialNames, liveName)
			}
			if testCase.initialStg {
				initialNames = append(initialNames, stagingName)
			}
			server.setCollections(initialNames...)

			if err := service.recoverInterruptedPromotions(context.Background()); err != nil {
				t.Fatalf("recoverInterruptedPromotions returned error: %v", err)
			}
			if !server.hasCollection(liveName) {
				t.Fatal("recovery did not leave an authoritative live collection")
			}
			if server.hasCollection(recoveryName) {
				t.Fatal("recovery left the reserved recovery collection present")
			}
			if got := server.hasCollection(stagingName); got != testCase.wantStg {
				t.Fatalf("staging present = %t, want %t", got, testCase.wantStg)
			}
		})
	}
}

func TestRecoverInterruptedPromotionsRestoresEveryAcceptedRecoveryNameLength(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service := newPromotionTestService(t, server)
	maxLiveLength := maxCollectionNameLength - len(recoveryCollectionSuffix)
	for liveLength := 1; liveLength <= maxLiveLength; liveLength++ {
		liveName := strings.Repeat("x", liveLength)
		recoveryName := mustRecoveryCollectionName(t, liveName)
		server.setCollections(recoveryName)

		if err := service.recoverInterruptedPromotions(context.Background()); err != nil {
			t.Fatalf("recover length %d returned error: %v", liveLength, err)
		}
		if !server.hasCollection(liveName) {
			t.Fatalf("recover length %d did not restore the exact live name", liveLength)
		}
		if server.hasCollection(recoveryName) {
			t.Fatalf("recover length %d retained the recovery name", liveLength)
		}
	}
}

func TestPublishClientWaitsForPromotionRecoveryBeforeAvailability(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service, client := newDisconnectedPromotionTestService(t, server)
	liveName := service.CollectionName(t.TempDir())
	recoveryName := liveName + recoveryCollectionSuffix
	server.setCollections(recoveryName)
	server.blockRename = recoveryName
	server.renameStarted = make(chan struct{})
	server.resumeRename = make(chan struct{})
	publishResult := make(chan error, 1)
	go func() {
		publishResult <- service.publishClient(context.Background(), client)
	}()

	select {
	case <-server.renameStarted:
	case <-publishResult:
		t.Fatal("client publication returned before promotion recovery started")
	}
	if service.Available() {
		t.Fatal("semantic service became available before promotion recovery completed")
	}
	close(server.resumeRename)
	if err := <-publishResult; err != nil {
		t.Fatalf("publishClient returned error: %v", err)
	}
	if !service.Available() {
		t.Fatal("semantic service stayed unavailable after promotion recovery completed")
	}
	if !server.hasCollection(liveName) || server.hasCollection(recoveryName) {
		t.Fatal("client publication did not restore the authoritative live collection")
	}
}

func TestPromoteStagingTransfersActivePinToLiveResidency(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service := newPromotionTestService(t, server)
	if err := service.residency.Close(context.Background()); err != nil {
		t.Fatalf("close initial residency controller: %v", err)
	}
	clock := newTestResidencyClock()
	unloads := make(chan string, 1)
	service.residency = newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		idleTimeout: time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(_ context.Context, collectionName string) error {
			unloads <- collectionName
			return nil
		},
	})
	codebasePath := t.TempDir()
	liveName := service.CollectionName(codebasePath)
	stagingName := stagingCollectionName(liveName)
	server.setCollections(liveName, stagingName)
	lease, err := service.residency.Acquire(context.Background(), stagingName)
	if err != nil {
		t.Fatalf("Acquire staging returned error: %v", err)
	}
	lease.Release()
	pin, err := service.residency.Pin(stagingName)
	if err != nil {
		t.Fatalf("Pin staging returned error: %v", err)
	}

	if err := service.PromoteStaging(context.Background(), codebasePath); err != nil {
		t.Fatalf("PromoteStaging returned error: %v", err)
	}
	clock.Advance(2 * time.Minute)
	select {
	case collectionName := <-unloads:
		t.Fatalf("unloaded %q while the staging build pin remained active", collectionName)
	default:
	}
	pin.Release()
	clock.Advance(time.Minute)
	if collectionName := <-unloads; collectionName != liveName {
		t.Fatalf("unloaded %q, want promoted live collection %q", collectionName, liveName)
	}
}

func TestCountHoldsLeaseThroughQuery(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service := newPromotionTestService(t, server)
	if err := service.residency.Close(context.Background()); err != nil {
		t.Fatalf("close initial residency controller: %v", err)
	}
	clock := newTestResidencyClock()
	unloads := make(chan string, 1)
	service.residency = newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: 15 * time.Second,
		idleTimeout: time.Minute,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(_ context.Context, collectionName string) error {
			unloads <- collectionName
			return nil
		},
	})
	codebasePath := t.TempDir()
	collectionName := service.CollectionName(codebasePath)
	server.setCollections(collectionName)
	lease, err := service.residency.Acquire(context.Background(), collectionName)
	if err != nil {
		t.Fatalf("Acquire returned error: %v", err)
	}
	lease.Release()
	server.queryStarted = make(chan struct{})
	server.resumeQuery = make(chan struct{})
	countContext, cancelCount := context.WithCancel(context.Background())
	t.Cleanup(cancelCount)
	countResult := make(chan error, 1)
	go func() {
		count, countErr := service.Count(countContext, codebasePath)
		if countErr == nil && count != 42 {
			countErr = fmt.Errorf("Count returned %d, want 42", count)
		}
		countResult <- countErr
	}()
	<-server.queryStarted
	waitForLeaseCount(t, service.residency, collectionName, 1)
	clock.Advance(time.Minute)
	select {
	case unloaded := <-unloads:
		t.Fatalf("unloaded %q while Count query remained active", unloaded)
	default:
	}
	close(server.resumeQuery)
	if err := <-countResult; err != nil {
		t.Fatal(err)
	}
	waitForLeaseCount(t, service.residency, collectionName, 0)
	clock.Advance(time.Minute)
	select {
	case unloaded := <-unloads:
		if unloaded != collectionName {
			t.Fatalf("unloaded %q, want %q", unloaded, collectionName)
		}
	case <-time.After(time.Second):
		t.Fatal("Count release did not rearm idle unloading")
	}
}

func TestRecoveryCollectionNameRejectsTruncationCollisions(t *testing.T) {
	maxLiveLength := maxCollectionNameLength - len(recoveryCollectionSuffix)
	maxLiveName := strings.Repeat("x", maxLiveLength)
	recoveryName, err := recoveryCollectionName(maxLiveName)
	if err != nil {
		t.Fatalf("maximum reversible live name returned error: %v", err)
	}
	if recoveryName != maxLiveName+recoveryCollectionSuffix {
		t.Fatalf("recovery name = %q, want exact reversible suffix", recoveryName)
	}

	for _, liveName := range []string{
		maxLiveName + "a",
		maxLiveName + "b",
	} {
		if _, nameErr := recoveryCollectionName(liveName); nameErr == nil {
			t.Fatalf("overlength live name ending in %q was accepted", liveName[len(liveName)-1:])
		}
	}
}

func mustRecoveryCollectionName(t *testing.T, liveName string) string {
	t.Helper()
	recoveryName, err := recoveryCollectionName(liveName)
	if err != nil {
		t.Fatalf("recoveryCollectionName(%q) returned error: %v", liveName, err)
	}
	return recoveryName
}

func TestReservedCollectionsStayWithinLimitAndOutOfListings(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service := newPromotionTestService(t, server)
	codebasePath := t.TempDir()
	liveName := service.CollectionName(codebasePath)
	server.setCollections(
		liveName,
		stagingCollectionName(liveName),
		mustRecoveryCollectionName(t, liveName),
	)
	collections, err := service.ListCollections(context.Background())
	if err != nil {
		t.Fatalf("ListCollections returned error: %v", err)
	}
	if len(collections) != 1 || collections[0] != liveName {
		t.Fatalf("ListCollections returned reserved names: %v", collections)
	}
}

func TestPromoteStagingRepairsInterruptedSwapBeforePromotion(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service := newPromotionTestService(t, server)
	codebasePath := t.TempDir()
	liveName := service.CollectionName(codebasePath)
	stagingName := stagingCollectionName(liveName)
	recoveryName := mustRecoveryCollectionName(t, liveName)
	server.setCollections(stagingName, recoveryName)

	if err := service.PromoteStaging(context.Background(), codebasePath); err != nil {
		t.Fatalf("PromoteStaging returned error: %v", err)
	}
	if !server.hasCollection(liveName) {
		t.Fatal("promotion did not leave an authoritative live collection")
	}
	if server.hasCollection(stagingName) || server.hasCollection(recoveryName) {
		t.Fatal("promotion left a reserved collection after repairing the prior swap")
	}
}

func TestPruneToCurrentHoldsLeaseThroughDelete(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service := newPromotionTestService(t, server)
	if err := service.residency.Close(context.Background()); err != nil {
		t.Fatalf("close initial residency controller: %v", err)
	}
	service.residency = newCollectionResidencyController(residencyControllerConfig{
		waitTimeout: 15 * time.Second,
		loadCeiling: time.Minute,
		load: func(context.Context, string) error {
			return nil
		},
	})
	codebasePath := t.TempDir()
	collectionName := service.CollectionName(codebasePath)
	server.setCollections(collectionName)
	server.deleteStarted = make(chan struct{})
	server.resumeDelete = make(chan struct{})
	pruneContext, cancelPrune := context.WithCancel(context.Background())
	t.Cleanup(cancelPrune)
	pruneResult := make(chan error, 1)
	go func() {
		pruneResult <- service.PruneToCurrent(pruneContext, codebasePath, []string{"keep.go"})
	}()
	<-server.deleteStarted
	waitForLeaseCount(t, service.residency, collectionName, 1)
	close(server.resumeDelete)
	if err := <-pruneResult; err != nil {
		t.Fatalf("PruneToCurrent returned error: %v", err)
	}
	waitForLeaseCount(t, service.residency, collectionName, 0)
}

func TestDropUsesExclusiveMaintenance(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service := newPromotionTestService(t, server)
	codebasePath := t.TempDir()
	collectionName := service.CollectionName(codebasePath)
	server.setCollections(collectionName)
	server.dropStarted = make(chan struct{})
	server.resumeDrop = make(chan struct{})
	dropContext, cancelDrop := context.WithCancel(context.Background())
	t.Cleanup(cancelDrop)
	dropResult := make(chan error, 1)
	go func() {
		dropResult <- service.Drop(dropContext, codebasePath)
	}()
	<-server.dropStarted
	waitForMaintenance(t, service.residency, collectionName)
	close(server.resumeDrop)
	if err := <-dropResult; err != nil {
		t.Fatalf("Drop returned error: %v", err)
	}
	service.residency.mutex.Lock()
	_, retained := service.residency.entries[collectionName]
	service.residency.mutex.Unlock()
	if retained {
		t.Fatal("Drop retained stale collection residency state")
	}
}

func TestSchemaMigrationUsesExclusiveMaintenance(t *testing.T) {
	server := resetPromotionRecoveryServer()
	server.missingSchema = true
	service := newPromotionTestService(t, server)
	collectionName := service.CollectionName(t.TempDir())
	server.setCollections(collectionName)
	server.addStarted = make(chan struct{})
	server.resumeAdd = make(chan struct{})
	migrationContext, cancelMigration := context.WithCancel(context.Background())
	t.Cleanup(cancelMigration)
	migrationResult := make(chan error, 1)
	go func() {
		migrationResult <- service.ensureSplitPartColumnOnce(
			migrationContext,
			collectionName,
		)
	}()
	<-server.addStarted
	waitForMaintenance(t, service.residency, collectionName)
	close(server.resumeAdd)
	if err := <-migrationResult; err != nil {
		t.Fatalf("ensureSplitPartColumnOnce returned error: %v", err)
	}
}

func TestCreateCollectionUsesExclusiveMaintenance(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service := newPromotionTestService(t, server)
	collectionName := stagingCollectionName(service.CollectionName(t.TempDir()))
	server.createStarted = make(chan struct{})
	server.resumeCreate = make(chan struct{})
	createContext, cancelCreate := context.WithCancel(context.Background())
	createResult := make(chan error, 1)
	go func() {
		lease, createErr := service.createCollection(createContext, collectionName, 3)
		if lease != nil {
			lease.Release()
		}
		createResult <- createErr
	}()
	<-server.createStarted
	waitForMaintenance(t, service.residency, collectionName)
	cancelCreate()
	if err := <-createResult; err == nil {
		t.Fatal("createCollection returned nil after its request was canceled")
	}
}

func TestReconciliationRejectsStaleStateWithoutOrphaningProtection(t *testing.T) {
	controller := newCollectionResidencyController(residencyControllerConfig{
		waitTimeout: time.Second,
		loadCeiling: time.Second,
		load: func(context.Context, string) error {
			return nil
		},
	})
	t.Cleanup(func() {
		if err := controller.Close(context.Background()); err != nil {
			t.Errorf("Close returned error: %v", err)
		}
	})
	collectionName := "live"
	initialLease, err := controller.Acquire(context.Background(), collectionName)
	if err != nil {
		t.Fatalf("initial Acquire returned error: %v", err)
	}
	initialLease.Release()
	controller.mutex.Lock()
	entry := controller.entries[collectionName]
	controller.mutex.Unlock()

	generation := controller.beginReconciliation()
	lease, err := controller.Acquire(context.Background(), collectionName)
	if err != nil {
		t.Fatalf("Acquire during reconciliation returned error: %v", err)
	}
	pin, err := controller.Pin(collectionName)
	if err != nil {
		t.Fatalf("Pin during reconciliation returned error: %v", err)
	}
	controller.applyReconciliation(
		context.Background(),
		generation,
		collectionName,
		collectionResidencyCold,
	)

	controller.mutex.Lock()
	gotEntry := controller.entries[collectionName]
	gotState := gotEntry.state
	gotLeases := gotEntry.leases
	gotPins := gotEntry.pins
	controller.mutex.Unlock()
	if gotEntry != entry {
		t.Fatal("reconciliation replaced the existing residency entry")
	}
	if gotState != collectionResidencyReady {
		t.Fatalf("state = %d, want newer ready state", gotState)
	}
	if gotLeases != 1 || gotPins != 1 {
		t.Fatalf("protection = (%d leases, %d pins), want (1, 1)", gotLeases, gotPins)
	}
	lease.Release()
	pin.Release()
}

func TestReconciliationExcludesRecoveryEntriesFromStateAndTimers(t *testing.T) {
	clock := newTestResidencyClock()
	controller := newCollectionResidencyController(residencyControllerConfig{
		clock:       clock,
		waitTimeout: time.Second,
		idleTimeout: time.Minute,
		loadCeiling: time.Second,
		load: func(context.Context, string) error {
			return nil
		},
		unload: func(context.Context, string) error {
			return nil
		},
	})
	t.Cleanup(func() {
		if err := controller.Close(context.Background()); err != nil {
			t.Errorf("Close returned error: %v", err)
		}
	})
	recoveryName := "live" + recoveryCollectionSuffix
	lease, err := controller.Acquire(context.Background(), recoveryName)
	if err != nil {
		t.Fatalf("Acquire returned error: %v", err)
	}
	lease.Release()

	generation := controller.beginReconciliation()
	controller.applyReconciliation(
		context.Background(),
		generation,
		recoveryName,
		collectionResidencyCold,
	)
	controller.invalidateResidency()

	controller.mutex.Lock()
	entry := controller.entries[recoveryName]
	state := entry.state
	reconciliation := entry.reconciliation
	idleTimer := entry.idleTimer
	idleDeadline := entry.idleDeadline
	controller.mutex.Unlock()
	if state != collectionResidencyReady {
		t.Fatalf("recovery state = %d, want unchanged ready state", state)
	}
	if reconciliation != 0 {
		t.Fatalf("recovery reconciliation generation = %d, want 0", reconciliation)
	}
	if idleTimer != nil || !idleDeadline.IsZero() {
		t.Fatal("reconciliation admitted a recovery idle timer")
	}
}

func TestPublishReconcilesAsynchronouslyWithoutWarmingOrStaleOverwrite(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service, client := newDisconnectedPromotionTestService(t, server)
	if err := service.residency.Close(context.Background()); err != nil {
		t.Fatalf("close initial residency controller: %v", err)
	}
	service.residency = newCollectionResidencyController(residencyControllerConfig{
		waitTimeout: time.Second,
		loadCeiling: time.Second,
		load: func(context.Context, string) error {
			return nil
		},
	})
	liveName := "a_live"
	afterName := "z_after"
	server.setCollections(liveName, afterName)
	server.loadStates = map[string]commonpb.LoadState{
		liveName:  commonpb.LoadState_LoadStateNotLoad,
		afterName: commonpb.LoadState_LoadStateLoaded,
	}
	server.blockLoadState = liveName
	server.loadStateStarted = make(chan struct{})
	server.resumeLoadState = make(chan struct{})
	server.afterLoadState = afterName
	server.afterLoadStateStarted = make(chan struct{})

	publishResult := make(chan error, 1)
	go func() {
		publishResult <- service.publishClient(context.Background(), client)
	}()
	select {
	case err := <-publishResult:
		if err != nil {
			t.Fatalf("publishClient returned error: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("publishClient waited for asynchronous reconciliation")
	}
	if !service.Available() {
		t.Fatal("service stayed unavailable while reconciliation ran")
	}
	select {
	case <-server.loadStateStarted:
	case <-time.After(time.Second):
		t.Fatal("asynchronous reconciliation did not inspect load state")
	}
	lease, err := service.residency.Acquire(context.Background(), liveName)
	if err != nil {
		t.Fatalf("Acquire during reconciliation returned error: %v", err)
	}
	pin, err := service.residency.Pin(liveName)
	if err != nil {
		t.Fatalf("Pin during reconciliation returned error: %v", err)
	}
	close(server.resumeLoadState)
	select {
	case <-server.afterLoadStateStarted:
	case <-time.After(time.Second):
		t.Fatal("reconciliation did not continue after the blocked result")
	}

	service.residency.mutex.Lock()
	entry := service.residency.entries[liveName]
	state := entry.state
	leases := entry.leases
	pins := entry.pins
	service.residency.mutex.Unlock()
	if state != collectionResidencyReady {
		t.Fatalf("state = %d, want newer ready state", state)
	}
	if leases != 1 || pins != 1 {
		t.Fatalf("protection = (%d leases, %d pins), want (1, 1)", leases, pins)
	}
	if calls := server.loadCallCount(); calls != 0 {
		t.Fatalf("LoadCollection calls = %d, want 0", calls)
	}
	lease.Release()
	pin.Release()
}

func TestConversationBackfillSweepDefersColdCollections(t *testing.T) {
	server := resetPromotionRecoveryServer()
	service := newPromotionTestService(t, server)
	if err := service.stopResidencyReconciliation(context.Background()); err != nil {
		t.Fatalf("stop reconciliation: %v", err)
	}
	if err := service.residency.Close(context.Background()); err != nil {
		t.Fatalf("close initial residency controller: %v", err)
	}
	loadStarted := make(chan struct{}, 1)
	service.residency = newCollectionResidencyController(residencyControllerConfig{
		waitTimeout: time.Second,
		loadCeiling: time.Second,
		load: func(context.Context, string) error {
			loadStarted <- struct{}{}
			return nil
		},
	})
	collectionName := conversationCollectionPrefix + "idle"
	server.setCollections(collectionName)
	service.residency.mutex.Lock()
	entry := service.residency.entryLocked(collectionName)
	entry.state = collectionResidencyCold
	service.residency.mutex.Unlock()

	service.BackfillConversationCollectionsOnce(context.Background())

	select {
	case <-loadStarted:
		t.Fatal("background backfill warmed a cold conversation collection")
	default:
	}
}

func TestConversationBackfillUsesMaintenanceForSchemaMigration(t *testing.T) {
	server := resetPromotionRecoveryServer()
	server.missingSchema = true
	service := newPromotionTestService(t, server)
	collectionName := conversationCollectionPrefix + "migration"
	server.setCollections(collectionName)
	server.addStarted = make(chan struct{})
	server.resumeAdd = make(chan struct{})
	backfillContext, cancelBackfill := context.WithCancel(context.Background())
	backfillResult := make(chan error, 1)
	go func() {
		_, backfillErr := service.BackfillConversationScalarColumns(
			backfillContext,
			collectionName,
		)
		backfillResult <- backfillErr
	}()
	select {
	case <-server.addStarted:
	case <-time.After(time.Second):
		t.Fatal("backfill did not reach AddCollectionField")
	}
	waitForMaintenance(t, service.residency, collectionName)
	cancelBackfill()
	if err := <-backfillResult; err == nil {
		t.Fatal("backfill returned nil after schema migration cancellation")
	}
}
