//go:build restartacceptance

package restartacceptance

import (
	"context"
	"net"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

func TestReadCloneMilvusCensusIgnoresResidencyChanges(t *testing.T) {
	server := &cloneCensusMilvusServer{
		databases:   []string{"sandbox", "default"},
		collections: []string{"beta", "alpha"},
		loadStates: map[string]commonpb.LoadState{
			"alpha": commonpb.LoadState_LoadStateLoaded,
			"beta":  commonpb.LoadState_LoadStateNotLoad,
		},
		rowCounts: map[string]int64{"alpha": 2, "beta": 1},
		rows: map[string][]cloneCensusRow{
			"alpha": {
				{identity: "alpha-1", vector: []float32{1, 2}},
				{identity: "alpha-2", vector: []float32{3, 4}},
			},
			"beta": {{identity: "beta-1", vector: []float32{5, 6}}},
		},
	}
	address := startCloneCensusServer(t, server)
	settings := cloneMilvusSettings{Address: address, Database: "default"}

	first, err := readCloneMilvusCensus(context.Background(), settings)
	if err != nil {
		t.Fatalf("read clone census: %v", err)
	}
	if !slices.Equal(first.Databases, []string{"default", "sandbox"}) {
		t.Fatalf("databases = %v", first.Databases)
	}
	alpha := collectionIdentity{Database: "default", Collection: "alpha"}
	beta := collectionIdentity{Database: "default", Collection: "beta"}
	if first.Collections[alpha] == "" || first.Collections[beta] == "" {
		t.Fatalf("collection census = %#v", first.Collections)
	}
	if first.Collections[alpha] == first.Collections[beta] {
		t.Fatal("distinct collection state produced the same census hash")
	}
	server.mutex.Lock()
	server.loadStates["alpha"] = commonpb.LoadState_LoadStateNotLoad
	server.mutex.Unlock()
	second, err := readCloneMilvusCensus(context.Background(), settings)
	if err != nil {
		t.Fatalf("read changed clone census: %v", err)
	}
	if second.Collections[alpha] != first.Collections[alpha] {
		t.Fatal("load-state change altered the durable collection hash")
	}
	server.mutex.Lock()
	methods := slices.Clone(server.methods)
	databases := slices.Clone(server.requestDatabases)
	server.mutex.Unlock()
	allowed := []string{
		"Connect",
		"DescribeCollection",
		"DescribeIndex",
		"GetCollectionStatistics",
		"GetLoadState",
		"ListDatabases",
		"Query",
		"ShowCollections",
	}
	for _, method := range methods {
		if !slices.Contains(allowed, method) {
			t.Fatalf("clone census called mutating or unexpected method %q", method)
		}
	}
	if !slices.Contains(databases, "default") {
		t.Fatalf("clone census request databases = %v, want default", databases)
	}
	if slices.Contains(server.queriedCollections, "beta") {
		t.Fatalf("cold collection was queried: %v", server.queriedCollections)
	}
}

func TestMarshalStableCollectionSchemaNormalizesPropertyOrder(t *testing.T) {
	first := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{{
			Name: "content",
			TypeParams: []*commonpb.KeyValuePair{
				{Key: "max_length", Value: "65535"},
				{Key: "mmap.enabled", Value: "true"},
			},
			IndexParams: []*commonpb.KeyValuePair{
				{Key: "metric_type", Value: "COSINE"},
				{Key: "index_type", Value: "AUTOINDEX"},
			},
		}},
		Functions: []*schemapb.FunctionSchema{{
			Name: "bm25",
			Params: []*commonpb.KeyValuePair{
				{Key: "b", Value: "0.75"},
				{Key: "k1", Value: "1.2"},
			},
		}},
	}
	second := &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{{
			Name: "content",
			TypeParams: []*commonpb.KeyValuePair{
				{Key: "mmap.enabled", Value: "true"},
				{Key: "max_length", Value: "65535"},
			},
			IndexParams: []*commonpb.KeyValuePair{
				{Key: "index_type", Value: "AUTOINDEX"},
				{Key: "metric_type", Value: "COSINE"},
			},
		}},
		Functions: []*schemapb.FunctionSchema{{
			Name: "bm25",
			Params: []*commonpb.KeyValuePair{
				{Key: "k1", Value: "1.2"},
				{Key: "b", Value: "0.75"},
			},
		}},
	}

	firstBody, err := marshalStableCollectionSchema(first)
	if err != nil {
		t.Fatalf("marshal first schema: %v", err)
	}
	secondBody, err := marshalStableCollectionSchema(second)
	if err != nil {
		t.Fatalf("marshal second schema: %v", err)
	}
	if !slices.Equal(firstBody, secondBody) {
		t.Fatal("equivalent schema property order produced different bytes")
	}
}

func TestReadCloneMilvusCensusHashesLoadedRowIdentityAndDenseVector(t *testing.T) {
	server := &cloneCensusMilvusServer{
		databases:   []string{"default"},
		collections: []string{"alpha"},
		loadStates: map[string]commonpb.LoadState{
			"alpha": commonpb.LoadState_LoadStateLoaded,
		},
		rowCounts: map[string]int64{"alpha": 2},
		rows: map[string][]cloneCensusRow{
			"alpha": {
				{identity: "alpha-1", vector: []float32{1, 2}},
				{identity: "alpha-2", vector: []float32{3, 4}},
			},
		},
	}
	address := startCloneCensusServer(t, server)
	settings := cloneMilvusSettings{Address: address, Database: "default"}
	identity := collectionIdentity{Database: "default", Collection: "alpha"}

	baseline, err := readCloneMilvusCensus(context.Background(), settings)
	if err != nil {
		t.Fatalf("read baseline census: %v", err)
	}
	server.mutex.Lock()
	server.rows["alpha"][0].vector[0] = 9
	server.mutex.Unlock()
	vectorChanged, err := readCloneMilvusCensus(context.Background(), settings)
	if err != nil {
		t.Fatalf("read vector-changed census: %v", err)
	}
	if vectorChanged.Samples[identity] == baseline.Samples[identity] {
		t.Fatal("dense-vector change did not change the collection hash")
	}

	server.mutex.Lock()
	server.rows["alpha"][0].identity = "alpha-renamed"
	server.mutex.Unlock()
	identityChanged, err := readCloneMilvusCensus(context.Background(), settings)
	if err != nil {
		t.Fatalf("read identity-changed census: %v", err)
	}
	if identityChanged.Samples[identity] == vectorChanged.Samples[identity] {
		t.Fatal("row-identity change did not change the collection hash")
	}

}

func TestReadCloneMilvusCensusReportsColdRowCountWithoutChangingMetadataHash(t *testing.T) {
	server := &cloneCensusMilvusServer{
		databases:   []string{"default"},
		collections: []string{"cold"},
		loadStates: map[string]commonpb.LoadState{
			"cold": commonpb.LoadState_LoadStateNotLoad,
		},
		rowCounts: map[string]int64{"cold": 1},
	}
	address := startCloneCensusServer(t, server)
	settings := cloneMilvusSettings{Address: address, Database: "default"}
	identity := collectionIdentity{Database: "default", Collection: "cold"}

	baseline, err := readCloneMilvusCensus(context.Background(), settings)
	if err != nil {
		t.Fatalf("read baseline census: %v", err)
	}
	server.mutex.Lock()
	server.rowCounts["cold"] = 2
	server.mutex.Unlock()
	changed, err := readCloneMilvusCensus(context.Background(), settings)
	if err != nil {
		t.Fatalf("read count-changed census: %v", err)
	}
	if changed.Collections[identity] != baseline.Collections[identity] {
		t.Fatal("row-count change altered the collection metadata hash")
	}
	if baseline.RowCounts[identity] != 1 || changed.RowCounts[identity] != 2 {
		t.Fatalf("row counts = %d then %d, want 1 then 2", baseline.RowCounts[identity], changed.RowCounts[identity])
	}
	server.mutex.Lock()
	queriedCollections := slices.Clone(server.queriedCollections)
	server.mutex.Unlock()
	if len(queriedCollections) != 0 {
		t.Fatalf("cold collection was queried: %v", queriedCollections)
	}
}

func TestReadCloneMilvusCensusKeepsSamplesStableAcrossQueryOrder(t *testing.T) {
	server := &cloneCensusMilvusServer{
		databases:   []string{"default"},
		collections: []string{"alpha"},
		loadStates: map[string]commonpb.LoadState{
			"alpha": commonpb.LoadState_LoadStateLoaded,
		},
		rowCounts: map[string]int64{"alpha": 2},
		rows: map[string][]cloneCensusRow{
			"alpha": {
				{identity: "alpha-1", vector: []float32{1, 2}},
				{identity: "alpha-2", vector: []float32{3, 4}},
			},
		},
	}
	address := startCloneCensusServer(t, server)
	settings := cloneMilvusSettings{Address: address, Database: "default"}
	identity := collectionIdentity{Database: "default", Collection: "alpha"}

	baseline, err := readCloneMilvusCensus(context.Background(), settings)
	if err != nil {
		t.Fatalf("read baseline census: %v", err)
	}
	server.mutex.Lock()
	server.reverseRows = true
	server.mutex.Unlock()
	reordered, err := readCloneMilvusCensus(context.Background(), settings)
	if err != nil {
		t.Fatalf("read reordered census: %v", err)
	}
	if reordered.Samples[identity] != baseline.Samples[identity] {
		t.Fatal("query row order changed the collection hash")
	}
}

func TestReadCloneMilvusCensusSamplesLoadedCollectionsLargerThanSampleLimit(t *testing.T) {
	server := &cloneCensusMilvusServer{
		databases:   []string{"default"},
		collections: []string{"large"},
		loadStates: map[string]commonpb.LoadState{
			"large": commonpb.LoadState_LoadStateLoaded,
		},
		rowCounts: map[string]int64{"large": cloneCensusSampleLimit + 1},
		rows: map[string][]cloneCensusRow{
			"large": {{identity: "large-1", vector: []float32{1, 2}}},
		},
	}
	address := startCloneCensusServer(t, server)
	settings := cloneMilvusSettings{Address: address, Database: "default"}
	identity := collectionIdentity{Database: "default", Collection: "large"}

	baseline, err := readCloneMilvusCensus(context.Background(), settings)
	if err != nil {
		t.Fatalf("read baseline census: %v", err)
	}
	server.mutex.Lock()
	server.rows["large"][0].vector[0] = 9
	server.mutex.Unlock()
	changed, err := readCloneMilvusCensus(context.Background(), settings)
	if err != nil {
		t.Fatalf("read changed census: %v", err)
	}
	if changed.Samples[identity] == baseline.Samples[identity] {
		t.Fatal("large loaded collection vector sample did not change the collection hash")
	}
}

func TestReadCloneMilvusCensusDoesNotRequestSparseVectorSamples(t *testing.T) {
	server := &cloneCensusMilvusServer{
		databases:          []string{"default"},
		collections:        []string{"hybrid"},
		rejectSparseOutput: true,
		loadStates: map[string]commonpb.LoadState{
			"hybrid": commonpb.LoadState_LoadStateLoaded,
		},
		rowCounts: map[string]int64{"hybrid": 1},
		rows: map[string][]cloneCensusRow{
			"hybrid": {{identity: "hybrid-1", vector: []float32{1, 2}, sparse: 3}},
		},
	}
	address := startCloneCensusServer(t, server)
	settings := cloneMilvusSettings{Address: address, Database: "default"}
	_, err := readCloneMilvusCensus(context.Background(), settings)
	if err != nil {
		t.Fatalf("read census without raw sparse vectors: %v", err)
	}
}

func TestReadCloneMilvusCensusReportsStrongLoadedRowCountWithoutChangingSample(t *testing.T) {
	server := &cloneCensusMilvusServer{
		databases:   []string{"default"},
		collections: []string{"loaded"},
		loadStates: map[string]commonpb.LoadState{
			"loaded": commonpb.LoadState_LoadStateLoaded,
		},
		rowCounts:     map[string]int64{"loaded": 1},
		logicalCounts: map[string]int64{"loaded": 1},
		rows: map[string][]cloneCensusRow{
			"loaded": {{identity: "loaded-1", vector: []float32{1, 2}}},
		},
	}
	address := startCloneCensusServer(t, server)
	settings := cloneMilvusSettings{Address: address, Database: "default"}
	identity := collectionIdentity{Database: "default", Collection: "loaded"}

	baseline, err := readCloneMilvusCensus(context.Background(), settings)
	if err != nil {
		t.Fatalf("read baseline census: %v", err)
	}
	server.mutex.Lock()
	server.rowCounts["loaded"] = 2
	server.mutex.Unlock()
	changed, err := readCloneMilvusCensus(context.Background(), settings)
	if err != nil {
		t.Fatalf("read logical-count-changed census: %v", err)
	}
	if changed.Samples[identity] != baseline.Samples[identity] {
		t.Fatal("physical row-count change altered the stable row sample")
	}
	if baseline.RowCounts[identity] != 1 || changed.RowCounts[identity] != 2 {
		t.Fatalf("row counts = %d then %d, want 1 then 2", baseline.RowCounts[identity], changed.RowCounts[identity])
	}
}

type cloneCensusMilvusServer struct {
	milvuspb.UnimplementedMilvusServiceServer

	mutex              sync.Mutex
	databases          []string
	collections        []string
	loadStates         map[string]commonpb.LoadState
	rowCounts          map[string]int64
	logicalCounts      map[string]int64
	rows               map[string][]cloneCensusRow
	reverseRows        bool
	methods            []string
	requestDatabases   []string
	queriedCollections []string
	rejectSparseOutput bool
}

type cloneCensusRow struct {
	identity string
	vector   []float32
	sparse   float32
}

func (server *cloneCensusMilvusServer) record(method string) {
	server.mutex.Lock()
	server.methods = append(server.methods, method)
	server.mutex.Unlock()
}

func (server *cloneCensusMilvusServer) Connect(context.Context, *milvuspb.ConnectRequest) (*milvuspb.ConnectResponse, error) {
	server.record("Connect")
	return &milvuspb.ConnectResponse{Status: cloneCensusSuccess(), Identifier: 1}, nil
}

func (server *cloneCensusMilvusServer) ListDatabases(context.Context, *milvuspb.ListDatabasesRequest) (*milvuspb.ListDatabasesResponse, error) {
	server.record("ListDatabases")
	server.mutex.Lock()
	databases := slices.Clone(server.databases)
	server.mutex.Unlock()
	return &milvuspb.ListDatabasesResponse{Status: cloneCensusSuccess(), DbNames: databases}, nil
}

func (server *cloneCensusMilvusServer) ShowCollections(ctx context.Context, request *milvuspb.ShowCollectionsRequest) (*milvuspb.ShowCollectionsResponse, error) {
	server.record("ShowCollections")
	database := request.GetDbName()
	if database == "" {
		incoming, _ := metadata.FromIncomingContext(ctx)
		if values := incoming.Get("dbname"); len(values) > 0 {
			database = values[0]
		}
	}
	server.mutex.Lock()
	server.requestDatabases = append(server.requestDatabases, database)
	server.mutex.Unlock()
	if database != "default" {
		return &milvuspb.ShowCollectionsResponse{Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_IllegalArgument, Reason: "wrong database"}}, nil
	}
	server.mutex.Lock()
	collections := slices.Clone(server.collections)
	server.mutex.Unlock()
	return &milvuspb.ShowCollectionsResponse{Status: cloneCensusSuccess(), CollectionNames: collections}, nil
}

func (server *cloneCensusMilvusServer) GetLoadState(_ context.Context, request *milvuspb.GetLoadStateRequest) (*milvuspb.GetLoadStateResponse, error) {
	server.record("GetLoadState")
	server.mutex.Lock()
	loadState := server.loadStates[request.GetCollectionName()]
	server.mutex.Unlock()
	return &milvuspb.GetLoadStateResponse{Status: cloneCensusSuccess(), State: loadState}, nil
}

func (server *cloneCensusMilvusServer) DescribeCollection(_ context.Context, request *milvuspb.DescribeCollectionRequest) (*milvuspb.DescribeCollectionResponse, error) {
	server.record("DescribeCollection")
	schema := entity.NewSchema().
		WithName(request.GetCollectionName()).
		WithField(entity.NewField().
			WithName("id").
			WithDataType(entity.FieldTypeVarChar).
			WithIsPrimaryKey(true).
			WithTypeParams("max_length", "128")).
		WithField(entity.NewField().
			WithName("dense").
			WithDataType(entity.FieldTypeFloatVector).
			WithDim(2)).
		WithField(entity.NewField().
			WithName("sparse").
			WithDataType(entity.FieldTypeSparseVector))
	return &milvuspb.DescribeCollectionResponse{
		Status:         cloneCensusSuccess(),
		CollectionName: request.GetCollectionName(),
		Schema:         schema.ProtoMessage(),
		ShardsNum:      2,
		Properties: []*commonpb.KeyValuePair{
			{Key: "mmap.enabled", Value: "true"},
			{Key: "collection.ttl.seconds", Value: "60"},
		},
	}, nil
}

func (server *cloneCensusMilvusServer) GetCollectionStatistics(
	_ context.Context,
	request *milvuspb.GetCollectionStatisticsRequest,
) (*milvuspb.GetCollectionStatisticsResponse, error) {
	server.record("GetCollectionStatistics")
	server.mutex.Lock()
	rowCount := server.rowCounts[request.GetCollectionName()]
	server.mutex.Unlock()
	return &milvuspb.GetCollectionStatisticsResponse{
		Status: cloneCensusSuccess(),
		Stats:  []*commonpb.KeyValuePair{{Key: "row_count", Value: strconv.FormatInt(rowCount, 10)}},
	}, nil
}

func (server *cloneCensusMilvusServer) Query(
	_ context.Context,
	request *milvuspb.QueryRequest,
) (*milvuspb.QueryResults, error) {
	server.record("Query")
	if server.rejectSparseOutput && slices.Contains(request.GetOutputFields(), "sparse") {
		return &milvuspb.QueryResults{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_IllegalArgument,
				Reason:    "not allowed to retrieve raw data of field sparse",
			},
		}, nil
	}
	server.mutex.Lock()
	server.queriedCollections = append(server.queriedCollections, request.GetCollectionName())
	logicalCount, hasLogicalCount := server.logicalCounts[request.GetCollectionName()]
	rows := slices.Clone(server.rows[request.GetCollectionName()])
	reverseRows := server.reverseRows
	server.mutex.Unlock()
	if slices.Contains(request.GetOutputFields(), "count(*)") {
		if !hasLogicalCount {
			server.mutex.Lock()
			logicalCount = server.rowCounts[request.GetCollectionName()]
			server.mutex.Unlock()
		}
		return &milvuspb.QueryResults{
			Status: cloneCensusSuccess(),
			FieldsData: []*schemapb.FieldData{
				column.NewColumnInt64("count(*)", []int64{logicalCount}).FieldData(),
			},
		}, nil
	}
	if reverseRows {
		slices.Reverse(rows)
	}
	identities := make([]string, 0, len(rows))
	vectors := make([][]float32, 0, len(rows))
	sparseVectors := make([]entity.SparseEmbedding, 0, len(rows))
	for _, row := range rows {
		identities = append(identities, row.identity)
		vectors = append(vectors, slices.Clone(row.vector))
		sparseVector, err := entity.NewSliceSparseEmbedding([]uint32{0}, []float32{row.sparse})
		if err != nil {
			return nil, err
		}
		sparseVectors = append(sparseVectors, sparseVector)
	}
	return &milvuspb.QueryResults{
		Status: cloneCensusSuccess(),
		FieldsData: []*schemapb.FieldData{
			column.NewColumnVarChar("id", identities).FieldData(),
			column.NewColumnFloatVector("dense", 2, vectors).FieldData(),
			column.NewColumnSparseVectors("sparse", sparseVectors).FieldData(),
		},
	}, nil
}

func (server *cloneCensusMilvusServer) DescribeIndex(_ context.Context, request *milvuspb.DescribeIndexRequest) (*milvuspb.DescribeIndexResponse, error) {
	server.record("DescribeIndex")
	description := &milvuspb.IndexDescription{
		IndexName: "dense_idx",
		FieldName: "dense",
		State:     commonpb.IndexState_Finished,
		Params: []*commonpb.KeyValuePair{
			{Key: "index_type", Value: "HNSW"},
			{Key: "mmap.enabled", Value: "true"},
		},
	}
	if request.GetIndexName() != "" && request.GetIndexName() != description.GetIndexName() {
		return &milvuspb.DescribeIndexResponse{Status: cloneCensusSuccess()}, nil
	}
	return &milvuspb.DescribeIndexResponse{Status: cloneCensusSuccess(), IndexDescriptions: []*milvuspb.IndexDescription{description}}, nil
}

func startCloneCensusServer(t *testing.T, server *cloneCensusMilvusServer) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	grpcServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(grpcServer, server)
	go func() { _ = grpcServer.Serve(listener) }()
	t.Cleanup(grpcServer.Stop)
	return listener.Addr().String()
}

func cloneCensusSuccess() *commonpb.Status {
	return &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}
}

func TestCloneCensusTimeoutIsBounded(t *testing.T) {
	settings := cloneMilvusSettings{Address: "127.0.0.1:1", Database: "default"}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	started := time.Now()
	_, err := readCloneMilvusCensus(ctx, settings)
	if err == nil {
		t.Fatal("unreachable Milvus census succeeded")
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("unreachable census took %s", elapsed)
	}
}
