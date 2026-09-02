package semantic

import (
	"context"
	"fmt"
	"net"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/embedding"
	"goodkind.io/lm-semantic-search/internal/model"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const reuseLookupTestDimension = 4096

type recordedReuseRequest struct {
	Filter       string
	OutputFields []string
	IDs          []string
}

type reuseLookupRow struct {
	id             string
	content        string
	contentHash    *string
	embeddingModel *string
	vector         []float32
}

type reuseLookupServer struct {
	milvuspb.UnimplementedMilvusServiceServer

	mutex         sync.Mutex
	rows          []reuseLookupRow
	requests      []recordedReuseRequest
	consistencies []commonpb.ConsistencyLevel
	queryErr      error
	getErr        error
	getResponse   func([]string, []reuseLookupRow) *milvuspb.QueryResults
	mutationCalls atomic.Int32
}

type reuseLookupCountingEmbedder struct {
	calls atomic.Int32
}

func (embedder *reuseLookupCountingEmbedder) Embed(
	context.Context,
	string,
) ([]float32, error) {
	embedder.calls.Add(1)
	return make([]float32, reuseLookupTestDimension), nil
}

func (embedder *reuseLookupCountingEmbedder) EmbedBatch(
	context.Context,
	[]string,
) (embedding.BatchResult, error) {
	embedder.calls.Add(1)
	return embedding.BatchResult{}, nil
}

func (*reuseLookupCountingEmbedder) ProviderName() model.EmbeddingProvider {
	return "reuse-lookup-counting"
}

func (*reuseLookupCountingEmbedder) Health(context.Context) error {
	return nil
}

func (embedder *reuseLookupCountingEmbedder) callCount() int32 {
	return embedder.calls.Load()
}

func (server *reuseLookupServer) Connect(
	context.Context,
	*milvuspb.ConnectRequest,
) (*milvuspb.ConnectResponse, error) {
	return &milvuspb.ConnectResponse{
		Status:     reuseLookupSuccessStatus(),
		Identifier: 1,
	}, nil
}

func (server *reuseLookupServer) DescribeCollection(
	_ context.Context,
	request *milvuspb.DescribeCollectionRequest,
) (*milvuspb.DescribeCollectionResponse, error) {
	if strings.HasPrefix(request.GetCollectionName(), reuseCatalogCollectionPrefix) {
		return &milvuspb.DescribeCollectionResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_CollectionNotExists,
				Reason:    "collection not found",
			},
		}, nil
	}
	return &milvuspb.DescribeCollectionResponse{
		Status:         reuseLookupSuccessStatus(),
		CollectionName: request.GetCollectionName(),
		Schema: &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
			{Name: idFieldName, DataType: schemapb.DataType_VarChar, IsPrimaryKey: true},
			{Name: contentFieldName, DataType: schemapb.DataType_VarChar},
			{Name: contentHashFieldName, DataType: schemapb.DataType_VarChar, Nullable: true},
			{Name: embeddingModelFieldName, DataType: schemapb.DataType_VarChar, Nullable: true},
			{
				Name:     denseVectorFieldName,
				DataType: schemapb.DataType_FloatVector,
				TypeParams: []*commonpb.KeyValuePair{{
					Key:   "dim",
					Value: strconv.Itoa(reuseLookupTestDimension),
				}},
			},
		}},
	}, nil
}

func (*reuseLookupServer) HasCollection(
	_ context.Context,
	request *milvuspb.HasCollectionRequest,
) (*milvuspb.BoolResponse, error) {
	return &milvuspb.BoolResponse{
		Status: reuseLookupSuccessStatus(),
		Value:  !strings.HasPrefix(request.GetCollectionName(), reuseCatalogCollectionPrefix),
	}, nil
}

func (server *reuseLookupServer) Insert(
	context.Context,
	*milvuspb.InsertRequest,
) (*milvuspb.MutationResult, error) {
	server.mutationCalls.Add(1)
	return &milvuspb.MutationResult{Status: reuseLookupSuccessStatus()}, nil
}

func (server *reuseLookupServer) Delete(
	context.Context,
	*milvuspb.DeleteRequest,
) (*milvuspb.MutationResult, error) {
	server.mutationCalls.Add(1)
	return &milvuspb.MutationResult{Status: reuseLookupSuccessStatus()}, nil
}

func (server *reuseLookupServer) Upsert(
	context.Context,
	*milvuspb.UpsertRequest,
) (*milvuspb.MutationResult, error) {
	server.mutationCalls.Add(1)
	return &milvuspb.MutationResult{Status: reuseLookupSuccessStatus()}, nil
}

func (server *reuseLookupServer) mutationCallCount() int32 {
	return server.mutationCalls.Load()
}

func (server *reuseLookupServer) Query(
	_ context.Context,
	request *milvuspb.QueryRequest,
) (*milvuspb.QueryResults, error) {
	ids := reuseLookupRequestIDs(request.GetExpr())
	server.mutex.Lock()
	server.requests = append(server.requests, recordedReuseRequest{
		Filter:       request.GetExpr(),
		OutputFields: slices.Clone(request.GetOutputFields()),
		IDs:          ids,
	})
	server.consistencies = append(server.consistencies, request.GetConsistencyLevel())
	rows := server.matchingRows(request.GetExpr(), ids)
	queryErr := server.queryErr
	getErr := server.getErr
	getResponse := server.getResponse
	server.mutex.Unlock()

	if slices.Contains(request.GetOutputFields(), denseVectorFieldName) && len(ids) == 0 {
		return nil, status.Error(codes.InvalidArgument, "scalar candidate discovery requested vector")
	}
	if len(ids) == 0 && queryErr != nil {
		return nil, queryErr
	}
	if len(ids) > 0 && getErr != nil {
		return nil, getErr
	}
	if len(ids) > 0 && getResponse != nil {
		return getResponse(request.GetOutputFields(), rows), nil
	}
	return reuseLookupQueryResults(request.GetOutputFields(), rows), nil
}

func (server *reuseLookupServer) matchingRows(filter string, ids []string) []reuseLookupRow {
	rows := make([]reuseLookupRow, 0, len(server.rows))
	for _, row := range server.rows {
		if slices.Contains(ids, row.id) {
			rows = append(rows, row)
		}
	}
	if len(ids) == 0 && strings.Contains(filter, contentHashFieldName+" is null") {
		contents := reuseLookupFilterValues(filter, contentFieldName)
		rows = rows[:0]
		for _, row := range server.rows {
			if row.contentHash == nil && slices.Contains(contents, row.content) {
				rows = append(rows, row)
			}
		}
	}
	if len(ids) == 0 && !strings.Contains(filter, contentHashFieldName+" is null") {
		hashes := reuseLookupFilterValues(filter, contentHashFieldName)
		rows = rows[:0]
		for _, row := range server.rows {
			if row.contentHash != nil && slices.Contains(hashes, *row.contentHash) {
				rows = append(rows, row)
			}
		}
	}
	minimumID, hasMinimumID := reuseLookupFilterGreaterThan(filter, idFieldName)
	if !hasMinimumID {
		return rows
	}
	filteredRows := rows[:0]
	for _, row := range rows {
		if row.id > minimumID {
			filteredRows = append(filteredRows, row)
		}
	}
	return filteredRows
}

func (server *reuseLookupServer) snapshot() ([]recordedReuseRequest, []commonpb.ConsistencyLevel) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	return slices.Clone(server.requests), slices.Clone(server.consistencies)
}

func reuseLookupRequestIDs(filter string) []string {
	return reuseLookupFilterValues(filter, idFieldName)
}

func reuseLookupFilterValues(filter string, fieldName string) []string {
	prefix := fieldName + " in ["
	start := strings.Index(filter, prefix)
	isList := true
	if start < 0 {
		prefix = fieldName + " == "
		start = strings.Index(filter, prefix)
		isList = false
	}
	if start < 0 {
		return nil
	}
	start += len(prefix)
	var ids []string
	for index := start; index < len(filter); {
		for index < len(filter) && filter[index] == ' ' {
			index++
		}
		if index >= len(filter) || filter[index] == ']' {
			return ids
		}
		if filter[index] != '"' {
			return nil
		}
		valueStart := index
		index++
		for index < len(filter) {
			if filter[index] == '\\' {
				index += 2
				continue
			}
			if filter[index] == '"' {
				break
			}
			index++
		}
		if index >= len(filter) {
			return nil
		}
		value, err := strconv.Unquote(filter[valueStart : index+1])
		if err != nil {
			return nil
		}
		ids = append(ids, value)
		if !isList {
			return ids
		}
		index++
		for index < len(filter) && filter[index] == ' ' {
			index++
		}
		if index < len(filter) && filter[index] == ',' {
			index++
		}
	}
	return ids
}

func reuseLookupFilterGreaterThan(filter string, fieldName string) (string, bool) {
	prefix := fieldName + " > "
	start := strings.LastIndex(filter, prefix)
	if start < 0 {
		return "", false
	}
	start += len(prefix)
	if start >= len(filter) || filter[start] != '"' {
		return "", false
	}
	end := start + 1
	for end < len(filter) {
		if filter[end] == '\\' {
			end += 2
			continue
		}
		if filter[end] == '"' {
			break
		}
		end++
	}
	if end >= len(filter) {
		return "", false
	}
	value, err := strconv.Unquote(filter[start : end+1])
	if err != nil {
		return "", false
	}
	return value, true
}

func reuseLookupQueryResults(outputFields []string, rows []reuseLookupRow) *milvuspb.QueryResults {
	fields := make([]*schemapb.FieldData, 0, len(outputFields))
	for _, outputField := range outputFields {
		switch outputField {
		case idFieldName:
			ids := make([]string, 0, len(rows))
			for _, row := range rows {
				ids = append(ids, row.id)
			}
			fields = append(fields, column.NewColumnVarChar(idFieldName, ids).FieldData())
		case contentFieldName:
			contents := make([]string, 0, len(rows))
			for _, row := range rows {
				contents = append(contents, row.content)
			}
			fields = append(fields, column.NewColumnVarChar(contentFieldName, contents).FieldData())
		case contentHashFieldName:
			hashes := make([]string, 0, len(rows))
			valid := make([]bool, 0, len(rows))
			for _, row := range rows {
				if row.contentHash == nil {
					hashes = append(hashes, "")
					valid = append(valid, false)
					continue
				}
				hashes = append(hashes, *row.contentHash)
				valid = append(valid, true)
			}
			field := column.NewColumnVarChar(contentHashFieldName, hashes).FieldData()
			field.ValidData = valid
			fields = append(fields, field)
		case embeddingModelFieldName:
			models := make([]string, 0, len(rows))
			valid := make([]bool, 0, len(rows))
			for _, row := range rows {
				if row.embeddingModel == nil {
					models = append(models, "")
					valid = append(valid, false)
					continue
				}
				models = append(models, *row.embeddingModel)
				valid = append(valid, true)
			}
			field := column.NewColumnVarChar(embeddingModelFieldName, models).FieldData()
			field.ValidData = valid
			fields = append(fields, field)
		case denseVectorFieldName:
			vectors := make([][]float32, 0, len(rows))
			for _, row := range rows {
				if row.vector == nil {
					vectors = append(vectors, make([]float32, reuseLookupTestDimension))
					continue
				}
				vectors = append(vectors, row.vector)
			}
			dimension := reuseLookupTestDimension
			if len(vectors) > 0 {
				dimension = len(vectors[0])
			}
			fields = append(fields, column.NewColumnFloatVector(denseVectorFieldName, dimension, vectors).FieldData())
		}
	}
	return &milvuspb.QueryResults{
		Status:       reuseLookupSuccessStatus(),
		FieldsData:   fields,
		OutputFields: outputFields,
	}
}

func reuseLookupSuccessStatus() *commonpb.Status {
	return &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}
}

func newReuseLookupTestService(t *testing.T, rows []reuseLookupRow) (*Service, *reuseLookupServer) {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen for fake Milvus: %v", err)
	}
	server := &reuseLookupServer{rows: rows}
	grpcServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(grpcServer, server)
	go func() {
		_ = grpcServer.Serve(listener)
	}()
	t.Cleanup(grpcServer.Stop)

	client, err := milvusclient.New(context.Background(), &milvusclient.ClientConfig{
		Address: listener.Addr().String(),
	})
	if err != nil {
		t.Fatalf("construct Milvus client: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := client.Close(context.Background()); closeErr != nil {
			t.Errorf("close Milvus client: %v", closeErr)
		}
	})
	return &Service{
		cfg: config.Config{
			EmbeddingDimension: reuseLookupTestDimension,
			EmbeddingModel:     "current-model",
		},
		milvus: client,
	}, server
}

func newPublicReuseLookupTestService(
	t *testing.T,
	rows []reuseLookupRow,
) (*Service, *reuseLookupServer, *reuseLookupCountingEmbedder) {
	t.Helper()
	service, server := newReuseLookupTestService(t, rows)
	embedder := &reuseLookupCountingEmbedder{}
	service.embedder = embedder
	service.available.Store(true)
	reuseIdentity := &reuseIdentityMigration{}
	reuseIdentity.once.Do(func() {})
	service.ensuredReuseIdentityColumns.Store("reuse_source", reuseIdentity)
	service.residency = newCollectionResidencyController(residencyControllerConfig{
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
		if err := service.residency.Close(context.Background()); err != nil {
			t.Errorf("close reuse lookup residency controller: %v", err)
		}
	})
	return service, server, embedder
}

func loadReuseLookupRows(t *testing.T, service *Service, contents []string) map[string][]float32 {
	t.Helper()
	contentsByHash := make(map[string]string, len(contents))
	for _, content := range contents {
		contentsByHash[contentHash(content)] = content
	}
	reuse := make(map[string][]float32)
	if err := service.loadCollectionReuseFromSource(
		context.Background(),
		"reuse_source",
		contentsByHash,
		reuse,
	); err != nil {
		t.Fatalf("load collection reuse: %v", err)
	}
	return reuse
}

func loadReuseLookupRowsResult(
	service *Service,
	contents []string,
) (map[string][]float32, error) {
	contentsByHash := make(map[string]string, len(contents))
	for _, content := range contents {
		contentsByHash[contentHash(content)] = content
	}
	reuse := make(map[string][]float32)
	err := service.loadCollectionReuseFromSource(
		context.Background(),
		"reuse_source",
		contentsByHash,
		reuse,
	)
	return reuse, err
}

func TestReuseVectorFetchBatchFitsExactBudget(t *testing.T) {
	ids := []string{"first", "second"}
	batches, err := packReuseVectorCandidateIDs(
		ids,
		reuseVectorFetchBudgetBytes/2,
		reuseVectorFetchBudgetBytes,
	)
	if err != nil {
		t.Fatalf("pack exact-budget candidates: %v", err)
	}
	if len(batches) != 1 || !slices.Equal(batches[0], ids) {
		t.Fatalf("batches = %v, want one exact-budget batch", batches)
	}
}

func TestReuseVectorFetchBatchSplitsOneByteOverBudget(t *testing.T) {
	ids := []string{"first", "second"}
	batches, err := packReuseVectorCandidateIDs(
		ids,
		reuseVectorFetchBudgetBytes/2+1,
		reuseVectorFetchBudgetBytes,
	)
	if err != nil {
		t.Fatalf("pack over-budget candidates: %v", err)
	}
	if len(batches) != 2 || !slices.Equal(batches[0], ids[:1]) ||
		!slices.Equal(batches[1], ids[1:]) {
		t.Fatalf("batches = %v, want two nonempty batches", batches)
	}
}

func TestReuseVectorFetchBatchNeverReturnsEmptyBatch(t *testing.T) {
	candidates := reuseVectorCandidateByID{
		"third":  {id: "third"},
		"first":  {id: "first"},
		"second": {id: "second"},
	}
	batches, err := packReuseVectorCandidates(candidates, reuseLookupTestDimension)
	if err != nil {
		t.Fatalf("pack candidates: %v", err)
	}
	for batchIndex, batch := range batches {
		if len(batch) == 0 {
			t.Fatalf("batch %d is empty", batchIndex)
		}
	}
	flattened := make([]string, 0, len(candidates))
	rowBytes, err := estimatedReuseVectorRowBytes(reuseLookupTestDimension)
	if err != nil {
		t.Fatalf("estimate candidate bytes: %v", err)
	}
	for _, batch := range batches {
		if int64(len(batch))*rowBytes > reuseVectorFetchBudgetBytes {
			t.Fatalf("batch estimated bytes exceed budget: %d", int64(len(batch))*rowBytes)
		}
		flattened = append(flattened, batch...)
	}
	if !slices.Equal(flattened, []string{"first", "second", "third"}) {
		t.Fatalf("packed IDs = %v, want deterministic order", flattened)
	}
}

func TestReuseVectorFetchBatchesSelectedIDs(t *testing.T) {
	firstContent := "first selected content"
	secondContent := "second selected content"
	service, server := newReuseLookupTestService(t, []reuseLookupRow{
		{
			id:             "second",
			content:        secondContent,
			contentHash:    ptr(contentHash(secondContent)),
			embeddingModel: ptr("current-model"),
		},
		{
			id:             "first",
			content:        firstContent,
			contentHash:    ptr(contentHash(firstContent)),
			embeddingModel: ptr("current-model"),
		},
	})

	reuse := loadReuseLookupRows(t, service, []string{secondContent, firstContent})
	if len(reuse) != 2 {
		t.Fatalf("reuse vector count = %d, want 2", len(reuse))
	}
	requests, _ := server.snapshot()
	getRequests := make([]recordedReuseRequest, 0)
	for _, request := range requests {
		if len(request.IDs) > 0 && slices.Contains(request.OutputFields, denseVectorFieldName) {
			getRequests = append(getRequests, request)
		}
	}
	if len(getRequests) != 1 {
		t.Fatalf("selected vector Get requests = %d, want 1", len(getRequests))
	}
	if !slices.Equal(getRequests[0].IDs, []string{"first", "second"}) {
		t.Fatalf("selected vector Get IDs = %v, want deterministic batch", getRequests[0].IDs)
	}
}

func TestReuseVectorFetchBatchRejectsOversizedRow(t *testing.T) {
	candidates := reuseVectorCandidateByID{"oversized": {id: "oversized"}}
	dimension := int(reuseVectorFetchBudgetBytes / 4)
	if _, err := packReuseVectorCandidates(candidates, dimension); err == nil {
		t.Fatal("pack oversized row succeeded, want error")
	}
	if _, err := packReuseVectorCandidates(candidates, 0); err == nil {
		t.Fatal("pack zero dimension succeeded, want error")
	}
}

func TestReuseSelectedRowValidation(t *testing.T) {
	content := "selected content"
	requestedRow := reuseLookupRow{
		id:             "selected",
		content:        content,
		contentHash:    ptr(contentHash(content)),
		embeddingModel: ptr("current-model"),
	}
	tests := []struct {
		name      string
		configure func(*reuseLookupServer)
	}{
		{
			name: "missing column",
			configure: func(server *reuseLookupServer) {
				server.getResponse = func(outputFields []string, rows []reuseLookupRow) *milvuspb.QueryResults {
					result := reuseLookupQueryResults(outputFields, rows)
					result.FieldsData = result.FieldsData[:len(result.FieldsData)-1]
					return result
				}
			},
		},
		{
			name: "unknown ID",
			configure: func(server *reuseLookupServer) {
				server.getResponse = func(outputFields []string, _ []reuseLookupRow) *milvuspb.QueryResults {
					row := requestedRow
					row.id = "unknown"
					return reuseLookupQueryResults(outputFields, []reuseLookupRow{row})
				}
			},
		},
		{
			name: "duplicate ID",
			configure: func(server *reuseLookupServer) {
				server.getResponse = func(outputFields []string, _ []reuseLookupRow) *milvuspb.QueryResults {
					return reuseLookupQueryResults(outputFields, []reuseLookupRow{requestedRow, requestedRow})
				}
			},
		},
		{
			name: "hash collision",
			configure: func(server *reuseLookupServer) {
				server.getResponse = func(outputFields []string, _ []reuseLookupRow) *milvuspb.QueryResults {
					row := requestedRow
					row.content = "different content"
					return reuseLookupQueryResults(outputFields, []reuseLookupRow{row})
				}
			},
		},
		{
			name: "nullable model decode failure",
			configure: func(server *reuseLookupServer) {
				server.getResponse = func(outputFields []string, rows []reuseLookupRow) *milvuspb.QueryResults {
					result := reuseLookupQueryResults(outputFields, rows)
					for fieldIndex, field := range result.FieldsData {
						if field.GetFieldName() == embeddingModelFieldName {
							result.FieldsData[fieldIndex] = column.NewColumnInt64(
								embeddingModelFieldName,
								[]int64{1},
							).FieldData()
						}
					}
					return result
				}
			},
		},
		{
			name: "vector decode failure",
			configure: func(server *reuseLookupServer) {
				server.getResponse = func(outputFields []string, rows []reuseLookupRow) *milvuspb.QueryResults {
					result := reuseLookupQueryResults(outputFields, rows)
					for fieldIndex, field := range result.FieldsData {
						if field.GetFieldName() == denseVectorFieldName {
							result.FieldsData[fieldIndex] = column.NewColumnVarChar(
								denseVectorFieldName,
								[]string{"not a vector"},
							).FieldData()
						}
					}
					return result
				}
			},
		},
		{
			name: "dimension mismatch",
			configure: func(server *reuseLookupServer) {
				server.getResponse = func(outputFields []string, _ []reuseLookupRow) *milvuspb.QueryResults {
					row := requestedRow
					row.vector = make([]float32, reuseLookupTestDimension-1)
					return reuseLookupQueryResults(outputFields, []reuseLookupRow{row})
				}
			},
		},
		{
			name: "incompatible model identity",
			configure: func(server *reuseLookupServer) {
				server.getResponse = func(outputFields []string, _ []reuseLookupRow) *milvuspb.QueryResults {
					row := requestedRow
					row.embeddingModel = ptr("different-model")
					return reuseLookupQueryResults(outputFields, []reuseLookupRow{row})
				}
			},
		},
		{
			name: "Query failure",
			configure: func(server *reuseLookupServer) {
				server.queryErr = status.Error(codes.Unavailable, "query failed")
			},
		},
		{
			name: "Get failure",
			configure: func(server *reuseLookupServer) {
				server.getErr = status.Error(codes.Unavailable, "get failed")
			},
		},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			service, server := newReuseLookupTestService(t, []reuseLookupRow{requestedRow})
			testCase.configure(server)
			reuse, err := loadReuseLookupRowsResult(service, []string{content})
			if err == nil {
				t.Fatal("reuse lookup succeeded, want error")
			}
			if len(reuse) != 0 {
				t.Fatalf("reuse vectors = %d, want none after validation failure", len(reuse))
			}
		})
	}
}

func TestReuseReadErrorHasNoSideEffects(t *testing.T) {
	content := "error boundary content"
	row := reuseLookupRow{
		id:             "selected",
		content:        content,
		contentHash:    ptr(contentHash(content)),
		embeddingModel: ptr("current-model"),
	}
	tests := []struct {
		name      string
		configure func(*reuseLookupServer)
		wantText  string
	}{
		{
			name: "scalar Query",
			configure: func(server *reuseLookupServer) {
				server.queryErr = status.Error(codes.Unavailable, "authoritative query failure")
			},
			wantText: "authoritative query failure",
		},
		{
			name: "selected Get",
			configure: func(server *reuseLookupServer) {
				server.getErr = status.Error(codes.Unavailable, "authoritative get failure")
			},
			wantText: "authoritative get failure",
		},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			service, server, embedder := newPublicReuseLookupTestService(t, []reuseLookupRow{row})
			testCase.configure(server)

			reuse, err := service.LoadReuseVectorsForContents(
				context.Background(),
				"reuse_source",
				[]model.StoredChunk{{Content: content}},
			)
			if status.Code(err) != codes.Unavailable || !strings.Contains(err.Error(), testCase.wantText) {
				t.Fatalf("LoadReuseVectorsForContents error = %v, want unavailable %q", err, testCase.wantText)
			}
			if reuse != nil {
				t.Fatalf("reuse = %v, want nil on authoritative read error", reuse)
			}
			if embedder.callCount() != 0 {
				t.Fatalf("embedding calls = %d, want 0", embedder.callCount())
			}
			if server.mutationCallCount() != 0 {
				t.Fatalf("Milvus mutation calls = %d, want 0", server.mutationCallCount())
			}
		})
	}
}

func TestReuseMissingSelectedIDHasNoSideEffects(t *testing.T) {
	content := "missing selected ID content"
	row := reuseLookupRow{
		id:             "selected",
		content:        content,
		contentHash:    ptr(contentHash(content)),
		embeddingModel: ptr("current-model"),
	}
	service, server, embedder := newPublicReuseLookupTestService(t, []reuseLookupRow{row})
	server.getResponse = func(outputFields []string, _ []reuseLookupRow) *milvuspb.QueryResults {
		return reuseLookupQueryResults(outputFields, nil)
	}

	reuse, err := service.LoadReuseVectorsForContents(
		context.Background(),
		"reuse_source",
		[]model.StoredChunk{{Content: content}},
	)
	if err == nil || !strings.Contains(err.Error(), `omitted requested ID "selected"`) {
		t.Fatalf("LoadReuseVectorsForContents error = %v, want omitted selected ID", err)
	}
	if reuse != nil {
		t.Fatalf("reuse = %v, want nil on authoritative read error", reuse)
	}
	if embedder.callCount() != 0 {
		t.Fatalf("embedding calls = %d, want 0", embedder.callCount())
	}
	if server.mutationCallCount() != 0 {
		t.Fatalf("Milvus mutation calls = %d, want 0", server.mutationCallCount())
	}
}

func TestReuseScalarCandidateDiscoveryNeverRequestsVectors(t *testing.T) {
	rows := make([]reuseLookupRow, 0, 16384)
	for index := range 16384 {
		rows = append(rows, reuseLookupRow{
			id:      fmt.Sprintf("legacy-%05d", index),
			content: "duplicate legacy content",
		})
	}
	service, server := newReuseLookupTestService(t, rows)

	_ = loadReuseLookupRows(t, service, []string{"duplicate legacy content"})

	requests, _ := server.snapshot()
	for _, request := range requests {
		if len(request.IDs) == 0 && slices.Contains(request.OutputFields, denseVectorFieldName) {
			t.Fatalf("scalar candidate request selected vector: %+v", request)
		}
	}
}

func TestReuseModernCandidateDiscoverySelectsOneCompatibleID(t *testing.T) {
	for _, testCase := range []struct {
		name           string
		embeddingModel *string
		wantReuse      bool
	}{
		{name: "null", wantReuse: true},
		{name: "empty", embeddingModel: ptr(""), wantReuse: true},
		{name: "equal", embeddingModel: ptr("current-model"), wantReuse: true},
		{name: "unequal", embeddingModel: ptr("different-model"), wantReuse: false},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			content := "modern " + testCase.name
			service, server := newReuseLookupTestService(t, []reuseLookupRow{{
				id:             testCase.name,
				content:        content,
				contentHash:    ptr(contentHash(content)),
				embeddingModel: testCase.embeddingModel,
			}})

			reuse := loadReuseLookupRows(t, service, []string{content})
			if testCase.wantReuse && len(reuse) != 1 {
				t.Fatalf("reuse vector count = %d, want 1", len(reuse))
			}
			if !testCase.wantReuse && len(reuse) != 0 {
				t.Fatalf("reuse vector count = %d, want 0", len(reuse))
			}
			requests, _ := server.snapshot()
			ids := reuseLookupGetIDs(requests)
			if testCase.wantReuse && !slices.Equal(ids, []string{testCase.name}) {
				t.Fatalf("vector Get ids = %v, want %q", ids, testCase.name)
			}
			if !testCase.wantReuse && len(ids) != 0 {
				t.Fatalf("vector Get ids = %v, want none", ids)
			}
		})
	}
}

func TestReuseLegacyCandidateDiscoveryQueriesOneContentAtATime(t *testing.T) {
	service, server := newReuseLookupTestService(t, []reuseLookupRow{
		{id: "legacy-first", content: "legacy first, [second]"},
		{id: "legacy-second", content: "legacy second"},
	})

	reuse := loadReuseLookupRows(t, service, []string{"legacy first, [second]", "legacy second"})
	if len(reuse) != 2 {
		t.Fatalf("reuse vector count = %d, want 2", len(reuse))
	}

	requests, _ := server.snapshot()
	legacyFilters := make([]string, 0)
	for _, request := range requests {
		if strings.Contains(request.Filter, contentHashFieldName+" is null") {
			legacyFilters = append(legacyFilters, request.Filter)
		}
	}
	if len(legacyFilters) != 2 {
		t.Fatalf("legacy candidate queries = %d, want 2", len(legacyFilters))
	}
	for _, filter := range legacyFilters {
		if strings.Count(filter, "\"") != 2 {
			t.Fatalf("legacy candidate filter = %q, want one content", filter)
		}
	}
}

func TestReuseCandidateDiscoveryRejectsKnownUnequalModel(t *testing.T) {
	unequalModel := "different-model"
	service, server := newReuseLookupTestService(t, []reuseLookupRow{{
		id:             "unequal",
		content:        "known unequal content",
		contentHash:    ptr(contentHash("known unequal content")),
		embeddingModel: &unequalModel,
	}})

	reuse := loadReuseLookupRows(t, service, []string{"known unequal content"})

	if len(reuse) != 0 {
		t.Fatalf("reuse vector count = %d, want 0", len(reuse))
	}
	requests, _ := server.snapshot()
	if ids := reuseLookupGetIDs(requests); len(ids) != 0 {
		t.Fatalf("vector Get ids = %v, included known unequal model", ids)
	}
}

func reuseLookupGetIDs(requests []recordedReuseRequest) []string {
	var ids []string
	for _, request := range requests {
		if len(request.IDs) > 0 && slices.Contains(request.OutputFields, denseVectorFieldName) {
			ids = append(ids, request.IDs...)
		}
	}
	return ids
}

func ptr(value string) *string {
	return &value
}
