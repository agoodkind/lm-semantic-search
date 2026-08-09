package semantic_test

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/client/v2/entity"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
)

const mmapTestCollection = "hybrid_code_chunks_mmaptest"

type mmapPolicyServer struct {
	milvuspb.UnimplementedMilvusServiceServer

	mutex            sync.Mutex
	exists           bool
	loaded           bool
	schema           *schemapb.CollectionSchema
	fieldProperties  map[string]map[string]string
	indexProperties  map[string]map[string]string
	indexTypes       map[string]string
	indexFields      map[string]string
	failAlterOnce    string
	ignoreAlterOnce  string
	ignoreAlter      string
	fieldAlterCalls  map[string]int
	awaitPending     map[string]bool
	delayedLists     int
	invisibleField   string
	hideAfterIndexes bool
	hiddenHasCalls   int
	createMmapParam  bool
	loadedTooEarly   bool
	events           []string
	blockMmapField   string
	inspectionReady  chan struct{}
	resumeInspect    chan struct{}
}

func newMmapPolicyServer() *mmapPolicyServer {
	return &mmapPolicyServer{
		exists: true,
		loaded: true,
		fieldProperties: map[string]map[string]string{
			"id":            {},
			"content":       {},
			"relativePath":  {},
			"contentHash":   {},
			"vector":        {},
			"sparse_vector": {},
		},
		indexProperties: map[string]map[string]string{
			"vector_idx":       {},
			"content_hash_idx": {},
			"sparse_idx":       {},
		},
		indexTypes: map[string]string{
			"vector_idx":       "AUTOINDEX",
			"content_hash_idx": "INVERTED",
			"sparse_idx":       "SPARSE_INVERTED_INDEX",
		},
		indexFields: map[string]string{
			"vector_idx":       "vector",
			"content_hash_idx": "contentHash",
			"sparse_idx":       "sparse_vector",
		},
		fieldAlterCalls: make(map[string]int),
		awaitPending:    make(map[string]bool),
	}
}

func (server *mmapPolicyServer) Connect(
	context.Context,
	*milvuspb.ConnectRequest,
) (*milvuspb.ConnectResponse, error) {
	return &milvuspb.ConnectResponse{Status: mmapSuccessStatus(), Identifier: 1}, nil
}

func (server *mmapPolicyServer) ShowCollections(
	context.Context,
	*milvuspb.ShowCollectionsRequest,
) (*milvuspb.ShowCollectionsResponse, error) {
	return &milvuspb.ShowCollectionsResponse{
		Status:          mmapSuccessStatus(),
		CollectionNames: []string{mmapTestCollection},
	}, nil
}

func (server *mmapPolicyServer) HasCollection(
	context.Context,
	*milvuspb.HasCollectionRequest,
) (*milvuspb.BoolResponse, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	if server.exists && server.hiddenHasCalls > 0 {
		server.hiddenHasCalls--
		return &milvuspb.BoolResponse{Status: mmapSuccessStatus(), Value: false}, nil
	}
	return &milvuspb.BoolResponse{Status: mmapSuccessStatus(), Value: server.exists}, nil
}

func (server *mmapPolicyServer) DescribeCollection(
	_ context.Context,
	request *milvuspb.DescribeCollectionRequest,
) (*milvuspb.DescribeCollectionResponse, error) {
	server.mutex.Lock()
	server.events = append(server.events, "describe_collection")
	if !server.exists {
		server.mutex.Unlock()
		return &milvuspb.DescribeCollectionResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_CollectionNotExists,
				Reason:    "collection not found",
			},
		}, nil
	}
	var schema *schemapb.CollectionSchema
	if server.schema != nil {
		schema = proto.Clone(server.schema).(*schemapb.CollectionSchema)
		for _, field := range schema.GetFields() {
			properties := server.fieldProperties[field.GetName()]
			for key, value := range properties {
				found := false
				for _, parameter := range field.TypeParams {
					if parameter.GetKey() == key {
						parameter.Value = value
						found = true
						break
					}
				}
				if !found {
					field.TypeParams = append(field.TypeParams, &commonpb.KeyValuePair{
						Key: key, Value: value,
					})
				}
			}
		}
	} else {
		entitySchema := entity.NewSchema().WithName(request.GetCollectionName())
		fields := []*entity.Field{
			entity.NewField().WithName("id").WithDataType(entity.FieldTypeVarChar).WithIsPrimaryKey(true),
			entity.NewField().WithName("content").WithDataType(entity.FieldTypeVarChar),
			entity.NewField().WithName("relativePath").WithDataType(entity.FieldTypeVarChar),
			entity.NewField().WithName("contentHash").WithDataType(entity.FieldTypeVarChar),
			entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(3),
			entity.NewField().WithName("sparse_vector").WithDataType(entity.FieldTypeSparseVector),
		}
		for _, field := range fields {
			for key, value := range server.fieldProperties[field.Name] {
				field.WithTypeParams(key, value)
			}
			entitySchema.WithField(field)
		}
		schema = entitySchema.ProtoMessage()
	}
	block := server.blockMmapField != "" &&
		server.fieldProperties[server.blockMmapField]["mmap.enabled"] == "true"
	inspectionReady := server.inspectionReady
	resumeInspect := server.resumeInspect
	if block {
		server.blockMmapField = ""
	}
	server.mutex.Unlock()
	if block {
		close(inspectionReady)
		<-resumeInspect
	}
	return &milvuspb.DescribeCollectionResponse{
		Status:         mmapSuccessStatus(),
		CollectionName: request.GetCollectionName(),
		Schema:         schema,
	}, nil
}

func (server *mmapPolicyServer) DescribeIndex(
	_ context.Context,
	request *milvuspb.DescribeIndexRequest,
) (*milvuspb.DescribeIndexResponse, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	server.events = append(server.events, "describe_indexes")
	if server.awaitPending[request.GetFieldName()] {
		server.awaitPending[request.GetFieldName()] = false
	} else if server.invisibleField != "" && request.GetFieldName() == server.invisibleField {
		return &milvuspb.DescribeIndexResponse{Status: mmapSuccessStatus()}, nil
	} else if server.delayedLists > 0 {
		server.delayedLists--
		return &milvuspb.DescribeIndexResponse{Status: mmapSuccessStatus()}, nil
	}
	descriptions := make([]*milvuspb.IndexDescription, 0, len(server.indexTypes))
	for indexName, indexType := range server.indexTypes {
		if request.GetIndexName() != "" && request.GetIndexName() != indexName {
			continue
		}
		if request.GetFieldName() != "" && request.GetFieldName() != server.indexFields[indexName] {
			continue
		}
		params := []*commonpb.KeyValuePair{{Key: "index_type", Value: indexType}}
		for key, value := range server.indexProperties[indexName] {
			params = append(params, &commonpb.KeyValuePair{Key: key, Value: value})
		}
		descriptions = append(descriptions, &milvuspb.IndexDescription{
			IndexName: indexName,
			FieldName: server.indexFields[indexName],
			Params:    params,
			State:     commonpb.IndexState_Finished,
		})
	}
	return &milvuspb.DescribeIndexResponse{
		Status:            mmapSuccessStatus(),
		IndexDescriptions: descriptions,
	}, nil
}

func (server *mmapPolicyServer) GetLoadState(
	context.Context,
	*milvuspb.GetLoadStateRequest,
) (*milvuspb.GetLoadStateResponse, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	state := commonpb.LoadState_LoadStateNotLoad
	event := "get_load_state:not_load"
	if server.loaded {
		state = commonpb.LoadState_LoadStateLoaded
		event = "get_load_state:loaded"
	}
	server.events = append(server.events, event)
	return &milvuspb.GetLoadStateResponse{Status: mmapSuccessStatus(), State: state}, nil
}

func (server *mmapPolicyServer) ReleaseCollection(
	context.Context,
	*milvuspb.ReleaseCollectionRequest,
) (*commonpb.Status, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	server.events = append(server.events, "release")
	server.loaded = false
	return mmapSuccessStatus(), nil
}

func (server *mmapPolicyServer) AlterCollectionField(
	_ context.Context,
	request *milvuspb.AlterCollectionFieldRequest,
) (*commonpb.Status, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	server.events = append(server.events, "alter_field:"+request.GetFieldName())
	server.fieldAlterCalls[request.GetFieldName()]++
	if server.failAlterOnce == request.GetFieldName() {
		server.failAlterOnce = ""
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    "injected one-time alteration failure",
		}, nil
	}
	if server.ignoreAlterOnce == request.GetFieldName() {
		server.ignoreAlterOnce = ""
		return mmapSuccessStatus(), nil
	}
	if server.ignoreAlter == request.GetFieldName() {
		return mmapSuccessStatus(), nil
	}
	if server.fieldProperties[request.GetFieldName()] == nil {
		server.fieldProperties[request.GetFieldName()] = make(map[string]string)
	}
	for _, property := range request.GetProperties() {
		server.fieldProperties[request.GetFieldName()][property.GetKey()] = property.GetValue()
	}
	return mmapSuccessStatus(), nil
}

func (server *mmapPolicyServer) AlterIndex(
	_ context.Context,
	request *milvuspb.AlterIndexRequest,
) (*commonpb.Status, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	server.events = append(server.events, "alter_index:"+request.GetIndexName())
	if server.indexProperties[request.GetIndexName()] == nil {
		server.indexProperties[request.GetIndexName()] = make(map[string]string)
	}
	for _, property := range request.GetExtraParams() {
		server.indexProperties[request.GetIndexName()][property.GetKey()] = property.GetValue()
	}
	return mmapSuccessStatus(), nil
}

func (server *mmapPolicyServer) LoadCollection(
	context.Context,
	*milvuspb.LoadCollectionRequest,
) (*commonpb.Status, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	server.events = append(server.events, "load")
	for fieldName, properties := range server.fieldProperties {
		if fieldName == "id" || fieldName == "sparse_vector" {
			continue
		}
		if properties["mmap.enabled"] != "true" {
			server.loadedTooEarly = true
		}
	}
	for _, properties := range server.indexProperties {
		if properties["mmap.enabled"] != "true" {
			server.loadedTooEarly = true
		}
	}
	server.loaded = true
	return mmapSuccessStatus(), nil
}

func (server *mmapPolicyServer) CreateCollection(
	_ context.Context,
	request *milvuspb.CreateCollectionRequest,
) (*commonpb.Status, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	server.events = append(server.events, "create_collection")
	schema := &schemapb.CollectionSchema{}
	if err := proto.Unmarshal(request.GetSchema(), schema); err != nil {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UnexpectedError,
			Reason:    err.Error(),
		}, nil
	}
	server.exists = true
	server.loaded = false
	server.schema = schema
	server.fieldProperties = make(map[string]map[string]string, len(schema.GetFields()))
	for _, field := range schema.GetFields() {
		server.fieldProperties[field.GetName()] = make(map[string]string)
		for _, property := range field.GetTypeParams() {
			server.fieldProperties[field.GetName()][property.GetKey()] = property.GetValue()
		}
	}
	server.indexProperties = make(map[string]map[string]string)
	server.indexTypes = make(map[string]string)
	server.indexFields = make(map[string]string)
	return mmapSuccessStatus(), nil
}

func (server *mmapPolicyServer) CreateIndex(
	_ context.Context,
	request *milvuspb.CreateIndexRequest,
) (*commonpb.Status, error) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	server.events = append(server.events, "create_index:"+request.GetFieldName())
	indexType := ""
	for _, property := range request.GetExtraParams() {
		if property.GetKey() == "mmap.enabled" {
			server.createMmapParam = true
		}
		if property.GetKey() == "index_type" {
			indexType = property.GetValue()
		}
	}
	indexName := request.GetIndexName()
	if indexName == "" {
		indexName = request.GetFieldName() + "_idx"
	}
	server.indexProperties[indexName] = make(map[string]string)
	server.indexTypes[indexName] = indexType
	server.indexFields[indexName] = request.GetFieldName()
	server.awaitPending[request.GetFieldName()] = true
	if server.hideAfterIndexes && request.GetFieldName() == "sparse_vector" {
		server.hiddenHasCalls = 1
	}
	return mmapSuccessStatus(), nil
}

func (server *mmapPolicyServer) Insert(
	context.Context,
	*milvuspb.InsertRequest,
) (*milvuspb.MutationResult, error) {
	server.mutex.Lock()
	server.events = append(server.events, "insert")
	server.mutex.Unlock()
	return &milvuspb.MutationResult{
		Status:       mmapSuccessStatus(),
		Acknowledged: true,
		InsertCnt:    1,
	}, nil
}

func (server *mmapPolicyServer) Flush(
	_ context.Context,
	request *milvuspb.FlushRequest,
) (*milvuspb.FlushResponse, error) {
	flushTimestamps := make(map[string]uint64, len(request.GetCollectionNames()))
	for _, collectionName := range request.GetCollectionNames() {
		flushTimestamps[collectionName] = 1
	}
	return &milvuspb.FlushResponse{
		Status:      mmapSuccessStatus(),
		CollFlushTs: flushTimestamps,
	}, nil
}

func (server *mmapPolicyServer) GetFlushState(
	context.Context,
	*milvuspb.GetFlushStateRequest,
) (*milvuspb.GetFlushStateResponse, error) {
	return &milvuspb.GetFlushStateResponse{Status: mmapSuccessStatus(), Flushed: true}, nil
}

func (server *mmapPolicyServer) propertySnapshot() (map[string]string, map[string]string, bool) {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	fields := make(map[string]string, len(server.fieldProperties))
	for name, properties := range server.fieldProperties {
		fields[name] = properties["mmap.enabled"]
	}
	indexes := make(map[string]string, len(server.indexProperties))
	for name, properties := range server.indexProperties {
		indexes[name] = properties["mmap.enabled"]
	}
	return fields, indexes, server.loaded
}

func (server *mmapPolicyServer) eventSnapshot() []string {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	return append([]string(nil), server.events...)
}

func (server *mmapPolicyServer) fieldAlterSnapshot() map[string]int {
	server.mutex.Lock()
	defer server.mutex.Unlock()
	return map[string]int{
		"content":      server.fieldAlterCalls["content"],
		"relativePath": server.fieldAlterCalls["relativePath"],
		"contentHash":  server.fieldAlterCalls["contentHash"],
		"vector":       server.fieldAlterCalls["vector"],
	}
}

func TestMmapPolicyMigratesEverySupportedPresentObjectAndRestoresReady(t *testing.T) {
	t.Parallel()

	server, service := newMmapPolicyTestService(t)
	service.EnsureMmapEnabledAllCollections(context.Background())

	fields, indexes, loaded := server.propertySnapshot()
	for _, fieldName := range []string{"content", "relativePath", "contentHash", "vector"} {
		if fields[fieldName] != "true" {
			t.Errorf("field %s mmap.enabled = %q, want true", fieldName, fields[fieldName])
		}
	}
	for _, fieldName := range []string{"id", "sparse_vector"} {
		if fields[fieldName] != "" {
			t.Errorf("field %s mmap.enabled = %q, want absent", fieldName, fields[fieldName])
		}
	}
	for _, indexName := range []string{"vector_idx", "content_hash_idx", "sparse_idx"} {
		if indexes[indexName] != "true" {
			t.Errorf("index %s mmap.enabled = %q, want true", indexName, indexes[indexName])
		}
	}
	if !loaded {
		t.Fatal("ready collection remained idle after migration")
	}
	assertMmapMigrationOrder(t, server.eventSnapshot())
}

func TestMmapPolicyHandlesMissingCollectionSchema(t *testing.T) {
	t.Parallel()

	server, service := newMmapPolicyTestService(t)
	server.mutex.Lock()
	server.schema = nil
	server.mutex.Unlock()

	service.EnsureMmapEnabledAllCollections(context.Background())
}

func TestMmapPolicyFakeInitializesPropertyMaps(t *testing.T) {
	t.Parallel()

	server := newMmapPolicyServer()
	delete(server.fieldProperties, "content")
	delete(server.indexProperties, "vector_idx")

	if _, err := server.AlterCollectionField(context.Background(), &milvuspb.AlterCollectionFieldRequest{
		FieldName: "content",
		Properties: []*commonpb.KeyValuePair{{
			Key:   "mmap.enabled",
			Value: "true",
		}},
	}); err != nil {
		t.Fatalf("AlterCollectionField returned error: %v", err)
	}
	if _, err := server.AlterIndex(context.Background(), &milvuspb.AlterIndexRequest{
		IndexName: "vector_idx",
		ExtraParams: []*commonpb.KeyValuePair{{
			Key:   "mmap.enabled",
			Value: "true",
		}},
	}); err != nil {
		t.Fatalf("AlterIndex returned error: %v", err)
	}
}

func TestMmapPolicyFakePreservesSchemaTypeParams(t *testing.T) {
	t.Parallel()

	server := newMmapPolicyServer()
	response, err := server.DescribeCollection(
		context.Background(),
		&milvuspb.DescribeCollectionRequest{CollectionName: mmapTestCollection},
	)
	if err != nil {
		t.Fatalf("DescribeCollection returned error: %v", err)
	}
	for _, field := range response.GetSchema().GetFields() {
		if field.GetName() != "vector" {
			continue
		}
		for _, parameter := range field.GetTypeParams() {
			if parameter.GetKey() == "dim" && parameter.GetValue() == "3" {
				return
			}
		}
		t.Fatalf("vector type parameters = %v, want dim=3", field.GetTypeParams())
	}
	t.Fatal("DescribeCollection omitted vector field")
}

func TestMmapPolicyPreservesIdleCollection(t *testing.T) {
	t.Parallel()

	server, service := newMmapPolicyTestService(t)
	server.mutex.Lock()
	server.loaded = false
	server.mutex.Unlock()
	service.EnsureMmapEnabledAllCollections(context.Background())

	_, _, loaded := server.propertySnapshot()
	if loaded {
		t.Fatal("idle collection was loaded by mmap migration")
	}
	for _, event := range server.eventSnapshot() {
		if event == "release" || event == "load" {
			t.Fatalf("idle migration issued residency transition %q", event)
		}
	}
}

func TestMmapPolicyRetriesOnlyMissingPropertiesAfterAlterFailure(t *testing.T) {
	t.Parallel()

	server, service := newMmapPolicyTestService(t)
	server.mutex.Lock()
	server.failAlterOnce = "contentHash"
	server.mutex.Unlock()

	service.EnsureMmapEnabledAllCollections(context.Background())
	_, _, loaded := server.propertySnapshot()
	if !loaded {
		t.Fatal("failed migration did not restore the prior ready state")
	}

	service.EnsureMmapEnabledAllCollections(context.Background())
	fields, indexes, loaded := server.propertySnapshot()
	for _, fieldName := range []string{"content", "relativePath", "contentHash", "vector"} {
		if fields[fieldName] != "true" {
			t.Errorf("field %s mmap.enabled = %q after retry, want true", fieldName, fields[fieldName])
		}
	}
	for _, indexName := range []string{"vector_idx", "content_hash_idx", "sparse_idx"} {
		if indexes[indexName] != "true" {
			t.Errorf("index %s mmap.enabled = %q after retry, want true", indexName, indexes[indexName])
		}
	}
	if !loaded {
		t.Fatal("retried migration did not preserve ready state")
	}
	calls := server.fieldAlterSnapshot()
	if calls["content"] != 1 || calls["relativePath"] != 1 {
		t.Fatalf("already persisted scalar fields were retried: %v", calls)
	}
	if calls["contentHash"] != 2 || calls["vector"] != 1 {
		t.Fatalf("missing scalar retry calls = %v, want contentHash=2 and vector=1", calls)
	}
}

func TestMmapPolicyRechecksEveryTargetBeforeStampingComplete(t *testing.T) {
	t.Parallel()

	server, service := newMmapPolicyTestService(t)
	server.mutex.Lock()
	server.ignoreAlterOnce = "contentHash"
	server.mutex.Unlock()

	service.EnsureMmapEnabledAllCollections(context.Background())
	service.EnsureMmapEnabledAllCollections(context.Background())

	fields, _, loaded := server.propertySnapshot()
	if fields["contentHash"] != "true" {
		t.Fatalf("contentHash mmap.enabled = %q after recheck retry, want true", fields["contentHash"])
	}
	if !loaded {
		t.Fatal("full-reverification retry did not restore ready state")
	}
	if calls := server.fieldAlterSnapshot()["contentHash"]; calls != 2 {
		t.Fatalf("contentHash alteration calls = %d, want 2 after ignored first change", calls)
	}
}

func TestMmapPolicyBacksOffWhenAPropertyNeverPersists(t *testing.T) {
	t.Parallel()

	server, service := newMmapPolicyTestService(t)
	server.mutex.Lock()
	server.ignoreAlter = "contentHash"
	server.mutex.Unlock()

	service.EnsureMmapEnabledAllCollections(context.Background())
	service.EnsureMmapEnabledAllCollections(context.Background())
	releasesAfterThreshold := countMmapEvents(server.eventSnapshot(), "release")
	service.EnsureMmapEnabledAllCollections(context.Background())
	if releases := countMmapEvents(server.eventSnapshot(), "release"); releases != releasesAfterThreshold {
		t.Fatalf("release calls during mmap backoff = %d, want %d", releases, releasesAfterThreshold)
	}

	server.mutex.Lock()
	server.ignoreAlter = ""
	server.mutex.Unlock()
	for range 11 {
		service.EnsureMmapEnabledAllCollections(context.Background())
	}
	if releases := countMmapEvents(server.eventSnapshot(), "release"); releases != releasesAfterThreshold {
		t.Fatalf("release calls before mmap backoff expired = %d, want %d", releases, releasesAfterThreshold)
	}
	service.EnsureMmapEnabledAllCollections(context.Background())
	fields, _, _ := server.propertySnapshot()
	if fields["contentHash"] != "true" {
		t.Fatalf("contentHash mmap.enabled = %q after retry, want true", fields["contentHash"])
	}
}

func TestMmapPolicyStampInvalidatesAfterConfirmedAbsence(t *testing.T) {
	t.Parallel()

	server, service := newMmapPolicyTestService(t)
	service.EnsureMmapEnabledAllCollections(context.Background())

	server.mutex.Lock()
	server.exists = false
	server.mutex.Unlock()
	facts, err := service.InspectCollection(context.Background(), mmapTestCollection)
	if err != nil {
		t.Fatalf("InspectCollection returned error: %v", err)
	}
	if facts.Exists {
		t.Fatal("InspectCollection reported the absent collection as present")
	}

	server.mutex.Lock()
	server.exists = true
	server.loaded = false
	server.schema = mmapPolicySchemaWithExtraScalar()
	server.fieldProperties["newScalar"] = make(map[string]string)
	server.mutex.Unlock()
	service.EnsureMmapEnabledAllCollections(context.Background())

	server.mutex.Lock()
	newScalarMmap := server.fieldProperties["newScalar"]["mmap.enabled"]
	server.mutex.Unlock()
	if newScalarMmap != "true" {
		t.Fatalf("recreated scalar mmap.enabled = %q, want true", newScalarMmap)
	}
}

func TestMmapPolicyDoesNotRestampAfterConcurrentInvalidation(t *testing.T) {
	t.Parallel()

	server, service := newMmapPolicyTestService(t)
	service.EnsureMmapEnabledAllCollections(context.Background())

	server.mutex.Lock()
	server.exists = false
	server.mutex.Unlock()
	if _, err := service.InspectCollection(context.Background(), mmapTestCollection); err != nil {
		t.Fatalf("invalidate initial mmap stamp: %v", err)
	}

	inspectionReady := make(chan struct{})
	resumeInspect := make(chan struct{})
	server.mutex.Lock()
	server.exists = true
	server.schema = mmapPolicySchemaWithExtraScalar()
	server.fieldProperties["newScalar"] = make(map[string]string)
	server.blockMmapField = "newScalar"
	server.inspectionReady = inspectionReady
	server.resumeInspect = resumeInspect
	server.mutex.Unlock()

	migrationDone := make(chan struct{})
	go func() {
		service.EnsureMmapEnabledAllCollections(context.Background())
		close(migrationDone)
	}()
	select {
	case <-inspectionReady:
	case <-time.After(5 * time.Second):
		t.Fatal("mmap verification did not reach deterministic race point")
	}

	server.mutex.Lock()
	server.exists = false
	server.mutex.Unlock()
	if _, err := service.InspectCollection(context.Background(), mmapTestCollection); err != nil {
		t.Fatalf("invalidate mmap stamp during verification: %v", err)
	}
	server.mutex.Lock()
	server.exists = true
	server.schema.Fields = append(server.schema.Fields, &schemapb.FieldSchema{
		Name:     "concurrentScalar",
		DataType: schemapb.DataType_Int64,
	})
	server.fieldProperties["concurrentScalar"] = make(map[string]string)
	server.mutex.Unlock()
	close(resumeInspect)
	select {
	case <-migrationDone:
	case <-time.After(5 * time.Second):
		t.Fatal("mmap migration did not finish after verification resumed")
	}

	service.EnsureMmapEnabledAllCollections(context.Background())
	fields, _, _ := server.propertySnapshot()
	if fields["concurrentScalar"] != "true" {
		t.Fatalf(
			"concurrent scalar mmap.enabled = %q, want true after invalidation retry",
			fields["concurrentScalar"],
		)
	}
}

func TestCreatedCollectionWaitsForIndexesAndLoadsAfterPolicy(t *testing.T) {
	t.Parallel()

	server, service := newMmapCreateTestService(t)
	server.mutex.Lock()
	server.delayedLists = 2
	server.mutex.Unlock()

	err := service.StageReindex(
		context.Background(),
		t.TempDir(),
		[]model.StoredChunk{{
			Content:       "package created",
			RelativePath:  "created.go",
			StartLine:     1,
			EndLine:       1,
			FileExtension: ".go",
		}},
		semantic.Removal{},
		nil,
		map[string][]float32{
			semantic.ContentVectorKey("package created"): {1, 0, 0},
		},
		semantic.StoreColumnSetCode,
	)
	if err != nil {
		t.Fatalf("StageReindex returned error: %v", err)
	}

	server.mutex.Lock()
	createMmapParam := server.createMmapParam
	loadedTooEarly := server.loadedTooEarly
	loaded := server.loaded
	server.mutex.Unlock()
	if createMmapParam {
		t.Fatal("AUTOINDEX creation included mmap.enabled")
	}
	if loadedTooEarly {
		t.Fatal("fresh collection loaded before every mmap property was present")
	}
	if !loaded {
		t.Fatal("fresh collection did not load after mmap policy completed")
	}
}

func TestCreatedCollectionRetriesBriefPostCreateInvisibility(t *testing.T) {
	t.Parallel()

	server, service := newMmapCreateTestService(t)
	server.mutex.Lock()
	server.hideAfterIndexes = true
	server.mutex.Unlock()

	err := stageMmapTestChunk(context.Background(), service, t.TempDir(), "briefly invisible")
	if err != nil {
		t.Fatalf("StageReindex returned error: %v", err)
	}
	server.mutex.Lock()
	loadedTooEarly := server.loadedTooEarly
	loaded := server.loaded
	server.mutex.Unlock()
	if countMmapEvents(server.eventSnapshot(), "create_index:sparse_vector") == 0 {
		t.Fatal("post-create invisibility trigger index was not created")
	}
	if loadedTooEarly {
		t.Fatal("fresh collection loaded after skipped mmap migration")
	}
	if !loaded {
		t.Fatal("fresh collection did not load after visibility recovered")
	}
}

func TestCreatedCollectionFailsWithoutLoadWhenRequiredIndexStaysInvisible(t *testing.T) {
	t.Parallel()

	server, service := newMmapCreateTestService(t)
	server.mutex.Lock()
	server.invisibleField = "contentHash"
	server.mutex.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	err := service.StageReindex(
		ctx,
		t.TempDir(),
		[]model.StoredChunk{{
			Content:       "package invisible",
			RelativePath:  "invisible.go",
			StartLine:     1,
			EndLine:       1,
			FileExtension: ".go",
		}},
		semantic.Removal{},
		nil,
		map[string][]float32{
			semantic.ContentVectorKey("package invisible"): {1, 0, 0},
		},
		semantic.StoreColumnSetCode,
	)
	if err == nil {
		t.Fatal("StageReindex returned nil while contentHash index stayed invisible")
	}
	if !strings.Contains(err.Error(), "wait for required mmap indexes") {
		t.Fatalf("StageReindex error = %v, want required mmap index wait failure", err)
	}
	server.mutex.Lock()
	loaded := server.loaded
	events := append([]string(nil), server.events...)
	server.mutex.Unlock()
	if loaded {
		t.Fatal("failed create-mode migration loaded the collection")
	}
	for _, event := range events {
		if event == "load" {
			t.Fatalf("failed create-mode migration issued load: %v", events)
		}
	}
}

func TestCreatedCollectionRetryResumesMmapWithoutCreatingAgain(t *testing.T) {
	t.Parallel()

	server, service := newMmapCreateTestService(t)
	server.mutex.Lock()
	server.invisibleField = "contentHash"
	server.mutex.Unlock()

	firstCtx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	codebasePath := t.TempDir()
	if err := stageMmapTestChunk(firstCtx, service, codebasePath, "first attempt"); err == nil {
		t.Fatal("first StageReindex returned nil while contentHash index stayed invisible")
	} else if !strings.Contains(err.Error(), "wait for required mmap indexes") {
		t.Fatalf("first StageReindex error = %v, want required mmap index wait failure", err)
	}
	server.mutex.Lock()
	server.invisibleField = ""
	server.mutex.Unlock()
	if err := stageMmapTestChunk(context.Background(), service, codebasePath, "retry attempt"); err != nil {
		t.Fatalf("retry StageReindex returned error: %v", err)
	}

	events := server.eventSnapshot()
	if creates := countMmapEvents(events, "create_collection"); creates != 1 {
		t.Fatalf("create collection calls = %d, want 1 across retry", creates)
	}
	fields, indexes, loaded := server.propertySnapshot()
	if fields["contentHash"] != "true" || indexes["contentHash_idx"] != "true" {
		t.Fatalf("retry left mmap incomplete: fields=%v indexes=%v", fields, indexes)
	}
	if !loaded {
		t.Fatal("retry did not load the repaired collection")
	}
}

func TestCreatedCollectionContinuesWhenMmapPropertyDoesNotPersist(t *testing.T) {
	t.Parallel()

	server, service := newMmapCreateTestService(t)
	server.mutex.Lock()
	server.ignoreAlter = "contentHash"
	server.mutex.Unlock()

	if err := stageMmapTestChunk(
		context.Background(),
		service,
		t.TempDir(),
		"unsupported mmap property",
	); err != nil {
		t.Fatalf("StageReindex returned error: %v", err)
	}
	events := server.eventSnapshot()
	if countMmapEvents(events, "insert") == 0 {
		t.Fatalf("StageReindex did not write after mmap nonconvergence: %v", events)
	}
}

func stageMmapTestChunk(
	ctx context.Context,
	service *semantic.Service,
	codebasePath string,
	content string,
) error {
	return service.StageReindex(
		ctx,
		codebasePath,
		[]model.StoredChunk{{
			Content:       content,
			RelativePath:  "created.go",
			StartLine:     1,
			EndLine:       1,
			FileExtension: ".go",
		}},
		semantic.Removal{},
		nil,
		map[string][]float32{
			semantic.ContentVectorKey(content): {1, 0, 0},
		},
		semantic.StoreColumnSetCode,
	)
}

func countMmapEvents(events []string, expected string) int {
	count := 0
	for _, event := range events {
		if event == expected {
			count++
		}
	}
	return count
}

func assertMmapMigrationOrder(t *testing.T, events []string) {
	t.Helper()
	releaseIndex := -1
	notLoadIndex := -1
	firstAlterIndex := -1
	loadIndex := -1
	lastAlterIndex := -1
	recheckIndex := -1
	for index, event := range events {
		switch {
		case event == "release":
			releaseIndex = index
		case len(event) >= len("alter_") && event[:len("alter_")] == "alter_":
			lastAlterIndex = index
		}
	}
	for index, event := range events {
		switch {
		case event == "get_load_state:not_load" && index > releaseIndex && notLoadIndex < 0:
			notLoadIndex = index
		case len(event) >= len("alter_") && event[:len("alter_")] == "alter_" &&
			index > releaseIndex && firstAlterIndex < 0:
			firstAlterIndex = index
		case event == "describe_collection" && index > lastAlterIndex && recheckIndex < 0:
			recheckIndex = index
		case event == "load" && index > recheckIndex && recheckIndex >= 0 && loadIndex < 0:
			loadIndex = index
		}
	}
	if releaseIndex < 0 || notLoadIndex < releaseIndex || firstAlterIndex < notLoadIndex ||
		recheckIndex < lastAlterIndex ||
		loadIndex < recheckIndex {
		t.Fatalf("migration call order = %v", events)
	}
}

func newMmapPolicyTestService(t *testing.T) (*mmapPolicyServer, *semantic.Service) {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen for fake Milvus: %v", err)
	}
	server := newMmapPolicyServer()
	grpcServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(grpcServer, server)
	t.Cleanup(grpcServer.Stop)
	go func() {
		if serveErr := grpcServer.Serve(listener); serveErr != nil {
			t.Errorf("serve fake Milvus: %v", serveErr)
		}
	}()

	service, err := semantic.NewService(context.Background(), config.Config{
		EmbeddingProvider: "OpenAI",
		EmbeddingModel:    "test-model",
		OpenAIAPIKey:      "test-key",
		OpenAIBaseURL:     "http://127.0.0.1:1/v1",
		MilvusAddress:     listener.Addr().String(),
		HybridMode:        true,
	})
	if err != nil {
		t.Fatalf("NewService returned error: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := service.Close(context.Background()); closeErr != nil {
			t.Errorf("Close returned error: %v", closeErr)
		}
	})
	return server, service
}

func newMmapCreateTestService(t *testing.T) (*mmapPolicyServer, *semantic.Service) {
	t.Helper()
	embeddingServer := httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, _ *http.Request) {
			writer.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(writer).Encode(map[string]any{
				"data": []map[string]any{{"embedding": []float64{1, 0, 0}}},
			}); err != nil {
				t.Errorf("encode embedding response: %v", err)
			}
		},
	))
	t.Cleanup(embeddingServer.Close)

	server := newMmapPolicyServer()
	server.exists = false
	server.loaded = false
	server.fieldProperties = make(map[string]map[string]string)
	server.indexProperties = make(map[string]map[string]string)
	server.indexTypes = make(map[string]string)
	server.indexFields = make(map[string]string)

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen for fake Milvus: %v", err)
	}
	grpcServer := grpc.NewServer()
	milvuspb.RegisterMilvusServiceServer(grpcServer, server)
	t.Cleanup(grpcServer.Stop)
	go func() {
		if serveErr := grpcServer.Serve(listener); serveErr != nil {
			t.Errorf("serve fake Milvus: %v", serveErr)
		}
	}()

	service, err := semantic.NewService(context.Background(), config.Config{
		EmbeddingProvider:  "OpenAI",
		EmbeddingModel:     "test-model",
		EmbeddingDimension: 3,
		OpenAIAPIKey:       "test-key",
		OpenAIBaseURL:      embeddingServer.URL,
		MilvusAddress:      listener.Addr().String(),
		HybridMode:         true,
	})
	if err != nil {
		t.Fatalf("NewService returned error: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := service.Close(context.Background()); closeErr != nil {
			t.Errorf("Close returned error: %v", closeErr)
		}
	})
	return server, service
}

func mmapSuccessStatus() *commonpb.Status {
	return &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}
}

func mmapPolicySchemaWithExtraScalar() *schemapb.CollectionSchema {
	return entity.NewSchema().
		WithName(mmapTestCollection).
		WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeVarChar).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName("content").WithDataType(entity.FieldTypeVarChar)).
		WithField(entity.NewField().WithName("relativePath").WithDataType(entity.FieldTypeVarChar)).
		WithField(entity.NewField().WithName("contentHash").WithDataType(entity.FieldTypeVarChar)).
		WithField(entity.NewField().WithName("newScalar").WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName("vector").WithDataType(entity.FieldTypeFloatVector).WithDim(3)).
		WithField(entity.NewField().WithName("sparse_vector").WithDataType(entity.FieldTypeSparseVector)).
		ProtoMessage()
}
