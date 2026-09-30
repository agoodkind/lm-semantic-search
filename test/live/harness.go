//go:build live

// Package live holds the build-tagged, end-to-end validation of the merged
// conversation-marker feature against a real Milvus.
//
// Every run boots the daemon gRPC server in-process on a throwaway unix socket,
// points embedding at a local fake, and connects every Milvus client to a unique
// per-test database. Teardown drops every tracked collection and that database.
//
// Paths come from internal/sandbox, the same isolation the sandbox command
// gives a daemon run by hand. The store and the embedder are named here instead,
// because validating against the real store is the point.
//
// Run with:
//
//	go test -tags live -count=1 ./test/live/
//
// or `make live`. Residency tests fail when Milvus is unavailable because a skip
// cannot satisfy their acceptance gate.
package live

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/sandbox"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/semantic/milvusgrpc"
	"goodkind.io/lm-semantic-search/internal/tshash"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
)

const (
	defaultMilvusDatabase = "default"
	liveDatabasePrefix    = "lms_live_"

	// productionConversationCollection is the operator's real conversation
	// collection. The harness asserts every throwaway collection differs from it,
	// so a live run can never read, write, or drop production conversation rows.
	productionConversationCollection = "conv_chunks_09cfca5e"

	// fakeEmbeddingDimension is the width of every vector the fake embedder
	// returns. It defines the throwaway collection's dimension, learned lazily on
	// first insert, so a small fixed width keeps the collection cheap.
	fakeEmbeddingDimension = 16

	// relativePathField mirrors the collection's scalar column name
	// (internal/semantic), so the scenario-4 direct query can count rows by
	// relative-path prefix without importing unexported constants.
	relativePathField = "relativePath"
	countOutputField  = "count(*)"

	jobPollTimeout  = 90 * time.Second
	jobPollInterval = 100 * time.Millisecond
)

// harness owns one live test's isolated daemon, its throwaway collection, the
// gRPC client that drives ingest, and a direct Milvus client for row-level
// assertions and teardown. Every field is scoped to this test; nothing is shared
// with the operator's running daemon.
type harness struct {
	t                 *testing.T
	config            config.Config
	manager           *daemon.Manager
	conn              *grpc.ClientConn
	client            pb.SemanticSearchDaemonServiceClient
	operatorMilvus    *milvusclient.Client
	milvus            *milvusclient.Client
	databaseName      string
	collectionID      string
	collectionName    string
	reuseCatalogName  string
	codebaseID        string
	stateRoot         string
	merkleDir         string
	embedGate         *embedGate
	beforeDatabases   []string
	operatorBefore    milvusInventory
	sandboxBefore     milvusInventory
	temporaryNames    map[string]struct{}
	callRecorder      *milvusCallRecorder
	embeddingRecorder *embeddingCallRecorder
	milvusContext     func() context.Context
	stopServer        func()
}

type milvusInventory map[string]map[string]string

type operatorStateAudit struct {
	violations          []string
	concurrentAdditions []string
	// concurrentDatabases lists the databases outside the harness database name
	// that appeared in or disappeared from the database list during the test.
	concurrentDatabases []string
}

type milvusCall struct {
	databaseName            string
	destinationDatabaseName string
	method                  string
	collectionNames         []string
	recordedAt              time.Time
	caller                  string
}

type milvusCallRecorder struct {
	mutex sync.Mutex
	calls []milvusCall
}

type embeddingCallRecorder struct {
	mutex sync.Mutex
	calls [][]string
}

func (recorder *embeddingCallRecorder) record(inputs []string) {
	recorder.mutex.Lock()
	recorder.calls = append(recorder.calls, slices.Clone(inputs))
	recorder.mutex.Unlock()
}

func (recorder *embeddingCallRecorder) snapshot() [][]string {
	recorder.mutex.Lock()
	defer recorder.mutex.Unlock()
	calls := make([][]string, len(recorder.calls))
	for index, inputs := range recorder.calls {
		calls[index] = slices.Clone(inputs)
	}
	return calls
}

func (recorder *milvusCallRecorder) observe(
	method string,
	databaseName string,
	request proto.Message,
) {
	collectionNames := make([]string, 0, 2)
	destinationDatabaseName := ""
	if named, ok := request.(interface{ GetCollectionName() string }); ok {
		collectionNames = appendNonEmpty(collectionNames, named.GetCollectionName())
	}
	if named, ok := request.(interface{ GetCollectionNames() []string }); ok {
		for _, collectionName := range named.GetCollectionNames() {
			collectionNames = appendNonEmpty(collectionNames, collectionName)
		}
	}
	if renamed, ok := request.(interface {
		GetOldName() string
		GetNewName() string
		GetNewDBName() string
	}); ok {
		collectionNames = appendNonEmpty(collectionNames, renamed.GetOldName())
		collectionNames = appendNonEmpty(collectionNames, renamed.GetNewName())
		destinationDatabaseName = renamed.GetNewDBName()
	}
	if separator := strings.LastIndex(method, "/"); separator >= 0 {
		method = method[separator+1:]
	}
	recorder.mutex.Lock()
	recorder.calls = append(recorder.calls, milvusCall{
		databaseName:            databaseName,
		destinationDatabaseName: destinationDatabaseName,
		method:                  method,
		collectionNames:         slices.Clone(collectionNames),
		recordedAt:              clock.Now(),
		caller:                  milvusCallContext(),
	})
	recorder.mutex.Unlock()
}

func milvusCallContext() string {
	programCounters := make([]uintptr, 64)
	count := runtime.Callers(2, programCounters)
	frames := runtime.CallersFrames(programCounters[:count])
	for {
		frame, more := frames.Next()
		isRepositoryFrame := strings.HasPrefix(
			frame.Function,
			"goodkind.io/lm-semantic-search/",
		)
		isRecorderFrame := strings.Contains(frame.Function, "milvusCallRecorder")
		isTransportFrame := strings.Contains(frame.Function, "/internal/semantic/milvusgrpc.")
		if isRepositoryFrame && !isRecorderFrame && !isTransportFrame {
			return fmt.Sprintf("%s:%d", frame.Function, frame.Line)
		}
		if !more {
			return ""
		}
	}
}

func appendNonEmpty(values []string, value string) []string {
	if value == "" || slices.Contains(values, value) {
		return values
	}
	return append(values, value)
}

func (recorder *milvusCallRecorder) snapshot() []milvusCall {
	recorder.mutex.Lock()
	defer recorder.mutex.Unlock()
	result := make([]milvusCall, len(recorder.calls))
	for index, call := range recorder.calls {
		result[index] = milvusCall{
			databaseName:            call.databaseName,
			destinationDatabaseName: call.destinationDatabaseName,
			method:                  call.method,
			collectionNames:         slices.Clone(call.collectionNames),
			recordedAt:              call.recordedAt,
			caller:                  call.caller,
		}
	}
	return result
}

func (recorder *milvusCallRecorder) reset() {
	recorder.mutex.Lock()
	recorder.calls = nil
	recorder.mutex.Unlock()
}

func (recorder *milvusCallRecorder) count(method string, collectionName string) int {
	count := 0
	for _, call := range recorder.snapshot() {
		if call.method == method && slices.Contains(call.collectionNames, collectionName) {
			count++
		}
	}
	return count
}

// newHarness builds the isolated daemon and returns a ready harness, or skips the
// test when Milvus is unreachable (a BLOCKED environment condition, not a code
// failure). It registers a per-test UUID conversation collection and asserts the
// derived Milvus name is not the production collection before any ingest runs.
func newHarness(t *testing.T) *harness {
	t.Helper()
	return newHarnessWithGate(t, nil)
}

func newResidencyHarness(t *testing.T, idleTimeout time.Duration) *harness {
	t.Helper()
	return newHarnessWithOptions(t, nil, idleTimeout, true)
}

// newHarnessWithGate builds the isolated daemon like newHarness but installs an
// embedGate so a test can pace embedding requests and read the job's progress
// between batches. A nil gate is the normal, ungated path.
func newHarnessWithGate(t *testing.T, gate *embedGate) *harness {
	t.Helper()
	return newHarnessWithOptions(t, gate, 0, false)
}

func newHarnessWithOptions(t *testing.T, gate *embedGate, idleTimeout time.Duration, requireMilvus bool) *harness {
	t.Helper()

	defaultConfig := resolveHarnessConfig(t, requireMilvus)
	milvusAddress := strings.TrimSpace(defaultConfig.MilvusAddress)
	harnessID := randomID()
	databaseName := liveDatabasePrefix + harnessID
	callRecorder := &milvusCallRecorder{}
	operatorContext := context.WithValue(
		context.Background(),
		milvusgrpc.CallObserverContextKey{},
		milvusgrpc.CallObserver(callRecorder.observe),
	)
	operatorMilvus := connectLiveOperator(t, operatorContext, milvusAddress, defaultConfig.MilvusToken, requireMilvus)
	var (
		databaseCreated bool
		manager         *daemon.Manager
		conn            *grpc.ClientConn
		sandboxMilvus   *milvusclient.Client
		setupComplete   bool
		stopServer      func()
	)
	t.Cleanup(func() {
		if setupComplete {
			return
		}
		cleanupPartialHarness(
			t,
			conn,
			stopServer,
			manager,
			sandboxMilvus,
			operatorMilvus,
			databaseName,
			databaseCreated,
		)
	})

	operatorBefore, err := readMilvusInventory(operatorMilvus)
	if err != nil {
		t.Fatalf("read operator Milvus inventory before: %v", err)
	}
	beforeDatabases, err := listMilvusDatabases(operatorMilvus)
	if err != nil {
		t.Fatalf("list Milvus databases before: %v", err)
	}
	t.Logf("Milvus operator inventory before: %v", operatorBefore)
	t.Logf("Milvus databases before: %v", beforeDatabases)
	if slices.Contains(beforeDatabases, databaseName) {
		t.Fatalf("temporary Milvus database %q already exists; refusing collision", databaseName)
	}
	createCtx, createCancel := context.WithTimeout(operatorContext, 15*time.Second)
	if err := operatorMilvus.CreateDatabase(
		createCtx,
		milvusclient.NewCreateDatabaseOption(databaseName),
	); err != nil {
		createCancel()
		t.Fatalf("CreateDatabase(%s) returned error: %v", databaseName, err)
	}
	createCancel()
	databaseCreated = true

	sandboxContext, sandboxClient, sandboxBefore := connectLiveSandbox(t, milvusAddress, defaultConfig.MilvusToken, databaseName, callRecorder)
	sandboxMilvus = sandboxClient

	cfg, stateRoot, embeddingRecorder := prepareLiveDaemonConfig(t, gate, milvusAddress, defaultConfig.MilvusToken, databaseName, harnessID, idleTimeout)

	collectionID := "live-marker-" + harnessID
	started := startLiveHarnessDaemon(t, sandboxContext, cfg, collectionID)
	manager, conn, stopServer = started.manager, started.conn, started.stopServer

	h := &harness{
		t:                 t,
		config:            cfg,
		manager:           manager,
		conn:              conn,
		client:            started.client,
		operatorMilvus:    operatorMilvus,
		milvus:            sandboxMilvus,
		databaseName:      databaseName,
		collectionID:      collectionID,
		collectionName:    started.collectionName,
		reuseCatalogName:  semantic.ReuseCatalogCollectionName(cfg),
		codebaseID:        started.codebaseID,
		stateRoot:         stateRoot,
		merkleDir:         cfg.MerkleDir,
		embedGate:         gate,
		beforeDatabases:   beforeDatabases,
		operatorBefore:    operatorBefore,
		sandboxBefore:     sandboxBefore,
		temporaryNames:    make(map[string]struct{}),
		callRecorder:      callRecorder,
		embeddingRecorder: embeddingRecorder,
		milvusContext:     func() context.Context { return sandboxContext },
		stopServer:        stopServer,
	}
	h.trackCollectionFamily(started.collectionName)
	h.trackTemporaryCollection(h.reuseCatalogName)
	t.Cleanup(func() { h.teardown(h.stopServer) })
	setupComplete = true
	return h
}

// restart stops the in-process daemon, runs between while no daemon owns the
// state root, and starts a new daemon over the same state root, temporary
// Milvus database, and socket.
func (h *harness) restart(between func()) {
	h.t.Helper()
	if err := h.conn.Close(); err != nil {
		h.t.Fatalf("close gRPC connection before restart returned error: %v", err)
	}
	h.stopServer()
	closeCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	closeErr := h.manager.Close(closeCtx)
	cancel()
	if closeErr != nil {
		h.t.Fatalf("close manager before restart returned error: %v", closeErr)
	}
	if between != nil {
		between()
	}
	manager, err := daemon.NewManager(h.milvusContext(), h.config)
	if err != nil {
		h.t.Fatalf("NewManager on restart returned error: %v", err)
	}
	h.manager = manager
	h.stopServer = startInProcessServer(h.t, h.milvusContext(), manager, h.config.SocketPath)
	conn, client, err := grpcutil.DialDaemon(context.Background(), h.config.SocketPath)
	if err != nil {
		h.t.Fatalf("DialDaemon on restart returned error: %v", err)
	}
	h.conn = conn
	h.client = client
}

func (h *harness) childConfig() config.Config {
	h.t.Helper()
	childConfig, err := config.Default()
	if err != nil {
		h.t.Fatalf("load child live config: %v", err)
	}
	if childConfig.MilvusDatabase != h.databaseName {
		h.t.Fatalf(
			"child MilvusDatabase = %q, want temporary database %q",
			childConfig.MilvusDatabase,
			h.databaseName,
		)
	}
	return childConfig
}

func cleanupPartialHarness(
	t *testing.T,
	conn *grpc.ClientConn,
	stopServer func(),
	manager *daemon.Manager,
	sandboxMilvus *milvusclient.Client,
	operatorMilvus *milvusclient.Client,
	databaseName string,
	databaseCreated bool,
) {
	t.Helper()
	if conn != nil {
		_ = conn.Close()
	}
	if stopServer != nil {
		stopServer()
	}
	if manager != nil {
		closeCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		if err := manager.Close(closeCtx); err != nil {
			t.Errorf("close partial manager returned error: %v", err)
		}
		cancel()
	}
	if sandboxMilvus != nil {
		for _, cleanupErr := range dropEveryCollection(sandboxMilvus) {
			t.Errorf("clean partial sandbox Milvus: %v", cleanupErr)
		}
		closeMilvusClient(sandboxMilvus)
	}
	if databaseCreated && operatorMilvus != nil {
		dropCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		if err := operatorMilvus.DropDatabase(
			dropCtx,
			milvusclient.NewDropDatabaseOption(databaseName),
		); err != nil {
			t.Errorf("DropDatabase(%s) after partial setup returned error: %v", databaseName, err)
		}
		cancel()
	}
	closeMilvusClient(operatorMilvus)
}

// teardown drops every tracked collection and the unique temporary database.
// It then verifies the operator database and database list are unchanged.
func (h *harness) teardown(stopServer func()) {
	if err := h.conn.Close(); err != nil {
		h.t.Errorf("close gRPC connection returned error: %v", err)
	}
	stopServer()
	closeManagerContext, cancelManagerClose := context.WithTimeout(context.Background(), 15*time.Second)
	if err := h.manager.Close(closeManagerContext); err != nil {
		h.t.Errorf("close manager returned error: %v", err)
	}
	cancelManagerClose()
	for _, cleanupErr := range h.cleanupMilvus() {
		h.t.Error(cleanupErr)
	}

	calls := h.callRecorder.snapshot()
	h.t.Logf("Milvus calls: %+v", calls)
}

func (h *harness) cleanupMilvus() []error {
	cleanupErrors := make([]error, 0)
	temporaryNames := make([]string, 0, len(h.temporaryNames))
	for collectionName := range h.temporaryNames {
		temporaryNames = append(temporaryNames, collectionName)
	}
	slices.Sort(temporaryNames)
	for _, collectionName := range temporaryNames {
		if err := dropCollectionIfPresent(h.milvus, collectionName); err != nil {
			cleanupErrors = append(cleanupErrors, err)
		}
	}
	sandboxAfter, err := readMilvusInventory(h.milvus)
	if err != nil {
		cleanupErrors = append(cleanupErrors, fmt.Errorf("read sandbox Milvus inventory after: %w", err))
		cleanupErrors = append(cleanupErrors, dropEveryCollection(h.milvus)...)
	} else {
		h.t.Logf("Milvus sandbox inventory after: %v", sandboxAfter)
	}
	if err == nil && !reflect.DeepEqual(sandboxAfter, h.sandboxBefore) {
		cleanupErrors = append(cleanupErrors, fmt.Errorf(
			"temporary Milvus database inventory changed\nbefore: %v\nafter: %v",
			h.sandboxBefore,
			sandboxAfter,
		))
		cleanupErrors = append(cleanupErrors, dropEveryCollection(h.milvus)...)
	}
	closeMilvusClient(h.milvus)

	dropDatabaseCtx, cancelDropDatabase := context.WithTimeout(context.Background(), 15*time.Second)
	if err := h.operatorMilvus.DropDatabase(
		dropDatabaseCtx,
		milvusclient.NewDropDatabaseOption(h.databaseName),
	); err != nil {
		cleanupErrors = append(cleanupErrors, fmt.Errorf(
			"DropDatabase(%s) returned error: %w",
			h.databaseName,
			err,
		))
	}
	cancelDropDatabase()
	afterDatabases := h.beforeDatabases
	listedDatabases, databaseErr := listMilvusDatabases(h.operatorMilvus)
	if databaseErr != nil {
		cleanupErrors = append(cleanupErrors, fmt.Errorf("list Milvus databases after: %w", databaseErr))
	} else {
		afterDatabases = listedDatabases
		h.t.Logf("Milvus databases after: %v", afterDatabases)
	}
	operatorAfter := h.operatorBefore
	readOperatorAfter, inventoryErr := readMilvusInventory(h.operatorMilvus)
	if inventoryErr != nil {
		cleanupErrors = append(cleanupErrors, fmt.Errorf("read operator Milvus inventory after: %w", inventoryErr))
	} else {
		operatorAfter = readOperatorAfter
		h.t.Logf("Milvus operator inventory after: %v", operatorAfter)
	}
	audit := auditOperatorState(
		h.databaseName,
		h.beforeDatabases,
		afterDatabases,
		h.operatorBefore,
		operatorAfter,
		h.temporaryNames,
		h.callRecorder.snapshot(),
	)
	if len(audit.concurrentAdditions) > 0 {
		h.t.Logf("Concurrent operator additions: %v", audit.concurrentAdditions)
	}
	if len(audit.concurrentDatabases) > 0 {
		h.t.Logf("Database changes outside the harness database: %v", audit.concurrentDatabases)
	}
	for _, violation := range audit.violations {
		cleanupErrors = append(cleanupErrors, fmt.Errorf("%s", violation))
	}
	closeMilvusClient(h.operatorMilvus)
	return cleanupErrors
}

func auditOperatorState(
	databaseName string,
	beforeDatabases []string,
	afterDatabases []string,
	beforeInventory milvusInventory,
	afterInventory milvusInventory,
	temporaryNames map[string]struct{},
	calls []milvusCall,
) operatorStateAudit {
	audit := operatorStateAudit{
		violations: milvusIsolationViolations(databaseName, temporaryNames, calls),
	}
	hasHarnessMutationEvidence := len(audit.violations) > 0
	audit.concurrentDatabases, audit.violations = auditDatabaseInventory(databaseName, beforeDatabases, afterDatabases, audit.violations)
	baselineNames := make([]string, 0, len(beforeInventory))
	for collectionName := range beforeInventory {
		baselineNames = append(baselineNames, collectionName)
	}
	slices.Sort(baselineNames)
	for _, collectionName := range baselineNames {
		beforeProperties := beforeInventory[collectionName]
		afterProperties, present := afterInventory[collectionName]
		if !present {
			audit.violations = append(audit.violations, fmt.Sprintf(
				"operator Milvus baseline collection %q was removed",
				collectionName,
			))
			continue
		}
		if !reflect.DeepEqual(afterProperties, beforeProperties) {
			audit.violations = append(audit.violations, fmt.Sprintf(
				"operator Milvus baseline collection %q changed\nbefore: %v\nafter: %v",
				collectionName,
				beforeProperties,
				afterProperties,
			))
		}
	}
	addedNames := make([]string, 0)
	for collectionName := range afterInventory {
		if _, present := beforeInventory[collectionName]; !present {
			addedNames = append(addedNames, collectionName)
		}
	}
	slices.Sort(addedNames)
	for _, collectionName := range addedNames {
		if _, temporary := temporaryNames[collectionName]; temporary {
			audit.violations = append(audit.violations, fmt.Sprintf(
				"tracked temporary collection %q appeared in operator Milvus database",
				collectionName,
			))
			continue
		}
		if hasHarnessMutationEvidence {
			audit.violations = append(audit.violations, fmt.Sprintf(
				"untracked operator collection %q appeared with harness mutation evidence",
				collectionName,
			))
			continue
		}
		audit.concurrentAdditions = append(audit.concurrentAdditions, collectionName)
	}
	return audit
}

// auditDatabaseInventory compares the database lists before and after one
// test. After teardown, a database name that starts with databaseName is a
// violation: the harness left its own database behind. Any other added or
// removed database is returned as a change outside the harness database. The
// audit does not identify what made that change. The Milvus call recorder
// separately rejects a CreateDatabase or DropDatabase that the harness sends
// for any other database.
func auditDatabaseInventory(
	databaseName string,
	beforeDatabases []string,
	afterDatabases []string,
	violations []string,
) ([]string, []string) {
	concurrent := make([]string, 0)
	for _, name := range afterDatabases {
		if strings.HasPrefix(name, databaseName) {
			violations = append(violations, fmt.Sprintf(
				"temporary Milvus database %q remains after teardown",
				name,
			))
			continue
		}
		if !slices.Contains(beforeDatabases, name) {
			concurrent = append(concurrent, "added "+name)
		}
	}
	for _, name := range beforeDatabases {
		if !slices.Contains(afterDatabases, name) {
			concurrent = append(concurrent, "removed "+name)
		}
	}
	return concurrent, violations
}

func milvusIsolationViolations(
	databaseName string,
	temporaryNames map[string]struct{},
	calls []milvusCall,
) []string {
	violations := make([]string, 0)
	for _, call := range calls {
		if call.method == "CreateDatabase" || call.method == "DropDatabase" {
			if call.databaseName != databaseName {
				violations = append(violations, fmt.Sprintf(
					"Milvus call %s targeted database %q, want temporary database %q",
					call.method,
					call.databaseName,
					databaseName,
				))
			}
			continue
		}
		if !protectedMilvusCall(call.method) {
			continue
		}
		if call.databaseName != databaseName {
			violations = append(violations, fmt.Sprintf(
				"protected Milvus call %s targeted database %q, want temporary database %q",
				call.method,
				call.databaseName,
				databaseName,
			))
			continue
		}
		if call.destinationDatabaseName != "" && call.destinationDatabaseName != databaseName {
			violations = append(violations, fmt.Sprintf(
				"protected Milvus call %s targeted destination database %q, want temporary database %q",
				call.method,
				call.destinationDatabaseName,
				databaseName,
			))
			continue
		}
		if len(call.collectionNames) == 0 {
			violations = append(violations, fmt.Sprintf(
				"protected Milvus call %s in database %s has no auditable collection target",
				call.method,
				call.databaseName,
			))
			continue
		}
		for _, collectionName := range call.collectionNames {
			if _, temporary := temporaryNames[collectionName]; !temporary {
				violations = append(violations, fmt.Sprintf(
					"protected Milvus call %s touched untracked collection %s in database %s",
					call.method,
					collectionName,
					call.databaseName,
				))
			}
		}
	}
	return violations
}

func protectedMilvusCall(method string) bool {
	if strings.HasPrefix(method, "Alter") {
		return true
	}
	return slices.Contains([]string{
		"CreateAlias", "CreateCollection", "CreateIndex", "CreatePartition",
		"Delete", "DropAlias", "DropCollection", "DropIndex", "DropPartition",
		"Flush", "FlushAll", "Import", "Insert", "LoadCollection",
		"ReleaseCollection", "RenameCollection", "ReplicateMessage",
		"TruncateCollection", "Upsert",
	}, method)
}

func (h *harness) trackTemporaryCollection(collectionName string) {
	h.t.Helper()
	if collectionName == "" {
		return
	}
	if _, preexisting := h.sandboxBefore[collectionName]; preexisting {
		h.t.Fatalf("temporary collection %q collides with a preexisting collection", collectionName)
	}
	h.temporaryNames[collectionName] = struct{}{}
}

func (h *harness) trackCollectionFamily(collectionName string) {
	h.t.Helper()
	for _, name := range []string{
		collectionName,
		collectionName + "_stg",
		collectionName + "_swap_previous",
	} {
		h.trackTemporaryCollection(name)
	}
}

func (h *harness) trackCodebasePath(codebasePath string) string {
	h.t.Helper()
	if h.config.CollectionNameOverride != "" {
		h.t.Fatalf("live harness requires an empty collection name override, got %q", h.config.CollectionNameOverride)
	}
	canonicalPath, err := filepath.EvalSymlinks(codebasePath)
	if err != nil {
		h.t.Fatalf("resolve live codebase path: %v", err)
	}
	prefix := "code_chunks"
	if h.config.HybridMode {
		prefix = "hybrid_code_chunks"
	}
	collectionName := prefix + "_" + tshash.PathPrefix(canonicalPath)
	h.trackCollectionFamily(collectionName)
	return collectionName
}

func closeMilvusClient(client *milvusclient.Client) {
	if client == nil {
		return
	}
	closeCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	_ = client.Close(closeCtx)
	cancel()
}

// resolveLiveConfig takes the sandbox isolation and names the parts this suite
// needs to differ: a real Milvus and a fake embedder.
//
// Naming those first and letting the sandbox fill in the rest is what proves
// each sandbox value is a default rather than a forced setting. If this stops
// being expressible, the resolver has started forcing values and the resolver is
// what should change.
func resolveLiveConfig(
	t *testing.T,
	sandboxRoot string,
	socketPath string,
	embedServerURL string,
	milvusAddress string,
	milvusToken string,
	databaseName string,
	harnessID string,
	idleTimeout time.Duration,
) config.Config {
	t.Helper()

	chosen := []struct {
		name  string
		value string
	}{
		// The real store, which is what this suite exists to exercise.
		{name: "CLAUDE_CONTEXT_PROFILE", value: config.ProfileStandard},
		{name: "MILVUS_ADDRESS", value: milvusAddress},
		{name: "MILVUS_TOKEN", value: milvusToken},
		{name: "MILVUS_DATABASE", value: databaseName},
		// A local fake stands in for the embedder, so no run spends GPU time or
		// depends on a model server being up.
		{name: "EMBEDDING_PROVIDER", value: "OpenAI"},
		{name: "EMBEDDING_MODEL", value: "live-harness-" + harnessID},
		{name: "OPENAI_BASE_URL", value: embedServerURL},
		{name: "OPENAI_API_KEY", value: "live-harness-dummy-key"}, //gitleaks:allow // not a secret: the fake embedder accepts any non-empty key
		{name: "EMBEDDING_DIMENSION", value: strconv.Itoa(fakeEmbeddingDimension)},
		{name: "EMBEDDING_BATCH_SIZE", value: "8"},
		// The sandbox default sits under a temp root long enough to overflow
		// the platform's socket path limit.
		{name: "CLAUDE_CONTEXTD_SOCKET_PATH", value: socketPath},
		// Background work is off so a scenario observes only what it asked for.
		{name: "CLAUDE_CONTEXT_BACKGROUND_SYNC", value: "false"},
		{name: "CLAUDE_CONTEXT_TRIGGER_WATCHER", value: "false"},
		{name: "CLAUDE_CONTEXT_FILE_WATCHER", value: "false"},
		{name: "CLAUDE_CONTEXT_DEBUG_LISTENER", value: "false"},
		{name: "CLAUDE_CONTEXT_PERF_COUNTERS_INTERVAL_MS", value: "0"},
		{name: "CLAUDE_CONTEXT_MAX_CONCURRENT_INDEX_JOBS", value: "1"},
		{name: "CLAUDE_CONTEXT_RESUME_ON_BOOT", value: "false"},
		{name: "CLAUDE_CONTEXT_MILVUS_COLLECTION_IDLE_TIMEOUT_MS", value: strconv.FormatInt(idleTimeout.Milliseconds(), 10)},
	}
	for _, setting := range chosen {
		t.Setenv(setting.name, setting.value)
	}
	for _, variable := range sandbox.Env(sandboxRoot) {
		if _, alreadySet := os.LookupEnv(variable.Name); alreadySet {
			continue
		}
		t.Setenv(variable.Name, variable.Value)
	}

	resolved, err := config.Default()
	if err != nil {
		t.Fatalf("resolve live config through config.Default: %v", err)
	}
	if resolved.MilvusAddress == "" {
		t.Fatal("resolved config has no Milvus address; this suite must run against the real store")
	}
	if resolved.MilvusDatabase != databaseName {
		t.Fatalf(
			"resolved MilvusDatabase = %q, want temporary database %q",
			resolved.MilvusDatabase,
			databaseName,
		)
	}
	if resolved.OpenAIBaseURL != embedServerURL {
		t.Fatalf(
			"resolved OpenAIBaseURL = %q, want the fake embedder at %q",
			resolved.OpenAIBaseURL,
			embedServerURL,
		)
	}
	return resolved
}
