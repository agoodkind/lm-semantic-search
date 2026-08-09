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
	cryptorand "crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
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
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/daemon"
	"goodkind.io/lm-semantic-search/internal/grpcutil"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/sandbox"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"goodkind.io/lm-semantic-search/internal/semantic/milvusgrpc"
	"goodkind.io/lm-semantic-search/internal/store"
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
	milvusContext     context.Context
}

type milvusInventory map[string]map[string]string

type operatorStateAudit struct {
	violations          []string
	concurrentAdditions []string
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
		recordedAt:              time.Now(),
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

func newHarnessWithOptions(
	t *testing.T,
	gate *embedGate,
	idleTimeout time.Duration,
	requireMilvus bool,
) *harness {
	t.Helper()

	defaultConfig, err := config.Default()
	if err != nil {
		t.Fatalf("config.Default returned error: %v", err)
	}
	milvusAddress := strings.TrimSpace(defaultConfig.MilvusAddress)
	if milvusAddress == "" {
		if requireMilvus {
			t.Fatal("BLOCKED: MilvusAddress is empty; set MILVUS_ADDRESS or local config before running the residency suite")
		}
		t.Skip("BLOCKED: MilvusAddress is empty; set MILVUS_ADDRESS or local config before running the live suite")
	}

	harnessID := randomID()
	databaseName := liveDatabasePrefix + harnessID
	callRecorder := &milvusCallRecorder{}
	operatorContext := context.WithValue(
		context.Background(),
		milvusgrpc.CallObserverContextKey{},
		milvusgrpc.CallObserver(callRecorder.observe),
	)
	// Probe Milvus directly first. A dial failure here means the backend is down,
	// so the whole scenario is blocked on the environment rather than the code.
	dialCtx, dialCancel := context.WithTimeout(operatorContext, 5*time.Second)
	operatorMilvus, err := milvusclient.New(dialCtx, &milvusclient.ClientConfig{
		Address:     milvusAddress,
		APIKey:      defaultConfig.MilvusToken,
		DialOptions: milvusgrpc.DialOptions(operatorContext, slog.Default(), milvusgrpc.DefaultCallTimeouts()),
	})
	dialCancel()
	if err != nil {
		if requireMilvus {
			t.Fatalf("BLOCKED: Milvus unreachable at %s: %v", milvusAddress, err)
		}
		t.Skipf("BLOCKED: Milvus unreachable at %s: %v", milvusAddress, err)
	}

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

	sandboxContext := context.WithValue(
		context.Background(),
		milvusgrpc.CallObserverContextKey{},
		milvusgrpc.CallObserver(callRecorder.observe),
	)
	dialCtx, dialCancel = context.WithTimeout(sandboxContext, 5*time.Second)
	sandboxMilvus, err = milvusclient.New(dialCtx, &milvusclient.ClientConfig{
		Address:     milvusAddress,
		APIKey:      defaultConfig.MilvusToken,
		DBName:      databaseName,
		DialOptions: milvusgrpc.DialOptions(sandboxContext, slog.Default(), milvusgrpc.DefaultCallTimeouts()),
	})
	dialCancel()
	if err != nil {
		t.Fatalf("connect to temporary Milvus database %s: %v", databaseName, err)
	}
	sandboxBefore, err := readMilvusInventory(sandboxMilvus)
	if err != nil {
		t.Fatalf("read sandbox Milvus inventory before: %v", err)
	}
	if len(sandboxBefore) != 0 {
		t.Fatalf(
			"temporary Milvus database %q started with collections: %v",
			databaseName,
			sandboxBefore,
		)
	}

	stateRoot := t.TempDir()
	// The unix socket path must fit macOS's ~104-char sun_path limit, and
	// t.TempDir lives under a long /var/folders path that overflows it, so the
	// socket gets a short /tmp dir instead. State and merkle can use the long temp
	// root.
	socketDir, err := os.MkdirTemp("/tmp", "lms-live-")
	if err != nil {
		t.Fatalf("mkdir short socket dir returned error: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(socketDir) })
	socketPath := filepath.Join(socketDir, "daemon.sock")

	embeddingRecorder := &embeddingCallRecorder{}
	embedServer := newFakeEmbeddingServerWithRecorder(
		t,
		gate,
		fakeEmbeddingDimension,
		embeddingRecorder,
	)

	cfg := resolveLiveConfig(
		t,
		stateRoot,
		socketPath,
		embedServer.URL,
		milvusAddress,
		defaultConfig.MilvusToken,
		databaseName,
		harnessID,
		idleTimeout,
	)
	for _, dir := range sandbox.Directories(cfg) {
		if err := store.EnsureDir(dir); err != nil {
			t.Fatalf("EnsureDir(%s) returned error: %v", dir, err)
		}
	}
	if err := store.WriteRegistry(cfg.RegistryPath, model.RegistryFile{}); err != nil {
		t.Fatalf("WriteRegistry returned error: %v", err)
	}

	manager, err = daemon.NewManager(sandboxContext, cfg)
	if err != nil {
		t.Fatalf("NewManager returned error: %v", err)
	}

	stopServer = startInProcessServer(t, manager, socketPath)

	conn, client, err := grpcutil.DialDaemon(context.Background(), socketPath)
	if err != nil {
		t.Fatalf("DialDaemon returned error: %v", err)
	}

	// A fresh random id derives a unique conv_chunks_<hash> collection name, so
	// the throwaway collection can never be the production one.
	collectionID := "live-marker-" + harnessID
	codebase, err := manager.RegisterConversationCollection(context.Background(), collectionID)
	if err != nil {
		t.Fatalf("RegisterConversationCollection returned error: %v", err)
	}
	if codebase.CollectionName == "" {
		t.Fatal("RegisterConversationCollection returned an empty collection name")
	}
	if codebase.CollectionName == productionConversationCollection {
		t.Fatalf("throwaway collection name equals production %q; refusing to run", productionConversationCollection)
	}

	h := &harness{
		t:                 t,
		config:            cfg,
		manager:           manager,
		conn:              conn,
		client:            client,
		operatorMilvus:    operatorMilvus,
		milvus:            sandboxMilvus,
		databaseName:      databaseName,
		collectionID:      collectionID,
		collectionName:    codebase.CollectionName,
		reuseCatalogName:  semantic.ReuseCatalogCollectionName(cfg),
		codebaseID:        codebase.ID,
		stateRoot:         stateRoot,
		merkleDir:         cfg.MerkleDir,
		embedGate:         gate,
		beforeDatabases:   beforeDatabases,
		operatorBefore:    operatorBefore,
		sandboxBefore:     sandboxBefore,
		temporaryNames:    make(map[string]struct{}),
		callRecorder:      callRecorder,
		embeddingRecorder: embeddingRecorder,
		milvusContext:     sandboxContext,
	}
	h.trackCollectionFamily(codebase.CollectionName)
	h.trackTemporaryCollection(h.reuseCatalogName)
	t.Cleanup(func() { h.teardown(stopServer) })
	setupComplete = true
	return h
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
	if !reflect.DeepEqual(afterDatabases, beforeDatabases) {
		audit.violations = append(audit.violations, fmt.Sprintf(
			"Milvus database inventory changed\nbefore: %v\nafter: %v",
			beforeDatabases,
			afterDatabases,
		))
	}
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
	switch method {
	case "CreateAlias", "CreateCollection", "CreateIndex", "CreatePartition",
		"Delete", "DropAlias", "DropCollection", "DropIndex", "DropPartition",
		"Flush", "FlushAll", "Import", "Insert", "LoadCollection",
		"ReleaseCollection", "RenameCollection", "ReplicateMessage",
		"TruncateCollection", "Upsert":
		return true
	default:
		return false
	}
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

func readMilvusInventory(client *milvusclient.Client) (milvusInventory, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	collectionNames, err := client.ListCollections(ctx, milvusclient.NewListCollectionOption())
	if err != nil {
		return nil, fmt.Errorf("list Milvus collections for inventory: %w", err)
	}
	slices.Sort(collectionNames)
	inventory := make(milvusInventory, len(collectionNames))
	for _, collectionName := range collectionNames {
		properties := make(map[string]string)
		loadState, loadErr := client.GetLoadState(
			ctx,
			milvusclient.NewGetLoadStateOption(collectionName),
		)
		if loadErr != nil {
			return nil, fmt.Errorf(
				"get Milvus load state for %s inventory: %w",
				collectionName,
				loadErr,
			)
		}
		properties["load_state"] = strconv.FormatInt(int64(loadState.State), 10)
		collection, describeErr := client.DescribeCollection(
			ctx,
			milvusclient.NewDescribeCollectionOption(collectionName),
		)
		if describeErr != nil {
			return nil, fmt.Errorf(
				"describe Milvus collection %s for inventory: %w",
				collectionName,
				describeErr,
			)
		}
		for key, value := range collection.Properties {
			properties["collection:"+key] = value
		}
		if collection.Schema != nil {
			for _, field := range collection.Schema.Fields {
				properties["field:"+field.Name] = field.TypeParams["mmap.enabled"]
			}
		}
		indexNames, listErr := client.ListIndexes(
			ctx,
			milvusclient.NewListIndexOption(collectionName),
		)
		if listErr != nil {
			return nil, fmt.Errorf(
				"list Milvus indexes for %s inventory: %w",
				collectionName,
				listErr,
			)
		}
		slices.Sort(indexNames)
		for _, indexName := range indexNames {
			description, indexErr := client.DescribeIndex(
				ctx,
				milvusclient.NewDescribeIndexOption(collectionName, indexName),
			)
			if indexErr != nil {
				return nil, fmt.Errorf(
					"describe Milvus index %s on %s: %w",
					indexName,
					collectionName,
					indexErr,
				)
			}
			properties["index:"+indexName] = description.Params()["mmap.enabled"]
		}
		inventory[collectionName] = properties
	}
	return inventory, nil
}

func listMilvusDatabases(client *milvusclient.Client) ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	databaseNames, err := client.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil {
		return nil, fmt.Errorf("list Milvus databases: %w", err)
	}
	slices.Sort(databaseNames)
	return databaseNames, nil
}

func dropCollectionIfPresent(
	client *milvusclient.Client,
	collectionName string,
) error {
	dropCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := client.DropCollection(
		dropCtx,
		milvusclient.NewDropCollectionOption(collectionName),
	); err != nil {
		if !strings.Contains(err.Error(), "not exist") && !strings.Contains(err.Error(), "not found") {
			return fmt.Errorf("DropCollection(%s) returned error: %w", collectionName, err)
		}
	}
	return nil
}

func dropEveryCollection(client *milvusclient.Client) []error {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	collectionNames, err := client.ListCollections(ctx, milvusclient.NewListCollectionOption())
	cancel()
	if err != nil {
		return []error{fmt.Errorf("list temporary Milvus collections for cleanup: %w", err)}
	}
	cleanupErrors := make([]error, 0)
	slices.Sort(collectionNames)
	for _, collectionName := range collectionNames {
		if err := dropCollectionIfPresent(client, collectionName); err != nil {
			cleanupErrors = append(cleanupErrors, err)
		}
	}
	return cleanupErrors
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

// startInProcessServer serves the daemon gRPC service on a throwaway unix socket
// in a goroutine and returns a stop closure that GracefulStops the server and
// removes the socket. Readiness is a successful dial by the caller, so no log
// tailing is needed. It mirrors internal/daemon's own test helper.
func startInProcessServer(t *testing.T, manager *daemon.Manager, socketPath string) func() {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(socketPath), 0o755); err != nil {
		t.Fatalf("mkdir socket dir returned error: %v", err)
	}
	_ = os.Remove(socketPath)

	listener, err := (&net.ListenConfig{}).Listen(context.Background(), "unix", socketPath)
	if err != nil {
		t.Fatalf("listen on unix socket returned error: %v", err)
	}
	server := grpc.NewServer(
		grpc.MaxRecvMsgSize(grpcutil.MaxMessageBytes),
		grpc.MaxSendMsgSize(grpcutil.MaxMessageBytes),
	)
	pb.RegisterSemanticSearchDaemonServiceServer(server, daemon.NewGRPCServer(manager, nil))
	go func() {
		_ = server.Serve(listener)
	}()
	return func() {
		server.GracefulStop()
		_ = listener.Close()
		_ = os.Remove(socketPath)
	}
}

// newFakeEmbeddingServer starts a local OpenAI-compatible embedding endpoint. It
// answers the health probe (GET .../models) with a minimal models list and every
// embed request (POST .../embeddings) with one fixed-width vector per input,
// keyed by a content hash so identical content yields an identical vector and the
// engine's content-hash reuse path stays exercised.
// embedGate lets a test pace embedding requests. When installed, every embed
// request announces its batch size on arrived, then blocks until the test sends
// on release, so the test can read job progress between batches. The models
// (health) route is never gated.
type embedGate struct {
	arrived chan int
	release chan struct{}
}

func newFakeEmbeddingServer(t *testing.T, gate *embedGate) *httptest.Server {
	return newFakeEmbeddingServerWithDimension(t, gate, fakeEmbeddingDimension)
}

func newFakeEmbeddingServerWithDimension(
	t *testing.T,
	gate *embedGate,
	dimension int,
) *httptest.Server {
	return newFakeEmbeddingServerWithRecorder(t, gate, dimension, nil)
}

func newFakeEmbeddingServerWithRecorder(
	t *testing.T,
	gate *embedGate,
	dimension int,
	recorder *embeddingCallRecorder,
) *httptest.Server {
	t.Helper()
	handler := http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch {
		case strings.HasSuffix(request.URL.Path, "/models"):
			writeModelsList(writer)
		case strings.HasSuffix(request.URL.Path, "/embeddings"):
			writeEmbeddings(t, writer, request, gate, dimension, recorder)
		default:
			http.Error(writer, "unexpected path "+request.URL.Path, http.StatusNotFound)
		}
	})
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return server
}

func writeModelsList(writer http.ResponseWriter) {
	writer.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(writer).Encode(map[string]any{
		"object": "list",
		"data": []map[string]any{
			{"id": "text-embedding-3-small", "object": "model", "created": 0, "owned_by": "live-harness"},
		},
	})
}

func writeEmbeddings(
	t *testing.T,
	writer http.ResponseWriter,
	request *http.Request,
	gate *embedGate,
	dimension int,
	recorder *embeddingCallRecorder,
) {
	inputs, err := decodeEmbeddingInputs(request)
	if err != nil {
		http.Error(writer, err.Error(), http.StatusBadRequest)
		return
	}
	if recorder != nil {
		recorder.record(inputs)
	}
	if gate != nil {
		gate.arrived <- len(inputs)
		<-gate.release
	}
	type row struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
	}
	rows := make([]row, 0, len(inputs))
	for index, text := range inputs {
		rows = append(rows, row{
			Object:    "embedding",
			Index:     index,
			Embedding: deterministicVector(text, dimension),
		})
	}
	writer.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(writer).Encode(map[string]any{
		"object": "list",
		"model":  "text-embedding-3-small",
		"data":   rows,
		"usage":  map[string]int{"prompt_tokens": 1, "total_tokens": 1},
	}); err != nil {
		t.Logf("encode embedding response failed: %v", err)
	}
}

// decodeEmbeddingInputs reads the request's input field, accepting both the array
// form the batch embedder sends and a bare single string, so the fake is robust
// to either shape.
func decodeEmbeddingInputs(request *http.Request) ([]string, error) {
	var body struct {
		Input json.RawMessage `json:"input"`
	}
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("decode embedding request: %w", err)
	}
	var asArray []string
	if err := json.Unmarshal(body.Input, &asArray); err == nil {
		return asArray, nil
	}
	var asString string
	if err := json.Unmarshal(body.Input, &asString); err == nil {
		return []string{asString}, nil
	}
	return nil, fmt.Errorf("embedding request input was neither an array nor a string")
}

// deterministicVector maps content to a fixed-width unit vector derived from its
// SHA-256 digest, so identical content always yields an identical vector (reuse
// works) and distinct content yields a distinct one.
func deterministicVector(content string, dimension int) []float64 {
	digest := sha256.Sum256([]byte(content))
	vector := make([]float64, dimension)
	var norm float64
	for i := range dimension {
		value := (float64(digest[i%len(digest)]) - 128.0) / 128.0
		vector[i] = value
		norm += value * value
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		vector[0] = 1
		return vector
	}
	for i := range vector {
		vector[i] /= norm
	}
	return vector
}

// correlatedContext wraps ctx with the trace/span identity the daemon requires in
// strict mode, so every RPC and manager read carries a correlation.
func correlatedContext() context.Context {
	return grpcutil.WithCorrelation(context.Background())
}

// randomID returns a hex token unique per test, so each run's collection id (and
// therefore its derived Milvus collection name) is fresh and never collides with
// another run or with production.
func randomID() string {
	buffer := make([]byte, 16)
	if _, err := cryptorand.Read(buffer); err != nil {
		return strconv.FormatInt(time.Now().UnixNano(), 16)
	}
	return hex.EncodeToString(buffer)
}
