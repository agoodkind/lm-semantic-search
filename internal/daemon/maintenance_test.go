package daemon

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/adapterr"
	"goodkind.io/lm-semantic-search/internal/config"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// maintenanceScene is an indexed codebase whose backend records every call a
// sweep or a search would make against the store, so a test can prove the
// daemon made none of them during maintenance and all of them afterwards.
type maintenanceScene struct {
	manager      *Manager
	server       *GRPCServer
	repoPath     string
	fake         *fakeSemantic
	listCalls    atomic.Int32
	acquireCalls atomic.Int32
	mmapCalls    atomic.Int32
}

func newMaintenanceScene(t *testing.T) *maintenanceScene {
	t.Helper()
	manager, repoPath := newIndexedSearchManager(t)
	scene := &maintenanceScene{
		manager:      manager,
		server:       NewGRPCServer(manager, nil),
		repoPath:     repoPath,
		fake:         nil,
		listCalls:    atomic.Int32{},
		acquireCalls: atomic.Int32{},
		mmapCalls:    atomic.Int32{},
	}
	scene.fake = &fakeSemantic{
		listCollections: func(context.Context) ([]string, error) {
			scene.listCalls.Add(1)
			return []string{}, nil
		},
		acquireCollection: func(context.Context, string) (semantic.CollectionLease, error) {
			scene.acquireCalls.Add(1)
			return fakeCollectionLease{}, nil
		},
		ensureMmap: func(context.Context) {
			scene.mmapCalls.Add(1)
		},
		search: func(context.Context, string, string, int32, []string, string) ([]model.StoredChunk, error) {
			return []model.StoredChunk{{Content: "package main", RelativePath: "main.go"}}, nil
		},
	}
	manager.semantic = scene.fake
	return scene
}

func (scene *maintenanceScene) setMaintenance(t *testing.T, enabled bool, reason string) *pb.SetMaintenanceModeResponse {
	t.Helper()
	response, err := scene.server.SetMaintenanceMode(context.Background(), &pb.SetMaintenanceModeRequest{
		Enabled: enabled,
		Reason:  reason,
		Client:  &pb.ClientInfo{Name: "maintenance-test", Pid: 0},
	})
	if err != nil {
		t.Fatalf("SetMaintenanceMode(%t) returned error: %v", enabled, err)
	}
	return response
}

// Turning maintenance on stops every daemon-driven store interaction: the
// periodic sweep and the store-maintenance sweep make no store call and start
// no job, a search fails fast with the maintenance status without acquiring a
// collection, an index request is refused, and every status surface reports
// the mode. Turning it off restores all of it.
func TestMaintenanceModePausesSweepsAndSearchUntilTurnedOff(t *testing.T) {
	scene := newMaintenanceScene(t)
	syncer := NewBackgroundSync(scene.manager.config, scene.manager)

	on := scene.setMaintenance(t, true, "milvus restore")
	if !on.GetMaintenance().GetEnabled() || on.GetMaintenance().GetReason() != "milvus restore" {
		t.Fatalf("SetMaintenanceMode reply = %v, want enabled with the reason", on.GetMaintenance())
	}
	if !strings.Contains(on.GetDisplayText(), "Maintenance mode is on") {
		t.Fatalf("SetMaintenanceMode reply text lacks the banner:\n%s", on.GetDisplayText())
	}
	if !scene.fake.maintenanceGate.Load() {
		t.Fatal("the backend load gate was not closed")
	}

	syncer.runSyncAll(context.Background(), "interval")
	syncer.runPeriodicMaintenanceOnce(context.Background())
	if calls := scene.listCalls.Load(); calls != 0 {
		t.Fatalf("sweeps listed collections %d times during maintenance, want 0", calls)
	}
	if calls := scene.mmapCalls.Load(); calls != 0 {
		t.Fatalf("store-maintenance sweep ran mmap migration %d times during maintenance, want 0", calls)
	}
	if jobs := scene.manager.ListJobs(""); len(jobs) != 0 {
		t.Fatalf("sweeps registered %d jobs during maintenance, want 0", len(jobs))
	}

	_, searchErr := scene.server.SearchCode(context.Background(), &pb.SearchCodeRequest{
		Path:   scene.repoPath,
		Query:  "needle",
		Limit:  5,
		Client: &pb.ClientInfo{Name: "maintenance-test", Pid: 0},
	})
	assertMaintenanceRefusal(t, "SearchCode", searchErr)
	if calls := scene.acquireCalls.Load(); calls != 0 {
		t.Fatalf("search acquired a collection %d times during maintenance, want 0", calls)
	}

	_, indexErr := scene.server.StartIndex(context.Background(), &pb.StartIndexRequest{
		Path:   scene.repoPath,
		Client: &pb.ClientInfo{Name: "maintenance-test", Pid: 0},
	})
	assertMaintenanceRefusal(t, "StartIndex", indexErr)

	getIndex, err := scene.server.GetIndex(context.Background(), &pb.GetIndexRequest{Path: scene.repoPath})
	if err != nil {
		t.Fatalf("GetIndex returned error: %v", err)
	}
	if !getIndex.GetMaintenance().GetEnabled() {
		t.Fatal("GetIndex did not report maintenance mode")
	}
	if getIndex.Searchable == nil || getIndex.GetSearchable() {
		t.Fatalf("GetIndex searchable = %v during maintenance, want false", getIndex.Searchable)
	}
	if !strings.Contains(getIndex.GetDisplayText(), "Maintenance mode is on") || !strings.Contains(getIndex.GetDisplayText(), "milvus restore") {
		t.Fatalf("GetIndex text lacks the maintenance banner:\n%s", getIndex.GetDisplayText())
	}
	getStatus, err := scene.server.GetStatus(context.Background(), &pb.GetStatusRequest{})
	if err != nil {
		t.Fatalf("GetStatus returned error: %v", err)
	}
	if !getStatus.GetMaintenance().GetEnabled() {
		t.Fatal("GetStatus did not report maintenance mode")
	}
	if !strings.Contains(getStatus.GetDisplayText(), "maintenance.enabled true") {
		t.Fatalf("GetStatus text lacks the maintenance metric:\n%s", getStatus.GetDisplayText())
	}

	off := scene.setMaintenance(t, false, "")
	if off.GetMaintenance().GetEnabled() {
		t.Fatal("SetMaintenanceMode(false) left the mode on")
	}
	if scene.fake.maintenanceGate.Load() {
		t.Fatal("the backend load gate was not reopened")
	}
	syncer.runSyncAll(context.Background(), "interval")
	syncer.runPeriodicMaintenanceOnce(context.Background())
	if calls := scene.listCalls.Load(); calls == 0 {
		t.Fatal("the sweep after maintenance ended did not reach the store")
	}
	if calls := scene.mmapCalls.Load(); calls == 0 {
		t.Fatal("the store-maintenance sweep after maintenance ended did not run")
	}
	searchResponse, err := scene.server.SearchCode(context.Background(), &pb.SearchCodeRequest{
		Path:   scene.repoPath,
		Query:  "needle",
		Limit:  5,
		Client: &pb.ClientInfo{Name: "maintenance-test", Pid: 0},
	})
	if err != nil {
		t.Fatalf("SearchCode after maintenance ended returned error: %v", err)
	}
	if len(searchResponse.GetResults()) != 1 || scene.acquireCalls.Load() != 1 {
		t.Fatalf("search after maintenance ended returned %d results with %d acquires, want 1 and 1", len(searchResponse.GetResults()), scene.acquireCalls.Load())
	}
	getIndex, err = scene.server.GetIndex(context.Background(), &pb.GetIndexRequest{Path: scene.repoPath})
	if err != nil {
		t.Fatalf("GetIndex after maintenance ended returned error: %v", err)
	}
	if getIndex.GetMaintenance().GetEnabled() || strings.Contains(getIndex.GetDisplayText(), "Maintenance mode is on") {
		t.Fatalf("GetIndex still reports maintenance after it ended:\n%s", getIndex.GetDisplayText())
	}
}

// The mode is persisted beside the registry, so a daemon that restarts during
// a store restore comes back paused, with its load gate closed and its boot
// resume skipped, rather than resuming loads into a half-restored store.
func TestMaintenanceModeSurvivesDaemonRestart(t *testing.T) {
	scene := newMaintenanceScene(t)
	scene.setMaintenance(t, true, "milvus restore")

	restartedBackend := &fakeSemantic{}
	restarted, err := newManagerWithSemanticFactory(context.Background(), scene.manager.config, func(context.Context, config.Config) (semanticIndex, error) {
		return restartedBackend, nil
	})
	if err != nil {
		t.Fatalf("NewManager after restart returned error: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := restarted.cancelAndWaitForJobs(ctx); err != nil {
			t.Errorf("cancelAndWaitForJobs: %v", err)
		}
		restarted.CloseGraphEngines()
	})

	state := restarted.Maintenance()
	if !state.Enabled || state.Reason != "milvus restore" || state.Since.IsZero() {
		t.Fatalf("restarted maintenance state = %+v, want enabled with reason and start time", state)
	}
	if !restartedBackend.maintenanceGate.Load() {
		t.Fatal("restart did not close the backend load gate")
	}
	server := NewGRPCServer(restarted, nil)
	_, searchErr := server.SearchCode(context.Background(), &pb.SearchCodeRequest{
		Path:   scene.repoPath,
		Query:  "needle",
		Limit:  5,
		Client: &pb.ClientInfo{Name: "maintenance-test", Pid: 0},
	})
	assertMaintenanceRefusal(t, "SearchCode after restart", searchErr)
}

func assertMaintenanceRefusal(t *testing.T, operation string, err error) {
	t.Helper()
	if err == nil {
		t.Fatalf("%s succeeded during maintenance, want the maintenance refusal", operation)
	}
	grpcStatus, ok := status.FromError(err)
	if !ok {
		t.Fatalf("%s error %v is not a gRPC status", operation, err)
	}
	if grpcStatus.Code() != codes.FailedPrecondition {
		t.Fatalf("%s code = %s, want FailedPrecondition", operation, grpcStatus.Code())
	}
	if !strings.Contains(grpcStatus.Message(), "maintenance mode") || !strings.Contains(grpcStatus.Message(), "milvus restore") {
		t.Fatalf("%s message %q does not name maintenance mode and its reason", operation, grpcStatus.Message())
	}
}

// The manager-level refusal carries the maintenance class, so every boundary
// maps it to the same code and message.
func TestMaintenanceRefusalCarriesClass(t *testing.T) {
	scene := newMaintenanceScene(t)
	scene.setMaintenance(t, true, "backup")
	_, err := scene.manager.SearchCode(context.Background(), scene.repoPath, "needle", 5, nil)
	var adapterError *adapterr.AdapterError
	if !errors.As(err, &adapterError) || adapterError.Class != adapterr.ClassMaintenance {
		t.Fatalf("SearchCode error = %v, want the maintenance class", err)
	}
	if got := filepath.Base(scene.manager.maintenancePath()); got != "maintenance.json" {
		t.Fatalf("maintenance file = %q, want maintenance.json beside the registry", got)
	}
}
