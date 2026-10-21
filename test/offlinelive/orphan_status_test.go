//go:build offlinelive

package offlinelive

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/store"
)

func TestOrphanIndexingStatusDoesNotReportActivePreparation(t *testing.T) {
	harness := newUnstartedHarness(t)
	options := harnessOptions{}
	cfg := resolveOfflineConfig(t, harness.stateRoot, harness.socketPath, options)
	prepareState(t, cfg)
	const codebaseID = "cb_1234567890_0123456789ab"
	codebasePath, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("resolve source directory: %v", err)
	}
	registry := model.RegistryFile{Codebases: []model.Codebase{{
		ID: codebaseID, Kind: model.CodebaseKindCode,
		CanonicalPath: codebasePath, Status: model.CodebaseStatusIndexing,
		UpdatedAt: time.Now().Add(-time.Hour),
	}}}
	if err := store.WriteRegistry(cfg.RegistryPath, registry); err != nil {
		t.Fatalf("write interrupted registry: %v", err)
	}
	harness.start(options, false)
	t.Cleanup(harness.teardown)
	response, err := harness.client.GetIndex(correlatedContext(), &pb.GetIndexRequest{Path: codebasePath, Client: harnessClientInfo()})
	if err != nil {
		t.Fatalf("GetIndex: %v", err)
	}
	if response.GetCodebase().GetStatus() != string(model.CodebaseStatusIndexing) {
		t.Fatalf("GetIndex changed the persisted lifecycle: %s", response.GetCodebase())
	}
	if got := response.GetCodebase().GetDisplayStatus(); got != "stale" {
		t.Fatalf("orphan display = %q, want stale", got)
	}
	status, err := harness.client.GetStatus(context.Background(), &pb.GetStatusRequest{})
	if err != nil {
		t.Fatalf("GetStatus: %v", err)
	}
	if len(status.GetActivity()) != 0 {
		t.Fatalf("orphan status reported active work: %s", status.GetActivity())
	}
	counts := make(map[string]int64)
	for _, metric := range status.GetMetrics() {
		counts[metric.GetName()] = metric.GetIntValue()
	}
	if counts["codebases.status=preparing"] != 0 || counts["codebases.status=stale"] != 1 {
		t.Fatalf("orphan status counts = %v, want preparing 0 and stale 1", counts)
	}
	t.Log("GetIndex preserved indexing metadata and reported stale; GetStatus reported stale 1, preparing 0, and empty activity.")
}
