//go:build live

package live

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/daemon"
)

func TestLibraryCodebaseRepairRecognizesPublishedNamespaces(t *testing.T) {
	for _, shape := range []string{"published", "empty", "whitespace", "all_deleted"} {
		t.Run(shape, func(t *testing.T) {
			codebaseDaemon := newLibraryCodebaseDaemon(t)
			root := t.TempDir()
			if shape == "published" || shape == "all_deleted" {
				writeCodebaseFile(t, root, "repair.go", goFile(goFunction("Repair", "repairmarker")))
			}
			if shape == "whitespace" {
				writeCodebaseFile(t, root, "empty.go", " \t\n")
			}
			codebaseDaemon.index(t, root)
			if shape == "all_deleted" {
				if err := os.Remove(filepath.Join(root, "repair.go")); err != nil {
					t.Fatal(err)
				}
				codebaseDaemon.sync(t, root)
			}
			before := readCrossNamespaceOwners(t, codebaseDaemon, root)
			if shape == "published" && len(before) != 1 {
				t.Fatalf("published source has %d owners, want 1", len(before))
			}
			if shape != "published" && len(before) != 0 {
				t.Fatalf("empty source has %d owners", len(before))
			}
			index, err := codebaseDaemon.client.GetIndex(t.Context(), &pb.GetIndexRequest{Path: root})
			if err != nil {
				t.Fatal(err)
			}
			codebaseID := index.GetCodebase().GetId()
			jobs := codebaseRepairJobCount(t, codebaseDaemon, codebaseID)
			baseline := codebaseRepairCounters(t, codebaseDaemon)
			cfg := codebaseDaemon.config
			cfg.BackgroundSyncEnabled = true
			cfg.SyncIntervalMS = 1000
			ctx, cancel := context.WithCancel(codebaseDaemon.harness.context())
			t.Cleanup(cancel)
			background := daemon.NewBackgroundSync(cfg, codebaseDaemon.manager)
			background.Start(ctx)
			waitCodebaseRepairSweeps(t, codebaseDaemon, root, codebaseID, jobs, baseline["sweep_runs_total"]+2)
			cancel()
			if count := codebaseRepairCounters(t, codebaseDaemon)["embed_batches_total"]; count != baseline["embed_batches_total"] {
				t.Fatalf("automatic repair submitted %d new embedding batches", count-baseline["embed_batches_total"])
			}
			codebaseDaemon.stop()
			codebaseDaemon.config = cfg
			child := startCodebaseRestartChild(t, codebaseDaemon)
			waitCodebaseRepairSweeps(t, codebaseDaemon, root, codebaseID, jobs, 2)
			codebaseDaemon.sync(t, root)
			if count := codebaseRepairCounters(t, codebaseDaemon)["embed_batches_total"]; count != 0 {
				t.Fatalf("restart and unchanged sync submitted %d embedding batches", count)
			}
			requireCrossNamespaceOwnersEqual(t, before, readCrossNamespaceOwners(t, codebaseDaemon, root))
			killCodebaseRestartChild(t, child)
			t.Log("automatic repair and restart preserved public status and published rows; unchanged sync submitted zero embedding batches")
		})
	}
}

func codebaseRepairCounters(t *testing.T, codebaseDaemon *libraryCodebaseDaemon) map[string]int64 {
	t.Helper()
	response, err := codebaseDaemon.client.GetStatus(t.Context(), &pb.GetStatusRequest{})
	if err != nil {
		t.Fatal(err)
	}
	counters := make(map[string]int64)
	for _, metric := range response.GetMetrics() {
		if metric.GetName() == "embed_batches_total" || metric.GetName() == "sweep_runs_total" {
			value, ok := metric.GetValue().(*pb.Metric_IntValue)
			if !ok {
				t.Fatalf("counter %s has no integer value", metric.GetName())
			}
			counters[metric.GetName()] = value.IntValue
		}
	}
	if len(counters) != 2 {
		t.Fatal("public status omitted repair counters")
	}
	return counters
}

func codebaseRepairJobCount(t *testing.T, codebaseDaemon *libraryCodebaseDaemon, codebaseID string) int {
	t.Helper()
	response, err := codebaseDaemon.client.ListJobs(t.Context(), &pb.ListJobsRequest{CodebaseId: codebaseID})
	if err != nil {
		t.Fatal(err)
	}
	return len(response.GetJobs())
}

func waitCodebaseRepairSweeps(t *testing.T, codebaseDaemon *libraryCodebaseDaemon, root, codebaseID string, jobs int, sweeps int64) {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		response, err := codebaseDaemon.client.GetIndex(t.Context(), &pb.GetIndexRequest{Path: root})
		if err != nil {
			t.Fatal(err)
		}
		if response.GetCodebase().GetStatus() != "indexed" || response.GetCodebase().GetActiveJobId() != "" {
			t.Fatalf("automatic repair changed public status: %s", response.GetDisplayText())
		}
		if actual := codebaseRepairJobCount(t, codebaseDaemon, codebaseID); actual != jobs {
			t.Fatalf("automatic repair added jobs: got %d, want %d", actual, jobs)
		}
		if codebaseRepairCounters(t, codebaseDaemon)["sweep_runs_total"] >= sweeps {
			return
		}
		time.Sleep(codebaseJobPoll)
	}
	t.Fatal("the automatic repair loop did not complete two sweeps")
}
