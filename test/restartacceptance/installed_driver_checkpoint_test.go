//go:build restartacceptance

package restartacceptance

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
)

type checkpointPathServer struct {
	pb.UnimplementedSemanticSearchDaemonServiceServer
}

func (checkpointPathServer) GetIndex(
	context.Context,
	*pb.GetIndexRequest,
) (*pb.GetIndexResponse, error) {
	return &pb.GetIndexResponse{
		Codebase: &pb.Codebase{Id: "cb-first-build"},
	}, nil
}

func TestCheckpointObserverDerivesFirstBuildStagingPath(t *testing.T) {
	merkleRoot := t.TempDir()
	stagingPath := filepath.Join(merkleRoot, "cb-first-build.staging.json")
	body, err := json.Marshal(map[string]map[string]string{
		"files": {"01.go": "hash-1"},
	})
	if err != nil {
		t.Fatalf("encode checkpoint: %v", err)
	}
	if err := os.WriteFile(stagingPath, body, 0o600); err != nil {
		t.Fatalf("write checkpoint: %v", err)
	}

	client := startScenarioFakeGRPC(t, checkpointPathServer{})
	snapshot, err := checkpointObserver(client, "/fixture", merkleRoot)(context.Background())
	if err != nil {
		t.Fatalf("observe first-build checkpoint: %v", err)
	}
	if snapshot.CompletedCount != 1 {
		t.Fatalf("completed count = %d, want 1", snapshot.CompletedCount)
	}
	if _, found := snapshot.TrackedIDs["01.go"]; !found {
		t.Fatal("first-build checkpoint omitted 01.go")
	}
}
