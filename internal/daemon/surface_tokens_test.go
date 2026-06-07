package daemon

import (
	"context"
	"testing"

	pb "goodkind.io/lm-semantic-search/gen/go/lmsemanticsearch/v1"
	"goodkind.io/lm-semantic-search/internal/clock"
	"goodkind.io/lm-semantic-search/internal/model"
)

// TestListIndexesRowCarriesWaitingTokens proves a ListIndexes row carries the
// daemon-owned glyph and label tokens beside the display status, so the CLI list
// renders one vocabulary. An incomplete codebase under a hard dependency outage
// folds to "waiting", which must surface as ⋯ / "waiting".
func TestListIndexesRowCarriesWaitingTokens(t *testing.T) {
	manager, _, _ := newTestManager(t)

	manager.mu.Lock()
	manager.codebases["cb-wait"] = model.Codebase{
		ID:              "cb-wait",
		CanonicalPath:   "/tmp/cb-wait",
		Status:          model.CodebaseStatusIndexing,
		EffectiveConfig: defaultIndexConfig(),
	}
	manager.health = dependencyHealth{Mode: dependencyEmbedderUnreachable, Since: clock.Now(), LastHealthyAt: clock.Now()}
	manager.mu.Unlock()

	server := NewGRPCServer(manager, nil)
	response, err := server.ListIndexes(context.Background(), &pb.ListIndexesRequest{})
	if err != nil {
		t.Fatalf("ListIndexes returned error: %v", err)
	}

	var row *pb.Codebase
	for _, codebase := range response.GetIndexes() {
		if codebase.GetId() == "cb-wait" {
			row = codebase
			break
		}
	}
	if row == nil {
		t.Fatalf("ListIndexes did not return the cb-wait row: %+v", response.GetIndexes())
	}
	if got := row.GetDisplayStatus(); got != "waiting" {
		t.Fatalf("DisplayStatus = %q, want waiting", got)
	}
	if got := row.GetGlyphToken(); got != "⋯" {
		t.Fatalf("GlyphToken = %q, want ⋯", got)
	}
	if got := row.GetStatusLabel(); got != "waiting" {
		t.Fatalf("StatusLabel = %q, want waiting", got)
	}
}
