package daemon

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

func TestConversationDerivedMarkersRoundTripAndLegacy(t *testing.T) {
	t.Parallel()

	markerPath := filepath.Join(t.TempDir(), "conversation.json.derived")
	if got := loadConversationDerivedMarkers(markerPath); len(got) != 0 {
		t.Fatalf("missing marker file loaded %v, want empty", got)
	}

	want := map[string]string{
		"claude:a": derivedPipelineVersion,
		"codex:b":  "older-version",
	}
	if err := writeConversationDerivedMarkers(markerPath, want); err != nil {
		t.Fatalf("writeConversationDerivedMarkers returned error: %v", err)
	}
	got := loadConversationDerivedMarkers(markerPath)
	if len(got) != len(want) || got["claude:a"] != want["claude:a"] || got["codex:b"] != want["codex:b"] {
		t.Fatalf("round-trip markers = %v, want %v", got, want)
	}

	legacyData := []byte(`{"config_digest":"legacy","files":{"claude:a":"fp"}}`)
	if err := os.WriteFile(markerPath, legacyData, 0o644); err != nil {
		t.Fatalf("write legacy marker fixture: %v", err)
	}
	if got := loadConversationDerivedMarkers(markerPath); len(got) != 0 {
		t.Fatalf("legacy marker file loaded %v, want empty", got)
	}
}

func TestCurrentConversationMarkerSkipsStateLoad(t *testing.T) {
	t.Parallel()

	reader := &testConversationRowReader{}
	documents := []model.ConversationDocument{{
		ConversationID: "claude:current",
		MessageIndex:   0,
		Role:           "user",
		Text:           "unchanged",
	}}
	source := newConversationItemSource(
		"conversation_collection",
		map[string]string{"claude:current": "fp-current"},
		documents,
		reader,
		absenceRetain,
		true,
	)
	source.derivedVersions = map[string]string{"claude:current": derivedPipelineVersion}
	captured, err := source.capture(context.Background())
	if err != nil {
		t.Fatalf("capture returned error: %v", err)
	}
	diff := unionForcedItems(merkle.Diff{}, source.forcedItems(), captured)
	if !diff.Empty() {
		t.Fatalf("current marker produced changed diff: %+v", diff)
	}
	if calls := reader.callsSnapshot(); len(calls) != 0 {
		t.Fatalf("message state loads = %v, want none", calls)
	}
}

func TestConversationMarkerWrittenOnlyAfterSemanticSuccess(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name             string
		deliverDocuments bool
		reindexErr       error
		wantMarker       bool
	}{
		{name: "semantic success", deliverDocuments: true, reindexErr: nil, wantMarker: true},
		{name: "semantic failure", deliverDocuments: true, reindexErr: errors.New("semantic write failed"), wantMarker: false},
		{name: "pending delivery", deliverDocuments: false, reindexErr: nil, wantMarker: false},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			manager, cfg, _ := newTestManager(t)
			manager.semantic = &fakeSemantic{
				reindexWithReuse: func(context.Context, string, []model.StoredChunk, []string, func(semantic.Progress), map[string][]float32) error {
					return testCase.reindexErr
				},
			}
			var documents []model.ConversationDocument
			if testCase.deliverDocuments {
				documents = []model.ConversationDocument{{ConversationID: "claude:result", MessageIndex: 0, Role: "user", Text: "result"}}
			}
			source := newConversationItemSource(
				"conversation_collection",
				map[string]string{"claude:result": "fp-result"},
				documents,
				nil,
				absenceRetain,
				true,
			)
			snapshotPath := filepath.Join(cfg.MerkleDir, "marker-success.json")
			state := deltaState{
				plan: deltaPlan{
					diff:            merkle.Diff{Added: []string{"claude:result"}},
					currentSnapshot: merkle.Snapshot{Files: map[string]string{"claude:result": "fp-result"}},
				},
				snapshotPath: snapshotPath,
				working:      map[string]string{},
				source:       source,
				semantic:     true,
				chunkCounts:  &chunkCounters{},
				forced:       forcedItemsSet(source),
			}

			_, outcome := manager.applyDeltaChanges(context.Background(), model.Job{ID: "job-marker"}, state)
			if testCase.reindexErr == nil && (outcome.fallback || outcome.handled) {
				t.Fatalf("successful outcome = %+v, want normal completion", outcome)
			}
			if testCase.reindexErr != nil && !outcome.handled {
				t.Fatalf("failed outcome = %+v, want handled", outcome)
			}
			markers := loadConversationDerivedMarkers(conversationDerivedMarkerPath(snapshotPath))
			_, marked := markers["claude:result"]
			if marked != testCase.wantMarker {
				t.Fatalf("marker present = %v, want %v; markers = %v", marked, testCase.wantMarker, markers)
			}
		})
	}
}
