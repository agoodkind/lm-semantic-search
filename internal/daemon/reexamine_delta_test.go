package daemon

import (
	"context"
	"testing"

	"goodkind.io/lm-semantic-search/internal/indexer"
	"goodkind.io/lm-semantic-search/internal/merkle"
	"goodkind.io/lm-semantic-search/internal/model"
	"goodkind.io/lm-semantic-search/internal/semantic"
)

// TestApplyDeltaChangesRespectsForcedItems proves the hash-equality skip in
// applyDeltaChanges is bypassed for a forced item (so an operator backfill of an
// unchanged conversation actually re-runs indexOne) and stays in force for an
// unforced item whose fingerprint is unchanged (so the normal sync is untouched).
func TestApplyDeltaChangesRespectsForcedItems(t *testing.T) {
	cases := []struct {
		name        string
		forced      bool
		wantReindex bool
	}{
		{name: "forced item re-examined despite matching hash", forced: true, wantReindex: true},
		{name: "unforced item with matching hash is skipped", forced: false, wantReindex: false},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			manager, cfg, _ := newTestManager(t)
			fake := &fakeSemantic{}
			manager.semantic = fake

			source := oneFileResultOverrideSource{
				result: indexer.OneFileResult{
					Chunks:          []model.StoredChunk{{RelativePath: "convtool/conv-x/0/tok", Content: "grep /etc"}},
					FileHash:        "fp",
					Skipped:         false,
					SkipReason:      indexer.SkipNone,
					Removed:         false,
					RemovalOverride: false,
					RemovalPaths:    nil,
					RemovalPrefixes: nil,
					ReuseVectors:    nil,
				},
				fallbackRemoval: semantic.Removal{},
				reuse:           itemReuseSource{CollectionName: "", RelativePath: "", Scope: itemReuseScopeNone},
			}
			state := overrideDeltaState(cfg.MerkleDir, source)
			state.plan.diff = merkle.Diff{Added: nil, Modified: []string{"conv-x"}, Removed: nil}
			state.plan.seedSnapshot = merkle.Snapshot{ConfigDigest: "", Files: map[string]string{"conv-x": "fp"}, Inodes: nil}
			state.plan.currentSnapshot = merkle.Snapshot{ConfigDigest: "", Files: map[string]string{"conv-x": "fp"}, Inodes: nil}
			state.working = map[string]string{"conv-x": "fp"}
			if testCase.forced {
				state.forced = map[string]struct{}{"conv-x": {}}
			}

			_, outcome := manager.applyDeltaChanges(context.Background(), model.Job{ID: "job-forced"}, state)
			if outcome.fallback || outcome.handled {
				t.Fatalf("outcome = %+v, want normal completion", outcome)
			}
			reindexed := len(fake.reindexCallsSnapshot()) > 0
			if reindexed != testCase.wantReindex {
				t.Fatalf("reindex called = %v, want %v", reindexed, testCase.wantReindex)
			}
		})
	}
}
