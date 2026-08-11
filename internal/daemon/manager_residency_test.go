package daemon

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/semantic"
)

type residencySnapshotSemantic struct {
	semanticIndex
	snapshot semantic.ResidencySnapshot
}

func (backend *residencySnapshotSemantic) ResidencySnapshot() semantic.ResidencySnapshot {
	return backend.snapshot
}

func TestManagerResidencySnapshotUsesMilvusBackend(t *testing.T) {
	manager, _, _ := newTestManager(t)
	want := semantic.ResidencySnapshot{
		IdleTimeoutMS: 900_000,
		Collections: []semantic.ResidencyCollectionSnapshot{
			{Collection: "collection", State: "ready", TimerArmed: true},
		},
	}
	manager.semantic = &residencySnapshotSemantic{
		semanticIndex: manager.semantic,
		snapshot:      want,
	}

	got := manager.ResidencySnapshot()
	if got.IdleTimeoutMS != want.IdleTimeoutMS || len(got.Collections) != 1 ||
		got.Collections[0] != want.Collections[0] {
		t.Fatalf("ResidencySnapshot = %+v, want %+v", got, want)
	}
}
