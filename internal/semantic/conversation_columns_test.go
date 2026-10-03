package semantic

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

func TestConversationScalarColumnsAppendCarriesArchived(t *testing.T) {
	t.Parallel()
	columns := newConversationScalarColumns(true, 2)

	if columns.archiveds == nil {
		t.Fatal("archiveds is nil, want initialized slice")
	}
	if len(columns.archiveds) != 0 {
		t.Fatalf("archiveds length = %d, want 0", len(columns.archiveds))
	}
	if cap(columns.archiveds) != 2 {
		t.Fatalf("archiveds capacity = %d, want 2", cap(columns.archiveds))
	}

	columns.append(model.StoredChunk{ConversationID: "claude:one", Archived: true})
	columns.append(model.StoredChunk{ConversationID: "codex:two", Archived: false})

	if len(columns.archiveds) != 2 {
		t.Fatalf("archiveds length = %d, want 2", len(columns.archiveds))
	}
	if !columns.archiveds[0] {
		t.Fatal("archiveds[0] = false, want true")
	}
	if columns.archiveds[1] {
		t.Fatal("archiveds[1] = true, want false")
	}
}

func TestConversationScalarColumnsDisabledLeavesArchivedNil(t *testing.T) {
	t.Parallel()
	columns := newConversationScalarColumns(false, 2)

	if columns.archiveds != nil {
		t.Fatalf("archiveds = %v, want nil", columns.archiveds)
	}

	columns.append(model.StoredChunk{ConversationID: "claude:one", Archived: true})

	if columns.archiveds != nil {
		t.Fatalf("archiveds after append = %v, want nil", columns.archiveds)
	}
}
