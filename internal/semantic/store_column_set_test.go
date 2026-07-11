package semantic

import (
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

// TestStoreColumnSetRoutesWithoutNamePrefix proves the store write decides its
// column family from the caller-supplied StoreColumnSet, not from a collection
// name prefix. A code column set writes only base columns; a conversation column
// set enables the scalar columns. This is the seam insertBatch uses instead of
// isConversationCollection.
func TestStoreColumnSetRoutesWithoutNamePrefix(t *testing.T) {
	t.Parallel()

	if StoreColumnSetCode.ConversationScalars() {
		t.Fatal("StoreColumnSetCode.ConversationScalars() = true, want false")
	}
	if !StoreColumnSetConversation.ConversationScalars() {
		t.Fatal("StoreColumnSetConversation.ConversationScalars() = false, want true")
	}

	codeColumns := newConversationScalarColumns(StoreColumnSetCode.ConversationScalars(), 1)
	codeColumns.append(model.StoredChunk{ConversationID: "claude:one"})
	if codeColumns.conversationIDs != nil {
		t.Fatalf("code column set wrote conversation scalars = %v, want none", codeColumns.conversationIDs)
	}

	conversationColumns := newConversationScalarColumns(StoreColumnSetConversation.ConversationScalars(), 1)
	conversationColumns.append(model.StoredChunk{ConversationID: "claude:one"})
	if len(conversationColumns.conversationIDs) != 1 {
		t.Fatalf("conversation column set conversationIDs = %v, want one entry", conversationColumns.conversationIDs)
	}
}

// TestStoreColumnSetForCollectionClassifiesByName proves the fallback classifier
// (used only by the in-place row rewrite that has no item source) maps a
// conversation collection to the conversation column set and any other name to
// the code column set.
func TestStoreColumnSetForCollectionClassifiesByName(t *testing.T) {
	t.Parallel()

	if got := storeColumnSetForCollection(conversationCollectionPrefix + "abc"); got != StoreColumnSetConversation {
		t.Fatalf("storeColumnSetForCollection(conversation) = %v, want StoreColumnSetConversation", got)
	}
	if got := storeColumnSetForCollection("code_chunks_abc"); got != StoreColumnSetCode {
		t.Fatalf("storeColumnSetForCollection(code) = %v, want StoreColumnSetCode", got)
	}
}
