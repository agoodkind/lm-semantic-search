package semantic

import (
	"testing"

	"goodkind.io/lm-semantic-search/collection"
	"goodkind.io/lm-semantic-search/internal/model"
)

// TestStoreColumnSetRoutesWithoutNamePrefix proves the store write decides its
// column family from the caller-supplied StoreColumnSet, not from a collection
// name prefix. A code column set writes only base columns; a conversation column
// set enables the scalar columns. This is the seam insertBatch uses instead of
// isConversationCollection.
func TestStoreColumnSetRoutesWithoutNamePrefix(t *testing.T) {
	t.Parallel()

	if CodeColumns().ConversationScalars() {
		t.Fatal("StoreColumnSetCode.ConversationScalars() = true, want false")
	}
	if !ConversationColumns().ConversationScalars() {
		t.Fatal("StoreColumnSetConversation.ConversationScalars() = false, want true")
	}

	codeColumns := newConversationScalarColumns(CodeColumns().ConversationScalars(), 1)
	codeColumns.append(model.StoredChunk{ConversationID: "claude:one"})
	if codeColumns.conversationIDs != nil {
		t.Fatalf("code column set wrote conversation scalars = %v, want none", codeColumns.conversationIDs)
	}

	conversationColumns := newConversationScalarColumns(ConversationColumns().ConversationScalars(), 1)
	conversationColumns.append(model.StoredChunk{ConversationID: "claude:one"})
	if len(conversationColumns.conversationIDs) != 1 {
		t.Fatalf("conversation column set conversationIDs = %v, want one entry", conversationColumns.conversationIDs)
	}
}

// TestStoreColumnSetForCollectionClassifiesByName proves the fallback classifier
// (used only by the in-place row rewrite that has no item source) maps a
// conversation collection to the conversation column set and any other name to
// the code column set. A document collection with a recorded generic
// declaration shares the conversation name prefix and still classifies as a
// non-conversation collection, including its staging twin.
func TestStoreColumnSetForCollectionClassifiesByName(t *testing.T) {
	t.Parallel()

	service := &Service{}
	conversationName := conversationCollectionPrefix + "abc"
	if got := service.storeColumnSetForCollection(conversationName); !got.ConversationScalars() {
		t.Fatalf("storeColumnSetForCollection(conversation) = %+v, want conversation columns", got)
	}
	if got := service.storeColumnSetForCollection("code_chunks_abc"); got.ConversationScalars() {
		t.Fatalf("storeColumnSetForCollection(code) = %+v, want code columns", got)
	}

	genericName := conversationCollectionPrefix + "generic"
	service.RecordCollectionDeclaration(genericName, collection.Declaration{
		ItemIDColumn: "itemId",
		Scalars:      []collection.ScalarColumn{{Name: "itemId", Type: collection.ScalarTypeString, Nullable: false, MaxLength: 64}},
	})
	if service.isConversationCollection(genericName) {
		t.Fatal("isConversationCollection(generic) = true, want false")
	}
	if service.isConversationCollection(stagingCollectionName(genericName)) {
		t.Fatal("isConversationCollection(generic staging) = true, want false")
	}
	service.RecordCollectionDeclaration(genericName, ConversationDeclaration())
	if !service.isConversationCollection(genericName) {
		t.Fatal("isConversationCollection after recording the conversation declaration = false, want true")
	}
}
