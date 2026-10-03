package semantic

import (
	"reflect"
	"testing"

	"goodkind.io/lm-semantic-search/collection"
	milvusstore "goodkind.io/lm-semantic-search/collection/milvus"
)

func candidate(primaryKey string, relativePath string, conversationID string, score float64) milvusstore.Candidate {
	return milvusstore.Candidate{
		PrimaryKey:   primaryKey,
		RelativePath: relativePath,
		Group:        collection.ValueCell(conversationIDFieldName, collection.StringScalar(conversationID)),
		Score:        score,
	}
}

func candidateKeys(candidates []milvusstore.Candidate) []string {
	keys := make([]string, 0, len(candidates))
	for _, ranked := range candidates {
		keys = append(keys, ranked.PrimaryKey)
	}
	return keys
}

// TestApplyLegacyConversationGroupsDropsDeletedRows proves a null-group row
// deleted after the ranking search leaves the candidates before the cap
// applies. The deleted row takes no empty-id cap slot from a surviving row.
func TestApplyLegacyConversationGroupsDropsDeletedRows(t *testing.T) {
	t.Parallel()

	nullGroup := func(primaryKey string, relativePath string, score float64) milvusstore.Candidate {
		return milvusstore.Candidate{PrimaryKey: primaryKey, RelativePath: relativePath, Group: collection.AbsentCell(conversationIDFieldName), Score: score}
	}
	resolved := applyLegacyConversationGroups(
		[]milvusstore.Candidate{
			nullGroup("gone", "conv/legacy-a/0", 0.9),
			nullGroup("kept", "conv/legacy-b/0", 0.8),
			candidate("current", "conv/c/0", "c", 0.7),
			nullGroup("unnamed", "conv/legacy-c/0", 0.6),
		},
		conversationIDFieldName,
		map[string]string{"kept": "legacy-b", "unnamed": ""},
	)
	if got, want := candidateKeys(resolved), []string{"kept", "current", "unnamed"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("resolved keys = %v, want %v", got, want)
	}
	if got, want := candidateKeys(milvusstore.SelectCandidates(resolved, 1, 0, 10)), []string{"kept", "current", "unnamed"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("capped keys = %v, want %v", got, want)
	}
}
