package semantic

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

func candidate(primaryKey string, relativePath string, conversationID string, score float64) rankedCandidate {
	return rankedCandidate{
		PrimaryKey:         primaryKey,
		RelativePath:       relativePath,
		ConversationID:     conversationID,
		ConversationIDNull: false,
		Score:              score,
	}
}

func candidateKeys(candidates []rankedCandidate) []string {
	keys := make([]string, 0, len(candidates))
	for _, ranked := range candidates {
		keys = append(keys, ranked.PrimaryKey)
	}
	return keys
}

// TestSortRankedCandidatesBreaksTiesByPathThenKey proves the ranking order is
// total: descending score, then ascending relativePath, then ascending primary
// key.
func TestSortRankedCandidatesBreaksTiesByPathThenKey(t *testing.T) {
	t.Parallel()

	candidates := []rankedCandidate{
		candidate("k4", "conv/b/0", "b", 0.5),
		candidate("k2", "conv/a/1", "a", 0.9),
		candidate("k3", "conv/a/1", "a", 0.9),
		candidate("k1", "conv/a/0", "a", 0.9),
		candidate("k5", "conv/c/0", "c", 0.7),
	}
	sortRankedCandidates(candidates)
	want := []string{"k1", "k2", "k3", "k5", "k4"}
	if got := candidateKeys(candidates); !reflect.DeepEqual(got, want) {
		t.Fatalf("sorted keys = %v, want %v", got, want)
	}
}

// TestSelectRankedCandidatesFillsPastAnOverfilledTop proves the walk fills the
// limit when the top ranks belong to one conversation, keeping at most the cap
// from it and filling the rest from lower ranks.
func TestSelectRankedCandidatesFillsPastAnOverfilledTop(t *testing.T) {
	t.Parallel()

	candidates := make([]rankedCandidate, 0, 40)
	for index := range 30 {
		candidates = append(candidates, candidate(fmt.Sprintf("dense-%02d", index), fmt.Sprintf("conv/dense/%02d", index), "dense", 1.0))
	}
	for index := range 10 {
		conversationID := fmt.Sprintf("other-%02d", index)
		candidates = append(candidates, candidate(conversationID, "conv/"+conversationID+"/0", conversationID, 0.5-float64(index)/100))
	}
	sortRankedCandidates(candidates)
	selected := selectRankedCandidates(candidates, 2, 0, 10)
	if len(selected) != 10 {
		t.Fatalf("selected %d candidates, want 10", len(selected))
	}
	dense := 0
	for _, ranked := range selected {
		if ranked.ConversationID == "dense" {
			dense++
		}
	}
	if dense != 2 {
		t.Fatalf("selected %d dense candidates, want the cap of 2", dense)
	}
}

// TestSelectRankedCandidatesSmallerLimitIsPrefix proves a smaller limit selects
// a prefix of a larger limit's selection under the same cap and floor.
func TestSelectRankedCandidatesSmallerLimitIsPrefix(t *testing.T) {
	t.Parallel()

	candidates := []rankedCandidate{
		candidate("k1", "conv/a/0", "a", 0.9),
		candidate("k2", "conv/a/1", "a", 0.8),
		candidate("k3", "conv/a/2", "a", 0.7),
		candidate("k4", "conv/b/0", "b", 0.6),
		candidate("k5", "conv/c/0", "c", 0.5),
		candidate("k6", "conv/b/1", "b", 0.4),
	}
	sortRankedCandidates(candidates)
	larger := candidateKeys(selectRankedCandidates(candidates, 2, 0.45, 5))
	for limit := int32(1); limit <= 5; limit++ {
		smaller := candidateKeys(selectRankedCandidates(candidates, 2, 0.45, limit))
		if len(smaller) > len(larger) || !reflect.DeepEqual(smaller, larger[:len(smaller)]) {
			t.Fatalf("limit %d selected %v, want a prefix of %v", limit, smaller, larger)
		}
	}
	if want := []string{"k1", "k2", "k4", "k5"}; !reflect.DeepEqual(larger, want) {
		t.Fatalf("selected %v, want %v (cap 2 drops k3, floor 0.45 drops k6)", larger, want)
	}
}

// TestSelectRankedCandidatesCapsEachConversationID proves the cap counts each
// conversation id separately, including the empty id of a row with no
// conversation identity.
func TestSelectRankedCandidatesCapsEachConversationID(t *testing.T) {
	t.Parallel()

	candidates := []rankedCandidate{
		candidate("k1", "conv/x/0", "x", 0.9),
		candidate("k2", "conv/y/0", "y", 0.8),
		candidate("k3", "conv/x/1", "x", 0.7),
		candidate("k4", "code/a", "", 0.6),
		candidate("k5", "code/b", "", 0.5),
	}
	if got, want := candidateKeys(selectRankedCandidates(candidates, 1, 0, 10)), []string{"k1", "k2", "k4"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("selected %v, want %v", got, want)
	}
}

// TestRankedCandidatesRejectMissingScores proves a ranking result with fewer
// scores than rows fails instead of ranking the unscored rows at zero.
func TestRankedCandidatesRejectMissingScores(t *testing.T) {
	t.Parallel()

	resultSet := milvusclient.ResultSet{
		ResultCount: 2,
		IDs:         column.NewColumnVarChar(idFieldName, []string{"k1", "k2"}),
		Fields: milvusclient.DataSet{
			column.NewColumnVarChar(relativePathFieldName, []string{"conv/a/0", "conv/a/1"}),
			column.NewColumnVarChar(conversationIDFieldName, []string{"a", "a"}),
		},
		Scores: []float32{0.9},
	}
	_, err := rankedCandidatesFromResultSets(context.Background(), "conv_chunks_test", []milvusclient.ResultSet{resultSet})
	if !errors.Is(err, ErrSearchResultIncomplete) {
		t.Fatalf("rankedCandidatesFromResultSets error = %v, want ErrSearchResultIncomplete", err)
	}
}
