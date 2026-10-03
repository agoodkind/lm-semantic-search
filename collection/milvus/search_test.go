package milvus

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/collection"
)

const (
	testGroupColumn         = "conversationId"
	testWorkspaceRootColumn = "workspaceRoot"
	testArchivedColumn      = "archived"
	testMessageIndexColumn  = "messageIndex"
	testLoadRulesColumn     = "loadRules"
)

func candidate(primaryKey string, relativePath string, conversationID string, score float64) Candidate {
	return Candidate{
		PrimaryKey:   primaryKey,
		RelativePath: relativePath,
		Group:        collection.ValueCell(testGroupColumn, collection.StringScalar(conversationID)),
		Score:        score,
	}
}

func candidateKeys(candidates []Candidate) []string {
	keys := make([]string, 0, len(candidates))
	for _, ranked := range candidates {
		keys = append(keys, ranked.PrimaryKey)
	}
	return keys
}

// TestRankedCandidatesRejectMissingScores proves rankedCandidatesFromResultSets
// returns ErrSearchResultIncomplete when the result set has fewer scores than
// rows.
func TestRankedCandidatesRejectMissingScores(t *testing.T) {
	t.Parallel()

	resultSet := milvusclient.ResultSet{
		ResultCount: 2,
		IDs:         column.NewColumnVarChar(IDField, []string{"k1", "k2"}),
		Fields: milvusclient.DataSet{
			column.NewColumnVarChar(RelativePathField, []string{"conv/a/0", "conv/a/1"}),
		},
		Scores: []float32{0.9},
	}
	groupColumn := collection.ScalarColumn{Name: testGroupColumn, Type: collection.ScalarTypeString, Nullable: true, MaxLength: 256}
	_, err := rankedCandidatesFromResultSets(context.Background(), "conv_chunks_test", []milvusclient.ResultSet{resultSet}, groupColumn, false)
	if !errors.Is(err, collection.ErrSearchResultIncomplete) {
		t.Fatalf("rankedCandidatesFromResultSets error = %v, want ErrSearchResultIncomplete", err)
	}
}

// TestSortRankedCandidatesBreaksTiesByPathThenKey proves the ranking order is
// total: descending score, then ascending relativePath, then ascending primary
// key.
func TestSortRankedCandidatesBreaksTiesByPathThenKey(t *testing.T) {
	t.Parallel()

	candidates := []Candidate{
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
// limit when the top ranks belong to one group. It keeps the cap from that
// group and fills the rest from lower ranks.
func TestSelectRankedCandidatesFillsPastAnOverfilledTop(t *testing.T) {
	t.Parallel()

	candidates := make([]Candidate, 0, 40)
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
		if ranked.Group.Value.String == "dense" {
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

	candidates := []Candidate{
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

// TestSelectRankedCandidatesGroupsByCellValue proves the cap counts each
// group value separately, including the empty string, and puts null and absent
// group cells in one shared group.
func TestSelectRankedCandidatesGroupsByCellValue(t *testing.T) {
	t.Parallel()

	nullCandidate := func(primaryKey string, score float64, cell collection.ScalarCell) Candidate {
		return Candidate{PrimaryKey: primaryKey, RelativePath: "row/" + primaryKey, Group: cell, Score: score}
	}
	candidates := []Candidate{
		candidate("k1", "conv/x/0", "x", 0.9),
		candidate("k2", "conv/x/1", "x", 0.8),
		candidate("k3", "code/a", "", 0.7),
		candidate("k4", "code/b", "", 0.6),
		nullCandidate("k5", 0.5, collection.NullCell(testGroupColumn)),
		nullCandidate("k6", 0.4, collection.AbsentCell(testGroupColumn)),
	}
	if got, want := candidateKeys(selectRankedCandidates(candidates, 1, 0, 10)), []string{"k1", "k3", "k5"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("selected %v, want %v", got, want)
	}
}

// TestScalarCellsAtDecodesDeclaredScalars proves each row decodes every
// declared scalar column in declaration order. A null value decodes as null,
// a column the result set lacks decodes as absent, and concrete string, bool,
// and int64 values decode with their types.
func TestScalarCellsAtDecodesDeclaredScalars(t *testing.T) {
	t.Parallel()

	workspaceRoots, err := column.NewNullableColumnVarChar(
		testWorkspaceRootColumn,
		[]string{"", "/work/alpha"},
		[]bool{false, true},
		column.WithSparseNullableMode[string](true),
	)
	if err != nil {
		t.Fatalf("build nullable workspace column: %v", err)
	}
	resultSet := milvusclient.ResultSet{
		ResultCount: 2,
		Fields: milvusclient.DataSet{
			workspaceRoots,
			column.NewColumnBool(testArchivedColumn, []bool{true, false}),
			column.NewColumnInt64(testMessageIndexColumn, []int64{0, 1}),
		},
	}
	declared := []collection.ScalarColumn{
		{Name: testWorkspaceRootColumn, Type: collection.ScalarTypeString, Nullable: true, MaxLength: 1024},
		{Name: testArchivedColumn, Type: collection.ScalarTypeBool, Nullable: true, MaxLength: 0},
		{Name: testMessageIndexColumn, Type: collection.ScalarTypeInt64, Nullable: true, MaxLength: 0},
		{Name: testLoadRulesColumn, Type: collection.ScalarTypeString, Nullable: true, MaxLength: 256},
	}
	want := [][]collection.ScalarCell{
		{
			collection.NullCell(testWorkspaceRootColumn),
			collection.ValueCell(testArchivedColumn, collection.BoolScalar(true)),
			collection.ValueCell(testMessageIndexColumn, collection.Int64Scalar(0)),
			collection.AbsentCell(testLoadRulesColumn),
		},
		{
			collection.ValueCell(testWorkspaceRootColumn, collection.StringScalar("/work/alpha")),
			collection.ValueCell(testArchivedColumn, collection.BoolScalar(false)),
			collection.ValueCell(testMessageIndexColumn, collection.Int64Scalar(1)),
			collection.AbsentCell(testLoadRulesColumn),
		},
	}
	for rowIndex := range 2 {
		cells, cellErr := scalarCellsAt(resultSet, declared, rowIndex)
		if cellErr != nil {
			t.Fatalf("scalarCellsAt(%d) returned error: %v", rowIndex, cellErr)
		}
		if !reflect.DeepEqual(cells, want[rowIndex]) {
			t.Fatalf("row %d cells = %+v, want %+v", rowIndex, cells, want[rowIndex])
		}
	}
}
