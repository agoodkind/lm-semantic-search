package semantic

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"goodkind.io/lm-semantic-search/internal/model"
)

func candidate(primaryKey string, relativePath string, conversationID string, score float64) rankedCandidate {
	return rankedCandidate{
		PrimaryKey:   primaryKey,
		RelativePath: relativePath,
		Group:        ValueCell(conversationIDFieldName, StringScalar(conversationID)),
		Score:        score,
	}
}

func candidateKeys(candidates []rankedCandidate) []string {
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
		IDs:         column.NewColumnVarChar(idFieldName, []string{"k1", "k2"}),
		Fields: milvusclient.DataSet{
			column.NewColumnVarChar(relativePathFieldName, []string{"conv/a/0", "conv/a/1"}),
		},
		Scores: []float32{0.9},
	}
	groupColumn := model.ScalarColumn{Name: conversationIDFieldName, Type: model.ScalarTypeString, Nullable: true, MaxLength: 256}
	_, err := rankedCandidatesFromResultSets(context.Background(), "conv_chunks_test", []milvusclient.ResultSet{resultSet}, groupColumn, false)
	if !errors.Is(err, ErrSearchResultIncomplete) {
		t.Fatalf("rankedCandidatesFromResultSets error = %v, want ErrSearchResultIncomplete", err)
	}
}

// TestApplyLegacyConversationGroupsDropsDeletedRows proves a null-group row
// deleted after the ranking search leaves the candidates before the cap
// applies. The deleted row takes no empty-id cap slot from a surviving row.
func TestApplyLegacyConversationGroupsDropsDeletedRows(t *testing.T) {
	t.Parallel()

	nullGroup := func(primaryKey string, relativePath string, score float64) rankedCandidate {
		return rankedCandidate{PrimaryKey: primaryKey, RelativePath: relativePath, Group: AbsentCell(conversationIDFieldName), Score: score}
	}
	resolved := applyLegacyConversationGroups(
		[]rankedCandidate{
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
	if got, want := candidateKeys(selectRankedCandidates(resolved, 1, 0, 10)), []string{"kept", "current", "unnamed"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("capped keys = %v, want %v", got, want)
	}
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
// limit when the top ranks belong to one group. It keeps the cap from that
// group and fills the rest from lower ranks.
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

// TestSelectRankedCandidatesGroupsByCellValue proves the cap counts each
// group value separately, including the empty string, and puts null and absent
// group cells in one shared group.
func TestSelectRankedCandidatesGroupsByCellValue(t *testing.T) {
	t.Parallel()

	nullCandidate := func(primaryKey string, score float64, cell ScalarCell) rankedCandidate {
		return rankedCandidate{PrimaryKey: primaryKey, RelativePath: "row/" + primaryKey, Group: cell, Score: score}
	}
	candidates := []rankedCandidate{
		candidate("k1", "conv/x/0", "x", 0.9),
		candidate("k2", "conv/x/1", "x", 0.8),
		candidate("k3", "code/a", "", 0.7),
		candidate("k4", "code/b", "", 0.6),
		nullCandidate("k5", 0.5, NullCell(conversationIDFieldName)),
		nullCandidate("k6", 0.4, AbsentCell(conversationIDFieldName)),
	}
	if got, want := candidateKeys(selectRankedCandidates(candidates, 1, 0, 10)), []string{"k1", "k3", "k5"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("selected %v, want %v", got, want)
	}
}

// TestCompileCollectionFilterExprNestsBooleanNodes proves the compiler renders
// every predicate, binds every membership set as a numbered template
// parameter in tree order, parenthesizes an or group inside an and group, and
// parenthesizes a two-bound range and an and group inside an or group. Equality
// literals use the existing Milvus string escape.
func TestCompileCollectionFilterExprNestsBooleanNodes(t *testing.T) {
	t.Parallel()

	lower := int64(5)
	upper := int64(9)
	tree := AllOf(
		AnyOf(
			ColumnEquals(roleFieldName, StringScalar(`ro"le`)),
			ColumnRange(messageIndexFieldName, &lower, &upper),
			AllOf(ColumnEquals(archivedFieldName, BoolScalar(false)), ColumnIsNull(workspaceRootFieldName)),
		),
		Negate(ColumnIn(timestampUnixFieldName, []ScalarValue{Int64Scalar(1), Int64Scalar(2)})),
		ColumnIn(archivedFieldName, []ScalarValue{BoolScalar(true)}),
		ColumnIsPresent(loadRulesFieldName),
		ColumnRange(timestampUnixFieldName, nil, &upper),
	)
	got, err := compileCollectionFilterExpr(&tree)
	if err != nil {
		t.Fatalf("compile returned error: %v", err)
	}
	want := `(role == "ro\"le" or (messageIndex >= 5 and messageIndex < 9) or (archived == false and workspaceRoot IS NULL)) and not (timestampUnix in {p0}) and archived in {p1} and loadRules IS NOT NULL and timestampUnix < 9`
	if got.Expression != want {
		t.Fatalf("expression = %q, want %q", got.Expression, want)
	}
	wantParams := []filterTemplateParam{
		{Name: "p0", Type: model.ScalarTypeInt64, Strings: nil, Bools: nil, Int64s: []int64{1, 2}},
		{Name: "p1", Type: model.ScalarTypeBool, Strings: nil, Bools: []bool{true}, Int64s: nil},
	}
	if !reflect.DeepEqual(got.Params, wantParams) {
		t.Fatalf("params = %#v, want %#v", got.Params, wantParams)
	}

	empty, err := compileCollectionFilterExpr(nil)
	if err != nil || empty.Expression != "" || len(empty.Params) != 0 {
		t.Fatalf("compile(nil) = %+v, %v, want the empty expression", empty, err)
	}
	emptyGroup := AnyOf()
	if _, err := compileCollectionFilterExpr(&emptyGroup); err == nil {
		t.Fatal("compile of an empty any group succeeded, want an error")
	}
}

// TestScalarCellsAtDecodesDeclaredScalars proves each row decodes every
// declared scalar column in declaration order. A null value decodes as null,
// a column the result set lacks decodes as absent, and concrete string, bool,
// and int64 values decode with their types.
func TestScalarCellsAtDecodesDeclaredScalars(t *testing.T) {
	t.Parallel()

	workspaceRoots, err := column.NewNullableColumnVarChar(
		workspaceRootFieldName,
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
			column.NewColumnBool(archivedFieldName, []bool{true, false}),
			column.NewColumnInt64(messageIndexFieldName, []int64{0, 1}),
		},
	}
	declared := []model.ScalarColumn{
		{Name: workspaceRootFieldName, Type: model.ScalarTypeString, Nullable: true, MaxLength: conversationWorkspaceMaxLength},
		{Name: archivedFieldName, Type: model.ScalarTypeBool, Nullable: true, MaxLength: 0},
		{Name: messageIndexFieldName, Type: model.ScalarTypeInt64, Nullable: true, MaxLength: 0},
		{Name: loadRulesFieldName, Type: model.ScalarTypeString, Nullable: true, MaxLength: conversationLoadRulesMaxLength},
	}
	want := [][]ScalarCell{
		{
			NullCell(workspaceRootFieldName),
			ValueCell(archivedFieldName, BoolScalar(true)),
			ValueCell(messageIndexFieldName, Int64Scalar(0)),
			AbsentCell(loadRulesFieldName),
		},
		{
			ValueCell(workspaceRootFieldName, StringScalar("/work/alpha")),
			ValueCell(archivedFieldName, BoolScalar(false)),
			ValueCell(messageIndexFieldName, Int64Scalar(1)),
			AbsentCell(loadRulesFieldName),
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
