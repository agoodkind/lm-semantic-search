package collection_test

import (
	"reflect"
	"testing"

	"goodkind.io/lm-semantic-search/collection"
)

// TestCompileNestsBooleanNodes proves the compiler renders every predicate and
// binds every membership set as a numbered template parameter in tree order. An
// or group inside an and group gets parentheses. A two-bound range and an and
// group inside an or group get parentheses. Equality literals use the existing
// Milvus string escape.
func TestCompileNestsBooleanNodes(t *testing.T) {
	t.Parallel()

	lower := int64(5)
	upper := int64(9)
	tree := collection.AllOf(
		collection.AnyOf(
			collection.ColumnEquals("role", collection.StringScalar(`ro"le`)),
			collection.ColumnRange("messageIndex", &lower, &upper),
			collection.AllOf(collection.ColumnEquals("archived", collection.BoolScalar(false)), collection.ColumnIsNull("workspaceRoot")),
		),
		collection.Negate(collection.ColumnIn("timestampUnix", []collection.ScalarValue{collection.Int64Scalar(1), collection.Int64Scalar(2)})),
		collection.ColumnIn("archived", []collection.ScalarValue{collection.BoolScalar(true)}),
		collection.ColumnIsPresent("loadRules"),
		collection.ColumnRange("timestampUnix", nil, &upper),
	)
	got, err := collection.Compile(&tree)
	if err != nil {
		t.Fatalf("compile returned error: %v", err)
	}
	want := `(role == "ro\"le" or (messageIndex >= 5 and messageIndex < 9) or (archived == false and workspaceRoot IS NULL)) and not (timestampUnix in {p0}) and archived in {p1} and loadRules IS NOT NULL and timestampUnix < 9`
	if got.Expression != want {
		t.Fatalf("expression = %q, want %q", got.Expression, want)
	}
	wantParams := []collection.TemplateParam{
		{Name: "p0", Type: collection.ScalarTypeInt64, Strings: nil, Bools: nil, Int64s: []int64{1, 2}},
		{Name: "p1", Type: collection.ScalarTypeBool, Strings: nil, Bools: []bool{true}, Int64s: nil},
	}
	if !reflect.DeepEqual(got.Params, wantParams) {
		t.Fatalf("params = %#v, want %#v", got.Params, wantParams)
	}

	empty, err := collection.Compile(nil)
	if err != nil || empty.Expression != "" || len(empty.Params) != 0 {
		t.Fatalf("compile(nil) = %+v, %v, want the empty expression", empty, err)
	}
	emptyGroup := collection.AnyOf()
	if _, err := collection.Compile(&emptyGroup); err == nil {
		t.Fatal("compile of an empty any group succeeded, want an error")
	}
}
