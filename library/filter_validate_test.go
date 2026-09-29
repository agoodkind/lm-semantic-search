package library_test

import (
	"math"
	"strings"
	"testing"

	"goodkind.io/lm-semantic-search/library"
)

func workspaceValue(text string) library.ScalarValue {
	return library.ScalarValue{Type: library.String, String: text}
}

func messageIndexValue(value int64) *library.ScalarValue {
	return &library.ScalarValue{Type: library.Int64, Int64: value}
}

func validSearchRequest() library.SearchRequest {
	return library.SearchRequest{
		Namespace: conversationNamespace().ID,
		Query:     "daemon reload bind gap",
		PageSize:  10,
	}
}

// nestedFilter uses every operator on a path of three nodes, with four Values
// entries and three range bounds. The prefix contains SQL LIKE and GLOB
// metacharacters, which a literal prefix accepts as ordinary text.
func nestedFilter() library.Filter {
	return library.Filter{Op: library.All, Children: []library.Filter{
		{Op: library.Any, Children: []library.Filter{
			{Op: library.Equal, Column: "workspace", Values: []library.ScalarValue{workspaceValue("clyde")}},
			{Op: library.In, Column: "workspace", Values: []library.ScalarValue{workspaceValue("a"), workspaceValue("b")}},
			{Op: library.Prefix, Column: "workspace", Prefix: `50%_off\[*]?`},
			{Op: library.IsNull, Column: "workspace"},
		}},
		{Op: library.Not, Children: []library.Filter{
			{Op: library.Equal, Column: "archived", Values: []library.ScalarValue{{Type: library.Bool, Bool: true}}},
		}},
		{Op: library.Range, Column: "message_index", Lower: messageIndexValue(0), Upper: messageIndexValue(100)},
		{Op: library.Range, Column: "message_index", Upper: messageIndexValue(math.MinInt64 + 1)},
		{Op: library.IsPresent, Column: "archived"},
	}}
}

// filterDepth returns a chain of Not nodes over one leaf with depth nodes on
// its only path.
func filterDepth(depth int) library.Filter {
	filter := library.Filter{Op: library.IsPresent, Column: "archived"}
	for range depth - 1 {
		filter = library.Filter{Op: library.Not, Children: []library.Filter{filter}}
	}
	return filter
}

func TestValidateSearchRequestAcceptsEveryOperator(t *testing.T) {
	t.Parallel()
	request := validSearchRequest()
	filter := nestedFilter()
	request.Filter = &filter
	request.GroupBy = "workspace"
	request.PerGroupLimit = 3
	request.MinScore = 0.01
	config := library.Config{MaxPageSize: 10, MaxQueryBytes: len(request.Query), MaxFilterDepth: 3, MaxFilterValues: 7}
	if err := config.ValidateSearchRequest(conversationNamespace(), request); err != nil {
		t.Fatalf("ValidateSearchRequest(nested filter at every limit) = %v, want nil", err)
	}
	if err := (library.Config{}).ValidateSearchRequest(conversationNamespace(), validSearchRequest()); err != nil {
		t.Fatalf("ValidateSearchRequest(no filter, no limits) = %v, want nil", err)
	}
}

// ValidateSearchRequest accepts a comparison string longer than the column
// MaxLength. The filter matches no row, and the search returns an empty result.
func TestValidateSearchRequestAcceptsStringsLongerThanTheColumn(t *testing.T) {
	t.Parallel()
	long := strings.Repeat("w", 65)
	for _, testCase := range []struct {
		name   string
		filter library.Filter
	}{
		{name: "Equal", filter: library.Filter{Op: library.Equal, Column: "workspace", Values: []library.ScalarValue{workspaceValue(long)}}},
		{name: "In", filter: library.Filter{Op: library.In, Column: "workspace", Values: []library.ScalarValue{workspaceValue("a"), workspaceValue(long)}}},
		{name: "Prefix", filter: library.Filter{Op: library.Prefix, Column: "workspace", Prefix: long}},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			request := validSearchRequest()
			request.Filter = &testCase.filter
			if err := (library.Config{}).ValidateSearchRequest(conversationNamespace(), request); err != nil {
				t.Fatalf("ValidateSearchRequest(65-byte string on a 64-byte column) = %v, want nil", err)
			}
		})
	}
}

func TestValidateSearchRequestRejectsInvalidFilters(t *testing.T) {
	t.Parallel()
	nullWorkspace := library.ScalarValue{Type: library.String, Null: true}
	stringBound := library.ScalarValue{Type: library.String, String: "a"}
	for _, testCase := range []struct {
		name   string
		filter library.Filter
		want   string
	}{
		{name: "zero operator", filter: library.Filter{}, want: "operator 0 is not a FilterOp"},
		{name: "unknown operator", filter: library.Filter{Op: 99, Column: "workspace"}, want: "operator 99"},
		{name: "empty All", filter: library.Filter{Op: library.All}, want: "All has no children"},
		{name: "empty Any", filter: library.Filter{Op: library.Any}, want: "Any has no children"},
		{
			name:   "All with column",
			filter: library.Filter{Op: library.All, Column: "workspace", Children: []library.Filter{filterDepth(1)}},
			want:   "All sets Column",
		},
		{
			name:   "Not with two children",
			filter: library.Filter{Op: library.Not, Children: []library.Filter{filterDepth(1), filterDepth(1)}},
			want:   "Not has 2 children, want exactly 1",
		},
		{name: "Not without child", filter: library.Filter{Op: library.Not}, want: "Not has 0 children"},
		{name: "Equal without value", filter: library.Filter{Op: library.Equal, Column: "workspace"}, want: "Equal has 0 values"},
		{
			name: "Equal with two values",
			filter: library.Filter{
				Op: library.Equal, Column: "workspace",
				Values: []library.ScalarValue{workspaceValue("a"), workspaceValue("b")},
			},
			want: "Equal has 2 values",
		},
		{name: "In without value", filter: library.Filter{Op: library.In, Column: "workspace"}, want: "In has no values"},
		{
			name: "Equal with prefix",
			filter: library.Filter{
				Op: library.Equal, Column: "workspace", Prefix: "a",
				Values: []library.ScalarValue{workspaceValue("a")},
			},
			want: "Equal sets Prefix",
		},
		{
			name:   "undeclared column",
			filter: library.Filter{Op: library.Equal, Column: "provider", Values: []library.ScalarValue{workspaceValue("claude")}},
			want:   `column "provider" is not declared`,
		},
		{name: "leaf without column", filter: library.Filter{Op: library.IsNull}, want: "IsNull names no column"},
		{
			name:   "value type mismatch",
			filter: library.Filter{Op: library.Equal, Column: "message_index", Values: []library.ScalarValue{workspaceValue("7")}},
			want:   "does not match declared type",
		},
		{
			name:   "invalid comparison string",
			filter: library.Filter{Op: library.In, Column: "workspace", Values: []library.ScalarValue{workspaceValue("\xff")}},
			want:   "Filter.Values[0]: column \"workspace\" value is not valid UTF-8",
		},
		{
			name:   "null comparison value",
			filter: library.Filter{Op: library.Equal, Column: "workspace", Values: []library.ScalarValue{nullWorkspace}},
			want:   "compares column \"workspace\" with null; use IsNull",
		},
		{
			name:   "null range bound",
			filter: library.Filter{Op: library.Range, Column: "message_index", Lower: &library.ScalarValue{Type: library.Int64, Null: true}},
			want:   "Filter.Lower compares column \"message_index\" with null",
		},
		{name: "unbounded range", filter: library.Filter{Op: library.Range, Column: "message_index"}, want: "sets neither Lower nor Upper"},
		{
			name:   "string range",
			filter: library.Filter{Op: library.Range, Column: "workspace", Lower: &stringBound},
			want:   `Range column "workspace" is not Int64`,
		},
		{
			name:   "range with values",
			filter: library.Filter{Op: library.Range, Column: "message_index", Lower: messageIndexValue(1), Values: []library.ScalarValue{workspaceValue("a")}},
			want:   "Range sets Values",
		},
		{name: "prefix on bool", filter: library.Filter{Op: library.Prefix, Column: "archived", Prefix: "t"}, want: `Prefix column "archived" is not String`},
		{name: "empty prefix", filter: library.Filter{Op: library.Prefix, Column: "workspace"}, want: "Prefix text is empty"},
		{name: "invalid prefix", filter: library.Filter{Op: library.Prefix, Column: "workspace", Prefix: "\xff"}, want: "not valid UTF-8"},
		{name: "IsPresent with values", filter: library.Filter{Op: library.IsPresent, Column: "workspace", Values: []library.ScalarValue{workspaceValue("a")}}, want: "IsPresent sets Values"},
		{
			name: "nested violation reports its path",
			filter: library.Filter{Op: library.All, Children: []library.Filter{
				filterDepth(1),
				{Op: library.Any, Children: []library.Filter{filterDepth(1), {Op: library.IsNull, Column: "missing"}}},
			}},
			want: `Filter.Children[1].Children[1] column "missing" is not declared`,
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			request := validSearchRequest()
			request.Filter = &testCase.filter
			assertInvalidRequest(t, library.Config{}.ValidateSearchRequest(conversationNamespace(), request), testCase.want)
		})
	}
}

func TestValidateSearchRequestEnforcesFilterDepthAndValueLimits(t *testing.T) {
	t.Parallel()
	request := validSearchRequest()

	deep := filterDepth(4)
	request.Filter = &deep
	if err := (library.Config{MaxFilterDepth: 4}).ValidateSearchRequest(conversationNamespace(), request); err != nil {
		t.Fatalf("depth 4 at MaxFilterDepth 4 = %v, want nil", err)
	}
	assertInvalidRequest(t,
		library.Config{MaxFilterDepth: 3}.ValidateSearchRequest(conversationNamespace(), request),
		"at filter depth 4, over MaxFilterDepth 3",
	)
	veryDeep := filterDepth(10_000)
	request.Filter = &veryDeep
	if err := (library.Config{}).ValidateSearchRequest(conversationNamespace(), request); err != nil {
		t.Fatalf("depth 10000 with MaxFilterDepth disabled = %v, want nil", err)
	}

	nested := nestedFilter()
	request.Filter = &nested
	if err := (library.Config{MaxFilterValues: 7}).ValidateSearchRequest(conversationNamespace(), request); err != nil {
		t.Fatalf("7 values at MaxFilterValues 7 = %v, want nil", err)
	}
	assertInvalidRequest(t,
		library.Config{MaxFilterValues: 6}.ValidateSearchRequest(conversationNamespace(), request),
		"is filter value 7, over MaxFilterValues 6",
	)
	assertInvalidRequest(t,
		library.Config{MaxFilterDepth: -1}.ValidateSearchRequest(conversationNamespace(), request),
		"MaxFilterDepth is -1",
	)
}

func TestValidateSearchRequestRejectsInvalidRequestFields(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name   string
		config library.Config
		mutate func(*library.SearchRequest)
		want   string
	}{
		{name: "other namespace", mutate: func(r *library.SearchRequest) { r.Namespace = "code" }, want: `search namespace "code" does not match`},
		{name: "NUL in query", mutate: func(r *library.SearchRequest) { r.Query = "before\x00after" }, want: "NUL byte"},
		{name: "whitespace query", mutate: func(r *library.SearchRequest) { r.Query = " \t\n" }, want: "no non-whitespace"},
		{name: "invalid query", mutate: func(r *library.SearchRequest) { r.Query = "\xff" }, want: "query is not valid UTF-8"},
		{
			name:   "query over limit",
			config: library.Config{MaxQueryBytes: 4},
			mutate: func(r *library.SearchRequest) { r.Query = "alpha" },
			want:   "query is 5 bytes, over MaxQueryBytes 4",
		},
		{name: "zero page size", mutate: func(r *library.SearchRequest) { r.PageSize = 0 }, want: "PageSize 0 must be positive"},
		{
			name:   "page size over limit",
			config: library.Config{MaxPageSize: 100},
			mutate: func(r *library.SearchRequest) { r.PageSize = 101 },
			want:   "PageSize 101 is over MaxPageSize 100",
		},
		{name: "negative min score", mutate: func(r *library.SearchRequest) { r.MinScore = -0.5 }, want: "MinScore -0.5"},
		{name: "NaN min score", mutate: func(r *library.SearchRequest) { r.MinScore = math.NaN() }, want: "MinScore NaN"},
		{name: "group limit without column", mutate: func(r *library.SearchRequest) { r.PerGroupLimit = 2 }, want: "requires a GroupBy column"},
		{
			name:   "undeclared group column",
			mutate: func(r *library.SearchRequest) { r.GroupBy = "provider"; r.PerGroupLimit = 2 },
			want:   `GroupBy column "provider" is not declared`,
		},
		{name: "group without limit", mutate: func(r *library.SearchRequest) { r.GroupBy = "workspace" }, want: "PerGroupLimit 0 must be positive"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			request := validSearchRequest()
			testCase.mutate(&request)
			assertInvalidRequest(t, testCase.config.ValidateSearchRequest(conversationNamespace(), request), testCase.want)
		})
	}
	invalidSpec := conversationNamespace()
	invalidSpec.Policy = 0
	assertInvalidRequest(t, library.Config{}.ValidateSearchRequest(invalidSpec, validSearchRequest()), "policy 0")
}
