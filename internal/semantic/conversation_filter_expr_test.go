package semantic

import (
	"reflect"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
)

// conversationCompiled compiles the conversation filter through the
// conversation adapter and the generic expression compiler.
func conversationCompiled(t *testing.T, filter ConversationFilter) compiledFilter {
	t.Helper()
	compiled, err := compileCollectionFilterExpr(filter.CollectionFilter())
	if err != nil {
		t.Fatalf("compile conversation filter: %v", err)
	}
	return compiled
}

func conversationExpr(t *testing.T, filter ConversationFilter) string {
	t.Helper()
	return conversationCompiled(t, filter).Expression
}

func stringParam(name string, values ...string) filterTemplateParam {
	return filterTemplateParam{Name: name, Type: model.ScalarTypeString, Strings: values, Bools: nil, Int64s: nil}
}

func TestConversationFilterCompilesExpr(t *testing.T) {
	t.Parallel()

	filter := ConversationFilter{
		Providers:            []string{"claude", "codex"},
		WorkspaceRoots:       []string{"/work/alpha"},
		Roles:                []string{"Assistant", "USER"},
		ConversationIDs:      []string{"claude:thread-a", "codex:thread-b"},
		ParentConversationID: "claude:root",
		FromUnix:             100,
		UntilUnix:            200,
		MessageIndexFrom:     2,
		MessageIndexUntil:    9,
	}

	got := conversationCompiled(t, filter)
	want := `provider in {p0} and workspaceRoot in {p1} and role in {p2} and conversationId in {p3} and parentConversationId == "claude:root" and timestampUnix >= 100 and timestampUnix < 200 and messageIndex >= 2 and messageIndex < 9`
	if got.Expression != want {
		t.Fatalf("expression = %q, want %q", got.Expression, want)
	}
	wantParams := []filterTemplateParam{
		stringParam("p0", "claude", "codex"),
		stringParam("p1", "/work/alpha"),
		stringParam("p2", "assistant", "user"),
		stringParam("p3", "claude:thread-a", "codex:thread-b"),
	}
	if !reflect.DeepEqual(got.Params, wantParams) {
		t.Fatalf("params = %#v, want %#v", got.Params, wantParams)
	}
}

func TestConversationFilterCompilesExprEscapesStrings(t *testing.T) {
	t.Parallel()

	filter := ConversationFilter{
		Providers:            []string{`cla"ude`},
		ConversationIDs:      []string{`thread\one`},
		ParentConversationID: `parent"root`,
	}

	got := conversationCompiled(t, filter)
	want := `provider in {p0} and conversationId in {p1} and parentConversationId == "parent\"root"`
	if got.Expression != want {
		t.Fatalf("expression = %q, want %q", got.Expression, want)
	}
	wantParams := []filterTemplateParam{stringParam("p0", `cla"ude`), stringParam("p1", `thread\one`)}
	if !reflect.DeepEqual(got.Params, wantParams) {
		t.Fatalf("params = %#v, want %#v (template values are sent unescaped)", got.Params, wantParams)
	}
}

func TestConversationFilterCompilesExprArchived(t *testing.T) {
	t.Parallel()

	archivedFalse := false
	archivedTrue := true
	tests := []struct {
		name     string
		archived *bool
		want     string
	}{
		{name: "nil archived adds no clause", archived: nil, want: ""},
		{name: "false keeps non-archived rows", archived: &archivedFalse, want: "archived == false"},
		{name: "true keeps archived rows", archived: &archivedTrue, want: "archived == true"},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			got := conversationExpr(t, ConversationFilter{Archived: test.archived})
			if got != test.want {
				t.Fatalf("expression = %q, want %q", got, test.want)
			}
		})
	}
}

func TestBatchConversationIDs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		ids  []string
		size int
		want [][]string
	}{
		{
			name: "empty ids keep one unscoped batch",
			ids:  nil,
			size: 2,
			want: [][]string{nil},
		},
		{
			name: "size splits ids",
			ids:  []string{"a", "b", "c", "d", "e"},
			size: 2,
			want: [][]string{{"a", "b"}, {"c", "d"}, {"e"}},
		},
		{
			name: "non-positive size keeps one batch",
			ids:  []string{"a", "b", "c"},
			size: 0,
			want: [][]string{{"a", "b", "c"}},
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			got := batchConversationIDs(test.ids, test.size)
			if !reflect.DeepEqual(got, test.want) {
				t.Fatalf("batchConversationIDs() = %#v, want %#v", got, test.want)
			}
		})
	}
}
