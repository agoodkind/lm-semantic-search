package semantic

import (
	"reflect"
	"testing"
)

func TestConversationFilterBuildExpr(t *testing.T) {
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

	got := filter.buildExpr()
	want := `provider in ["claude", "codex"] and workspaceRoot in ["/work/alpha"] and role in ["assistant", "user"] and conversationId in {conversation_ids} and parentConversationId == "claude:root" and timestampUnix >= 100 and timestampUnix < 200 and messageIndex >= 2 and messageIndex < 9`
	if got != want {
		t.Fatalf("buildExpr() = %q, want %q", got, want)
	}
}

func TestConversationFilterBuildExprEscapesStrings(t *testing.T) {
	t.Parallel()

	filter := ConversationFilter{
		Providers:            []string{`cla"ude`},
		ConversationIDs:      []string{`thread\one`},
		ParentConversationID: `parent"root`,
	}

	got := filter.buildExpr()
	want := `provider in ["cla\"ude"] and conversationId in {conversation_ids} and parentConversationId == "parent\"root"`
	if got != want {
		t.Fatalf("buildExpr() = %q, want %q", got, want)
	}
}

func TestConversationFilterBuildExprArchived(t *testing.T) {
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

			got := ConversationFilter{Archived: test.archived}.buildExpr()
			if got != test.want {
				t.Fatalf("buildExpr() = %q, want %q", got, test.want)
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
