package semantic

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"goodkind.io/lm-semantic-search/internal/model"
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
	want := `provider in ["claude", "codex"] and workspaceRoot in ["/work/alpha"] and role in ["assistant", "user"] and conversationId in ["claude:thread-a", "codex:thread-b"] and parentConversationId == "claude:root" and timestampUnix >= 100 and timestampUnix < 200 and messageIndex >= 2 and messageIndex < 9`
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
	want := `provider in ["cla\"ude"] and conversationId in ["thread\\one"] and parentConversationId == "parent\"root"`
	if got != want {
		t.Fatalf("buildExpr() = %q, want %q", got, want)
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

func TestSearchConversationBatchedWithMergesBatchesByScore(t *testing.T) {
	t.Parallel()

	type searchCall struct {
		collectionName string
		query          string
		limit          int32
		expr           string
	}

	calls := make([]searchCall, 0, 2)
	search := func(ctx context.Context, collectionName string, query string, limit int32, expr string) ([]model.StoredChunk, error) {
		if ctx == nil {
			t.Fatal("search received nil context")
		}
		calls = append(calls, searchCall{
			collectionName: collectionName,
			query:          query,
			limit:          limit,
			expr:           expr,
		})
		switch expr {
		case `provider in ["claude"] and conversationId in ["a", "b"]`:
			return []model.StoredChunk{
				{Content: "first-batch-low", Score: 0.20},
				{Content: "first-batch-high", Score: 0.70},
			}, nil
		case `provider in ["claude"] and conversationId in ["c"]`:
			return []model.StoredChunk{
				{Content: "second-batch-high", Score: 0.95},
				{Content: "second-batch-mid", Score: 0.50},
			}, nil
		default:
			t.Fatalf("unexpected expression %q", expr)
			return nil, nil
		}
	}

	filter := ConversationFilter{
		Providers:       []string{"claude"},
		ConversationIDs: []string{"a", "b", "c"},
	}
	chunks, err := searchConversationBatchedWith(context.Background(), "conv_chunks_test", "needle", 2, filter, 2, search)
	if err != nil {
		t.Fatalf("searchConversationBatchedWith returned error: %v", err)
	}

	wantCalls := []searchCall{
		{
			collectionName: "conv_chunks_test",
			query:          "needle",
			limit:          2,
			expr:           `provider in ["claude"] and conversationId in ["a", "b"]`,
		},
		{
			collectionName: "conv_chunks_test",
			query:          "needle",
			limit:          2,
			expr:           `provider in ["claude"] and conversationId in ["c"]`,
		},
	}
	if !reflect.DeepEqual(calls, wantCalls) {
		t.Fatalf("search calls = %#v, want %#v", calls, wantCalls)
	}

	gotContents := []string{chunks[0].Content, chunks[1].Content}
	wantContents := []string{"second-batch-high", "first-batch-high"}
	if !reflect.DeepEqual(gotContents, wantContents) {
		t.Fatalf("merged contents = %#v, want %#v", gotContents, wantContents)
	}
}

func TestSearchConversationBatchedWithReturnsBatchError(t *testing.T) {
	t.Parallel()

	wantErr := errors.New("search failed")
	search := func(_ context.Context, _ string, _ string, _ int32, expr string) ([]model.StoredChunk, error) {
		if expr == `conversationId in ["b"]` {
			return nil, wantErr
		}
		return []model.StoredChunk{{Content: "a", Score: 0.5}}, nil
	}

	filter := ConversationFilter{ConversationIDs: []string{"a", "b"}}
	_, err := searchConversationBatchedWith(context.Background(), "conv_chunks_test", "needle", 10, filter, 1, search)
	if !errors.Is(err, wantErr) {
		t.Fatalf("searchConversationBatchedWith error = %v, want %v", err, wantErr)
	}
}
