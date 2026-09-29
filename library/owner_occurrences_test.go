package library_test

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"testing"

	"goodkind.io/lm-semantic-search/library"
)

func noteRow(rowKey string, text string) library.Occurrence {
	return library.Occurrence{
		RowKey:         rowKey,
		SortKey:        rowKey,
		SourceText:     text,
		SearchText:     text,
		EmbeddingInput: text,
		Scalars: map[string]library.ScalarValue{
			"archived": {Type: library.Bool, Null: false, String: "", Bool: false, Int64: 0},
		},
	}
}

func codePartRow(rowKey string, text string) library.Occurrence {
	return library.Occurrence{RowKey: rowKey, SortKey: rowKey, SourceText: text, SearchText: text, EmbeddingInput: text, Scalars: nil}
}

func mustApplyBatch(t *testing.T, opened *library.Library, batch library.Batch) {
	t.Helper()
	if _, err := opened.Apply(context.Background(), batch); err != nil {
		t.Fatalf("Apply %s order %d: %v", batch.OwnerID, batch.GenerationOrder, err)
	}
}

func TestListOwnerOccurrencesReturnsPublishedRowsAndProjectionOrder(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	opened, err := library.Open(ctx, library.Config{
		Store:    embeddedDescriptor(t, filepath.Join(root, "catalog")),
		Vectors:  embeddedPool(t, filepath.Join(root, "pool")),
		Embedder: offlineEmbedder(t),
	})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() {
		if err := opened.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	notes := library.NamespaceSpec{ID: "notes", Policy: library.AppendOnly, Scalars: []library.ScalarColumn{
		{Name: "archived", Type: library.Bool, Nullable: false, Mutable: true, MaxLength: 0},
	}}
	code := library.NamespaceSpec{ID: "code", Policy: library.ReplaceAllowed, Scalars: nil}
	for _, spec := range []library.NamespaceSpec{notes, code} {
		if err := opened.RegisterNamespace(ctx, spec); err != nil {
			t.Fatalf("RegisterNamespace %s: %v", spec.ID, err)
		}
	}

	mustApplyBatch(t, opened, library.Batch{
		Namespace: "notes", OwnerID: "note-1", GenerationOrder: 1, IdempotencyToken: "n1", Mode: library.Append,
		Rows: []library.Occurrence{noteRow("n2", "second note line"), noteRow("n1", "first note line")},
	})
	mustApplyBatch(t, opened, library.Batch{
		Namespace: "notes", OwnerID: "note-1", GenerationOrder: 2, IdempotencyToken: "n2", Mode: library.Append,
		Rows: []library.Occurrence{noteRow("n3", "third note line")},
	})
	if _, err := opened.ReprojectScalars(ctx, library.ScalarProjection{
		Namespace: "notes", OwnerID: "note-1", ProjectionOrder: 1, IdempotencyToken: "p1",
		Rows: map[string]map[string]library.ScalarValue{"n1": {"archived": {Type: library.Bool, Null: false, String: "", Bool: true, Int64: 0}}},
	}); err != nil {
		t.Fatalf("ReprojectScalars: %v", err)
	}
	staged := library.GenerationKey{Namespace: "notes", OwnerID: "note-1", GenerationOrder: 3, IdempotencyToken: "n3"}
	if err := opened.Stage(ctx, library.StageBatch{Key: staged, Mode: library.Append, Rows: []library.Occurrence{noteRow("n4", "staged note line")}}); err != nil {
		t.Fatalf("Stage: %v", err)
	}

	mustApplyBatch(t, opened, library.Batch{
		Namespace: "code", OwnerID: "main.go", GenerationOrder: 1, IdempotencyToken: "c1", Mode: library.Replace,
		Rows: []library.Occurrence{codePartRow("a", "func alpha() {}"), codePartRow("b", "func beta() {}"), codePartRow("c", "func gamma() {}")},
	})
	mustApplyBatch(t, opened, library.Batch{
		Namespace: "code", OwnerID: "main.go", GenerationOrder: 2, IdempotencyToken: "c2", Mode: library.Replace,
		Rows: []library.Occurrence{codePartRow("d", "func delta() {}"), codePartRow("a", "func alpha() {}")},
	})

	for _, testCase := range []struct {
		namespace string
		owner     string
		want      library.OwnerOccurrences
	}{
		{namespace: "notes", owner: "note-1", want: library.OwnerOccurrences{ProjectionOrder: 1, Rows: []library.OwnerOccurrence{
			{RowKey: "n1", GenerationOrder: 1}, {RowKey: "n2", GenerationOrder: 1}, {RowKey: "n3", GenerationOrder: 2},
		}}},
		{namespace: "code", owner: "main.go", want: library.OwnerOccurrences{ProjectionOrder: 0, Rows: []library.OwnerOccurrence{
			{RowKey: "a", GenerationOrder: 2}, {RowKey: "d", GenerationOrder: 2},
		}}},
		{namespace: "code", owner: "unknown.go", want: library.OwnerOccurrences{}},
	} {
		got, err := opened.ListOwnerOccurrences(ctx, testCase.namespace, testCase.owner)
		if err != nil {
			t.Fatalf("ListOwnerOccurrences %s/%s: %v", testCase.namespace, testCase.owner, err)
		}
		state, err := opened.GetOwnerState(ctx, testCase.namespace, testCase.owner)
		if err != nil {
			t.Fatalf("GetOwnerState %s/%s: %v", testCase.namespace, testCase.owner, err)
		}
		testCase.want.State = state
		if !reflect.DeepEqual(got, testCase.want) {
			t.Fatalf("ListOwnerOccurrences %s/%s = %+v, want %+v", testCase.namespace, testCase.owner, got, testCase.want)
		}
	}

	if _, err := opened.ListOwnerOccurrences(ctx, "missing", "note-1"); !errors.Is(err, library.ErrInvalidRequest) {
		t.Fatalf("ListOwnerOccurrences on an unregistered namespace error = %v, want ErrInvalidRequest", err)
	}
}
