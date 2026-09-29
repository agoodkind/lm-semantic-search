package library_test

import (
	"context"
	"errors"
	"io/fs"
	"path/filepath"
	"strings"
	"testing"

	internalonnx "goodkind.io/lm-semantic-search/internal/embedding/onnx"
	"goodkind.io/lm-semantic-search/internal/offlinemodel"
	"goodkind.io/lm-semantic-search/library"
	"goodkind.io/lm-semantic-search/library/embedded"
	"goodkind.io/lm-semantic-search/library/embedding/onnx"
)

func embeddedDescriptor(t *testing.T, root string) library.StoreDescriptor {
	t.Helper()
	preset, err := offlinemodel.Resolve(offlinemodel.BGESmall)
	if err != nil {
		t.Fatalf("resolve %s: %v", offlinemodel.BGESmall, err)
	}
	return library.StoreDescriptor{
		CatalogPath:       filepath.Join(root, "catalog.sqlite"),
		LockPath:          filepath.Join(root, "catalog.lock"),
		PoolID:            "embedded",
		EmbeddingModel:    preset.Name,
		EmbeddingRevision: "test",
		Dimension:         int(preset.Dimension),
		Normalization:     "none",
	}
}

func embeddedPool(t *testing.T, root string) *embedded.Store {
	t.Helper()
	store, err := embedded.New(embedded.Config{Root: root})
	if err != nil {
		t.Fatalf("create embedded pool %s: %v", root, err)
	}
	return store
}

func offlineEmbedder(t *testing.T) library.Embedder {
	t.Helper()
	embedder, err := onnx.New(context.Background(), offlineONNXConfig(t))
	if errors.Is(err, internalonnx.ErrArtifactUnavailable) {
		t.Skipf("offline embedding artifact unavailable: %v", err)
	}
	if err != nil {
		t.Fatalf("create ONNX embedder: %v", err)
	}
	return embedder
}

func countVectorFiles(t *testing.T, root string) int {
	t.Helper()
	count := 0
	err := filepath.WalkDir(filepath.Join(root, "vectors"), func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !entry.IsDir() && strings.HasSuffix(path, ".vec") {
			count++
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk vector files: %v", err)
	}
	return count
}

func TestOpenOverTheEmbeddedPoolWritesAndVerifiesVectors(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	descriptor := embeddedDescriptor(t, filepath.Join(root, "catalog"))
	poolRoot := filepath.Join(root, "pool")
	embedder := offlineEmbedder(t)

	opened, err := library.Open(ctx, library.Config{Store: descriptor, Vectors: embeddedPool(t, poolRoot), Embedder: embedder})
	if err != nil {
		t.Fatalf("Open over the embedded pool: %v", err)
	}
	spec := library.NamespaceSpec{ID: "code", Policy: library.ReplaceAllowed, Scalars: nil}
	if err := opened.RegisterNamespace(ctx, spec); err != nil {
		t.Fatalf("RegisterNamespace: %v", err)
	}
	batch := library.Batch{
		Namespace:        "code",
		OwnerID:          "main.go",
		GenerationOrder:  1,
		IdempotencyToken: "g1",
		Mode:             library.Replace,
		Rows: []library.Occurrence{
			{RowKey: "a", SortKey: "a", SourceText: "func alpha() {}", SearchText: "func alpha() {}", EmbeddingInput: "func alpha() {}", Scalars: nil},
			{RowKey: "b", SortKey: "b", SourceText: "func beta() {}", SearchText: "func beta() {}", EmbeddingInput: "func beta() {}", Scalars: nil},
			{RowKey: "c", SortKey: "c", SourceText: "func alpha() {}", SearchText: "func alpha() {}", EmbeddingInput: "func alpha() {}", Scalars: nil},
		},
	}
	receipt, err := opened.Apply(ctx, batch)
	if err != nil {
		t.Fatalf("Apply over the embedded pool: %v", err)
	}
	state, err := opened.GetOwnerState(ctx, "code", "main.go")
	if err != nil || state.GenerationOrder != 1 || state.Fingerprint != receipt.Fingerprint {
		t.Fatalf("owner state = %+v, %v, want order 1 with fingerprint %s", state, err, receipt.Fingerprint)
	}
	if got := countVectorFiles(t, poolRoot); got != 2 {
		t.Fatalf("embedded pool has %d vector files for 2 distinct inputs, want 2", got)
	}
	if err := opened.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	_, err = library.Open(ctx, library.Config{Store: descriptor, Vectors: embeddedPool(t, filepath.Join(root, "other-pool")), Embedder: embedder})
	if !errors.Is(err, library.ErrStoreMismatch) {
		t.Fatalf("Open against another pool error = %v, want ErrStoreMismatch", err)
	}

	reopened, err := library.Open(ctx, library.Config{Store: descriptor, Vectors: embeddedPool(t, poolRoot), Embedder: embedder})
	if err != nil {
		t.Fatalf("reopen over the embedded pool: %v", err)
	}
	defer func() {
		if err := reopened.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	}()
	replay, err := reopened.Apply(ctx, batch)
	if err != nil || replay != receipt {
		t.Fatalf("replay after reopen = %+v, %v, want %+v", replay, err, receipt)
	}
}
