package usearch

import (
	"path/filepath"
	"slices"
	"testing"
)

func TestIndexRoundTrip(t *testing.T) {
	t.Parallel()

	index, err := New(2)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	t.Cleanup(index.Close)

	if err := index.Reserve(3); err != nil {
		t.Fatalf("Reserve returned error: %v", err)
	}
	if err := index.Add(11, []float32{1, 0}); err != nil {
		t.Fatalf("Add first vector returned error: %v", err)
	}
	if err := index.Add(22, []float32{0, 1}); err != nil {
		t.Fatalf("Add second vector returned error: %v", err)
	}

	contains, err := index.Contains(11)
	if err != nil {
		t.Fatalf("Contains returned error: %v", err)
	}
	if !contains {
		t.Fatal("Contains returned false for an indexed key")
	}
	size, err := index.Size()
	if err != nil {
		t.Fatalf("Size returned error: %v", err)
	}
	if size != 2 {
		t.Fatalf("Size = %d, want 2", size)
	}

	keys, distances, err := index.Search([]float32{1, 0}, 2)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	if !slices.Equal(keys, []uint64{11, 22}) {
		t.Fatalf("Search keys = %v, want [11 22]", keys)
	}
	if len(distances) != 2 || distances[0] != 0 {
		t.Fatalf("Search distances = %v, want first distance 0", distances)
	}

	path := filepath.Join(t.TempDir(), "index.usearch")
	if err := index.Save(path); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	loaded, err := New(1)
	if err != nil {
		t.Fatalf("New loaded index returned error: %v", err)
	}
	t.Cleanup(loaded.Close)
	if err := loaded.Load(path); err != nil {
		t.Fatalf("Load returned error: %v", err)
	}
	dimensions, err := loaded.Dimensions()
	if err != nil {
		t.Fatalf("Dimensions returned error: %v", err)
	}
	if dimensions != 2 {
		t.Fatalf("Dimensions = %d, want 2", dimensions)
	}
	loadedKeys, _, err := loaded.Search([]float32{0, 1}, 1)
	if err != nil {
		t.Fatalf("loaded Search returned error: %v", err)
	}
	if !slices.Equal(loadedKeys, []uint64{22}) {
		t.Fatalf("loaded Search keys = %v, want [22]", loadedKeys)
	}

	viewed, err := New(1)
	if err != nil {
		t.Fatalf("New viewed index returned error: %v", err)
	}
	t.Cleanup(viewed.Close)
	if err := viewed.View(path); err != nil {
		t.Fatalf("View returned error: %v", err)
	}
	viewedSize, err := viewed.Size()
	if err != nil {
		t.Fatalf("viewed Size returned error: %v", err)
	}
	if viewedSize != 2 {
		t.Fatalf("viewed Size = %d, want 2", viewedSize)
	}

	removed, err := index.Remove(11)
	if err != nil {
		t.Fatalf("Remove returned error: %v", err)
	}
	if !removed {
		t.Fatal("Remove returned false for an indexed key")
	}
}
